#!/usr/bin/env python3
"""
Convergence Analysis for Physics-Informed Neural Networks.

This script performs a rigorous scientific study of how PINN error scales with:
1. Number of collocation points (training data) - Expected: O(1/N) decay
2. Network size (parameters) - Network capacity vs approximation error
3. Training epochs - Learning curve analysis

Key insights validated:
- Error decays as O(1/N) for collocation points (law of large numbers/Monte Carlo theory)
- Network capacity affects approximation error floor
- Diminishing returns after sufficient training

Usage:
    python scripts/convergence_study.py --quick   # Fast test (~5 min)
    python scripts/convergence_study.py --full    # Complete study (~30 min)

References:
- Raissi et al., "Physics-informed neural networks" (2019)
- Wang et al., "When and why PINNs fail to train" (2021)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classical.pinn import PINN, PINNTrainer
from src.pde.black_scholes import BSParams, bs_analytical
from src.data.collocation import generate_collocation_points

# Fixed random seed for reproducibility
SEED = 42


@dataclass
class ConvergenceConfig:
    """Configuration for convergence study."""
    # Black-Scholes parameters (standard test case)
    r: float = 0.05
    sigma: float = 0.2
    K: float = 100.0
    T: float = 1.0
    S_max: float = 200.0
    
    # Collocation point study
    n_interior_values: list[int] = field(
        default_factory=lambda: [100, 500, 1000, 2000, 5000, 10000]
    )
    
    # Network size study
    hidden_layers_values: list[int] = field(default_factory=lambda: [2, 3, 4])
    hidden_units_values: list[int] = field(default_factory=lambda: [32, 64, 128])
    
    # Training epochs study
    max_epochs: int = 10000
    record_every: int = 100
    
    # Training parameters for each study
    epochs_per_run: int = 100  # Fast runs for scaling studies
    collocation_epochs: int = 500  # More epochs for collocation study
    learning_rate: float = 1e-3
    n_test_points: int = 200
    
    # Default network for collocation study
    default_hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64])
    
    # Default collocation for network study
    default_n_interior: int = 1000


@dataclass
class StudyResults:
    """Results from convergence study."""
    # Collocation study results
    collocation_n_values: list[int] = field(default_factory=list)
    collocation_mse_values: list[float] = field(default_factory=list)
    collocation_fit_slope: Optional[float] = None
    collocation_fit_intercept: Optional[float] = None
    
    # Network size study results
    network_configs: list[dict] = field(default_factory=list)
    network_param_counts: list[int] = field(default_factory=list)
    network_mse_values: list[float] = field(default_factory=list)
    
    # Training epochs study results
    epoch_values: list[int] = field(default_factory=list)
    epoch_mse_values: list[float] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    study_times: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mse_vs_analytical(
    model: PINN,
    params: BSParams,
    n_test: int,
    device: torch.device,
) -> float:
    """
    Compute MSE between PINN prediction and analytical Black-Scholes.
    
    Evaluates on a test grid at t=0 (current time) for various spot prices.
    """
    model.eval()
    
    # Test grid: evaluate at t=0 for S in [10, 200]
    # Avoid S=0 where option value is exactly 0
    S_test = torch.linspace(10.0, params.K * 2, n_test, device=device)
    t_test = torch.zeros_like(S_test)
    
    with torch.no_grad():
        V_pinn = model(S_test, t_test)
    
    V_analytical = bs_analytical(S_test, t_test, params)
    
    mse = ((V_pinn - V_analytical) ** 2).mean().item()
    return mse


def train_pinn_fast(
    hidden_dims: list[int],
    n_interior: int,
    n_epochs: int,
    params: BSParams,
    device: torch.device,
    lr: float = 1e-3,
    record_every: Optional[int] = None,
    verbose: bool = False,
) -> tuple[PINN, list[tuple[int, float]]]:
    """
    Train a PINN with specified configuration.
    
    Returns the trained model and optionally a list of (epoch, mse) tuples.
    """
    set_seed(SEED)
    
    model = PINN(
        hidden_dims=hidden_dims,
        S_max=params.K * 2,  # S_max = 2 * strike
        T_max=params.T,
    ).to(device)
    
    trainer = PINNTrainer(
        model=model,
        params=params,
        lr=lr,
        lambda_pde=1.0,
        lambda_bc=10.0,
        lambda_ic=10.0,
    )
    
    mse_history = []
    
    for epoch in range(n_epochs):
        # Generate fresh collocation points each epoch
        S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
            n_interior=n_interior,
            n_boundary=200,
            n_terminal=200,
            S_max=model.S_max,
            T=params.T,
        )
        
        # Move to device
        S_int = S_int.to(device)
        t_int = t_int.to(device)
        S_bc = S_bc.to(device)
        t_bc = t_bc.to(device)
        S_term = S_term.to(device)
        
        losses = trainer.train_step(S_int, t_int, S_bc, t_bc, S_term)
        
        # Record MSE if requested
        if record_every is not None and (epoch + 1) % record_every == 0:
            mse = compute_mse_vs_analytical(model, params, 200, device)
            mse_history.append((epoch + 1, mse))
            if verbose:
                print(f"  Epoch {epoch + 1}: MSE = {mse:.6e}")
    
    return model, mse_history


def run_collocation_study(
    config: ConvergenceConfig,
    device: torch.device,
) -> tuple[list[int], list[float], float, float]:
    """
    Study 1: How does MSE scale with number of collocation points?
    
    Theory: Error should decay as O(1/N) due to law of large numbers.
    In practice, this convergence rate is observed when the network has
    sufficient capacity and training has converged.
    
    Note: If the network hasn't converged (training dominated regime),
    adding more collocation points won't help much. This is why we use
    more epochs for this study.
    """
    print("\n" + "=" * 60)
    print("Study 1: Collocation Point Convergence")
    print("=" * 60)
    print("Theory: Error ~ O(1/N) (law of large numbers)")
    print(f"Training each model for {config.collocation_epochs} epochs")
    print()
    
    params = BSParams(r=config.r, sigma=config.sigma, K=config.K, T=config.T)
    
    n_values = config.n_interior_values
    mse_values = []
    
    for n_interior in n_values:
        print(f"Training with n_interior = {n_interior}...", end=" ", flush=True)
        
        model, _ = train_pinn_fast(
            hidden_dims=config.default_hidden_dims,
            n_interior=n_interior,
            n_epochs=config.collocation_epochs,  # Use more epochs
            params=params,
            device=device,
            lr=config.learning_rate,
        )
        
        mse = compute_mse_vs_analytical(model, params, config.n_test_points, device)
        mse_values.append(mse)
        print(f"MSE = {mse:.6e}")
    
    # Fit log-log line: log(MSE) = m * log(N) + b
    # Expected: m ≈ -1 (O(1/N) scaling)
    log_n = np.log(n_values)
    log_mse = np.log(mse_values)
    slope, intercept, r_value, _, _ = stats.linregress(log_n, log_mse)
    
    print(f"\nLog-log fit: MSE ~ N^{slope:.3f}")
    print(f"R^2 = {r_value**2:.4f}")
    print(f"Expected slope: -1 (O(1/N) convergence)")
    
    # Interpretation
    if slope > -0.3:
        print("\nNote: Weak scaling suggests training-dominated regime.")
        print("      Network may need more epochs to converge.")
    elif slope < -0.7:
        print("\nNote: Strong scaling - approaching theoretical O(1/N) rate.")
    
    return n_values, mse_values, slope, intercept


def run_network_size_study(
    config: ConvergenceConfig,
    device: torch.device,
) -> tuple[list[dict], list[int], list[float]]:
    """
    Study 2: How does MSE scale with network size?
    
    Explores the trade-off between approximation capacity and generalization.
    Larger networks can approximate more complex functions but may overfit
    or be harder to optimize.
    """
    print("\n" + "=" * 60)
    print("Study 2: Network Size vs Approximation Error")
    print("=" * 60)
    print("Varying hidden layers and units, fixed collocation points")
    print(f"Training each model for {config.epochs_per_run} epochs")
    print()
    
    params = BSParams(r=config.r, sigma=config.sigma, K=config.K, T=config.T)
    
    configs = []
    param_counts = []
    mse_values = []
    
    for n_layers in config.hidden_layers_values:
        for n_units in config.hidden_units_values:
            hidden_dims = [n_units] * n_layers
            
            print(f"Training: {n_layers} layers x {n_units} units...", end=" ", flush=True)
            
            model, _ = train_pinn_fast(
                hidden_dims=hidden_dims,
                n_interior=config.default_n_interior,
                n_epochs=config.epochs_per_run,
                params=params,
                device=device,
                lr=config.learning_rate,
            )
            
            n_params = count_parameters(model)
            mse = compute_mse_vs_analytical(model, params, config.n_test_points, device)
            
            configs.append({"layers": n_layers, "units": n_units})
            param_counts.append(n_params)
            mse_values.append(mse)
            
            print(f"params = {n_params:,}, MSE = {mse:.6e}")
    
    # Find best configuration
    best_idx = np.argmin(mse_values)
    best_config = configs[best_idx]
    print(f"\nBest config: {best_config['layers']}x{best_config['units']} "
          f"({param_counts[best_idx]:,} params, MSE={mse_values[best_idx]:.6e})")
    
    return configs, param_counts, mse_values


def run_training_epochs_study(
    config: ConvergenceConfig,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """
    Study 3: Learning curve - how does MSE evolve during training?
    
    Records MSE at regular intervals to show convergence behavior.
    Key insight: Diminishing returns - most improvement happens early.
    """
    print("\n" + "=" * 60)
    print("Study 3: Training Epochs Learning Curve")
    print("=" * 60)
    print(f"Training for {config.max_epochs} epochs, recording every {config.record_every}")
    print()
    
    params = BSParams(r=config.r, sigma=config.sigma, K=config.K, T=config.T)
    
    _, mse_history = train_pinn_fast(
        hidden_dims=config.default_hidden_dims,
        n_interior=config.default_n_interior,
        n_epochs=config.max_epochs,
        params=params,
        device=device,
        lr=config.learning_rate,
        record_every=config.record_every,
        verbose=True,
    )
    
    epoch_values = [e for e, _ in mse_history]
    mse_values = [m for _, m in mse_history]
    
    return epoch_values, mse_values


def create_plots(results: StudyResults, output_dir: Path) -> None:
    """Create and save convergence analysis plots."""
    # Use style that doesn't require special fonts
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: MSE vs Collocation Points (log-log)
    ax1 = axes[0]
    ax1.loglog(results.collocation_n_values, results.collocation_mse_values, 
               "o-", linewidth=2, markersize=8, color="blue", label="Measured MSE")
    
    # Add theoretical O(1/N) reference line
    if results.collocation_fit_slope is not None:
        n_ref = np.array(results.collocation_n_values)
        mse_fit = np.exp(results.collocation_fit_intercept) * n_ref ** results.collocation_fit_slope
        ax1.loglog(n_ref, mse_fit, "--", color="red", linewidth=2, 
                   label=f"Fit: MSE ~ N^{results.collocation_fit_slope:.2f}")
        
        # Add O(1/N) reference
        mse_ref = mse_fit[0] * (n_ref[0] / n_ref)
        ax1.loglog(n_ref, mse_ref, ":", color="gray", linewidth=1.5, 
                   label="Reference: O(1/N)")
    
    ax1.set_xlabel("Number of Collocation Points (N)")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Collocation Point Convergence")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # Plot 2: MSE vs Network Parameters
    ax2 = axes[1]
    
    # Color by number of layers
    unique_layers = sorted(set(c["layers"] for c in results.network_configs))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_layers)))
    layer_colors = {l: c for l, c in zip(unique_layers, colors)}
    
    for config, n_params, mse in zip(results.network_configs, 
                                      results.network_param_counts, 
                                      results.network_mse_values):
        color = layer_colors[config["layers"]]
        ax2.scatter(n_params, mse, c=[color], s=100, edgecolors="black", linewidth=1)
        ax2.annotate(f"{config['layers']}x{config['units']}", 
                     (n_params, mse), textcoords="offset points", 
                     xytext=(5, 5), fontsize=8)
    
    # Legend for layers
    for n_layers, color in layer_colors.items():
        ax2.scatter([], [], c=[color], s=100, label=f"{n_layers} layers", 
                    edgecolors="black", linewidth=1)
    
    ax2.set_xlabel("Number of Parameters")
    ax2.set_ylabel("Mean Squared Error")
    ax2.set_title("Network Capacity vs Error")
    ax2.legend()
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Curve
    ax3 = axes[2]
    ax3.semilogy(results.epoch_values, results.epoch_mse_values, 
                 "-", linewidth=2, color="blue")
    ax3.fill_between(results.epoch_values, results.epoch_mse_values, 
                     alpha=0.2, color="blue")
    
    ax3.set_xlabel("Training Epochs")
    ax3.set_ylabel("Mean Squared Error")
    ax3.set_title("Training Learning Curve")
    ax3.grid(True, alpha=0.3)
    
    # Mark final MSE
    final_epoch = results.epoch_values[-1]
    final_mse = results.epoch_mse_values[-1]
    ax3.axhline(y=final_mse, color="red", linestyle="--", alpha=0.7)
    ax3.annotate(f"Final: {final_mse:.2e}", xy=(final_epoch * 0.6, final_mse * 1.5),
                 fontsize=10, color="red")
    
    plt.tight_layout()
    
    plot_path = output_dir / "convergence_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots saved to: {plot_path}")


def print_results_table(results: StudyResults) -> None:
    """Print a summary table of all results."""
    print("\n" + "=" * 70)
    print("CONVERGENCE STUDY RESULTS SUMMARY")
    print("=" * 70)
    
    # Collocation study table
    print("\n1. Collocation Point Convergence")
    print("-" * 40)
    print(f"{'N points':<12} {'MSE':<15}")
    print("-" * 40)
    for n, mse in zip(results.collocation_n_values, results.collocation_mse_values):
        print(f"{n:<12} {mse:<15.6e}")
    if results.collocation_fit_slope is not None:
        print("-" * 40)
        print(f"Empirical scaling: MSE ~ N^{results.collocation_fit_slope:.3f}")
        print(f"Theoretical: MSE ~ N^-1 (O(1/N))")
    
    # Network size table
    print("\n2. Network Size Study")
    print("-" * 50)
    print(f"{'Config':<15} {'Parameters':<12} {'MSE':<15}")
    print("-" * 50)
    for config, n_params, mse in zip(results.network_configs,
                                      results.network_param_counts,
                                      results.network_mse_values):
        config_str = f"{config['layers']}x{config['units']}"
        print(f"{config_str:<15} {n_params:<12,} {mse:<15.6e}")
    
    # Training epochs summary
    print("\n3. Training Epochs Study")
    print("-" * 40)
    print(f"Total epochs: {results.epoch_values[-1]}")
    print(f"Initial MSE:  {results.epoch_mse_values[0]:.6e}")
    print(f"Final MSE:    {results.epoch_mse_values[-1]:.6e}")
    print(f"Improvement:  {results.epoch_mse_values[0] / results.epoch_mse_values[-1]:.1f}x")
    
    # Timing
    print("\n" + "=" * 70)
    print("TIMING")
    print("-" * 40)
    for study, duration in results.study_times.items():
        print(f"{study}: {duration:.1f}s")
    print(f"Total: {results.total_time_seconds:.1f}s")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PINN Convergence Analysis Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with reduced parameters (~2-5 min)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full study with all parameters (~30-60 min)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/convergence",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto'",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    return parser.parse_args()


def get_config(args: argparse.Namespace) -> ConvergenceConfig:
    """Get configuration based on command-line arguments."""
    if args.quick:
        return ConvergenceConfig(
            n_interior_values=[100, 500, 1000, 2000],
            hidden_layers_values=[2, 3],
            hidden_units_values=[32, 64],
            max_epochs=1000,
            record_every=50,
            epochs_per_run=100,
            collocation_epochs=200,  # More epochs for collocation study
            n_test_points=100,
        )
    elif args.full:
        return ConvergenceConfig(
            n_interior_values=[100, 500, 1000, 2000, 5000, 10000],
            hidden_layers_values=[2, 3, 4],
            hidden_units_values=[32, 64, 128],
            max_epochs=10000,
            record_every=100,
            epochs_per_run=200,
            collocation_epochs=500,  # More epochs for collocation study
            n_test_points=200,
        )
    else:
        # Default: moderate settings
        return ConvergenceConfig(
            n_interior_values=[100, 500, 1000, 2000, 5000],
            hidden_layers_values=[2, 3, 4],
            hidden_units_values=[32, 64, 128],
            max_epochs=5000,
            record_every=100,
            epochs_per_run=200,
            collocation_epochs=500,
            n_test_points=200,
        )


def setup_device(device_arg: str) -> torch.device:
    """Set up computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def main():
    """Main entry point for convergence study."""
    args = parse_args()
    device = setup_device(args.device)
    config = get_config(args)
    
    print("\n" + "=" * 70)
    print("PINN CONVERGENCE ANALYSIS STUDY")
    print("=" * 70)
    print(f"\nMode: {'quick' if args.quick else 'full' if args.full else 'default'}")
    print(f"Output: {args.output_dir}")
    print("\nBlack-Scholes Parameters:")
    print(f"  r = {config.r}, sigma = {config.sigma}, K = {config.K}, T = {config.T}")
    
    results = StudyResults()
    start_time = time.time()
    
    # Study 1: Collocation point convergence
    study_start = time.time()
    n_values, mse_values, slope, intercept = run_collocation_study(config, device)
    results.collocation_n_values = n_values
    results.collocation_mse_values = mse_values
    results.collocation_fit_slope = float(slope)
    results.collocation_fit_intercept = float(intercept)
    results.study_times["collocation"] = time.time() - study_start
    
    # Study 2: Network size
    study_start = time.time()
    configs, param_counts, mse_values = run_network_size_study(config, device)
    results.network_configs = configs
    results.network_param_counts = param_counts
    results.network_mse_values = mse_values
    results.study_times["network_size"] = time.time() - study_start
    
    # Study 3: Training epochs
    study_start = time.time()
    epoch_values, mse_values = run_training_epochs_study(config, device)
    results.epoch_values = epoch_values
    results.epoch_mse_values = mse_values
    results.study_times["training_epochs"] = time.time() - study_start
    
    results.total_time_seconds = time.time() - start_time
    
    # Print results
    print_results_table(results)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "convergence_study.json"
    with open(results_path, "w") as f:
        json.dump(asdict(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate plots
    if not args.no_plots:
        create_plots(results, output_dir)
    
    print("\nConvergence study complete!")


if __name__ == "__main__":
    main()
