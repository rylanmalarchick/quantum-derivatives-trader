#!/usr/bin/env python3
"""
Benchmarking script comparing all option pricing methods.

Compares:
    - Analytical Black-Scholes
    - Classical PINN
    - Hybrid Quantum-Classical PINN
    - Monte Carlo simulation
    - Finite Difference methods

Measures:
    - Pricing accuracy (vs analytical)
    - Greeks accuracy (delta, gamma)
    - Computation time

Outputs:
    - Comparison plots
    - Results in JSON and CSV format

Usage:
    python benchmark.py --pinn_epochs 3000 --hybrid_epochs 500
    python benchmark.py --skip_hybrid  # Skip slow quantum simulation
    python benchmark.py --mc_paths 100000 --fd_grid_size 200
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical.pinn import PINN, PINNTrainer
from quantum.hybrid_pinn import HybridPINN
from pde.black_scholes import BSParams, bs_analytical, bs_delta, bs_gamma
from pricing.analytical import AnalyticalPricer
from pricing.monte_carlo import MonteCarloEngine
from pricing.finite_difference import FiniteDifferencePricer, FDGrid
from data.collocation import create_grid, generate_collocation_points
from classical.losses import PINNLoss
from utils.visualization import set_publication_style


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Black-Scholes parameters
    r: float = 0.05
    sigma: float = 0.2
    K: float = 100.0
    T: float = 1.0
    S_max: float = 200.0

    # Test points
    n_test_spots: int = 100
    test_time: float = 0.0  # Evaluate at t=0

    # PINN config
    pinn_hidden_dims: tuple = (64, 64, 64, 64)
    pinn_epochs: int = 3000
    pinn_lr: float = 1e-3
    pinn_n_interior: int = 1000
    pinn_n_boundary: int = 200
    pinn_n_terminal: int = 200

    # Hybrid PINN config
    hybrid_n_qubits: int = 4
    hybrid_n_layers: int = 3
    hybrid_classical_hidden: int = 32
    hybrid_epochs: int = 500
    hybrid_lr: float = 5e-3
    hybrid_n_interior: int = 200
    hybrid_n_boundary: int = 50
    hybrid_n_terminal: int = 50

    # Monte Carlo config
    mc_n_paths: int = 100000
    mc_seed: int = 42

    # Finite Difference config
    fd_n_S: int = 200
    fd_n_t: int = 2000


@dataclass
class MethodResult:
    """Results from a single pricing method."""
    name: str
    prices: np.ndarray
    delta: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    computation_time_seconds: float = 0.0
    training_time_seconds: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    max_error: float = 0.0
    delta_mse: float = 0.0
    gamma_mse: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Option Pricing Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Black-Scholes parameters
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--S_max", type=float, default=200.0, help="Max spot price")

    # PINN configuration
    parser.add_argument("--pinn_epochs", type=int, default=3000, help="PINN training epochs")
    parser.add_argument("--pinn_lr", type=float, default=1e-3, help="PINN learning rate")

    # Hybrid PINN configuration
    parser.add_argument("--hybrid_epochs", type=int, default=500, help="Hybrid PINN training epochs")
    parser.add_argument("--hybrid_n_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--hybrid_n_layers", type=int, default=3, help="Number of quantum layers")
    parser.add_argument("--skip_hybrid", action="store_true", help="Skip hybrid PINN (slow)")

    # Monte Carlo configuration
    parser.add_argument("--mc_paths", type=int, default=100000, help="Monte Carlo paths")
    parser.add_argument("--mc_seed", type=int, default=42, help="Random seed for MC")

    # Finite Difference configuration
    parser.add_argument("--fd_grid_size", type=int, default=200, help="FD grid size")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark", help="Output directory")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device for PINN training")

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    return device


def benchmark_analytical(
    config: BenchmarkConfig,
    S_test: np.ndarray,
) -> MethodResult:
    """Benchmark analytical Black-Scholes."""
    pricer = AnalyticalPricer(r=config.r, sigma=config.sigma)

    start = time.time()
    prices = pricer.black_scholes(S_test, config.K, config.T - config.test_time)
    delta = pricer.delta(S_test, config.K, config.T - config.test_time)
    gamma = pricer.gamma(S_test, config.K, config.T - config.test_time)
    comp_time = time.time() - start

    return MethodResult(
        name="Analytical (BS)",
        prices=prices,
        delta=delta,
        gamma=gamma,
        computation_time_seconds=comp_time,
    )


def benchmark_monte_carlo(
    config: BenchmarkConfig,
    S_test: np.ndarray,
    analytical_prices: np.ndarray,
) -> MethodResult:
    """Benchmark Monte Carlo pricing."""
    mc = MonteCarloEngine(r=config.r, sigma=config.sigma, seed=config.mc_seed)

    prices = []
    start = time.time()
    for S0 in S_test:
        result = mc.price_european(
            payoff_fn=lambda s: max(s - config.K, 0),
            S0=float(S0),
            T=config.T - config.test_time,
            n_paths=config.mc_n_paths,
        )
        prices.append(result.price)
    comp_time = time.time() - start
    prices = np.array(prices)

    # Compute errors
    mse = float(np.mean((prices - analytical_prices) ** 2))
    mae = float(np.mean(np.abs(prices - analytical_prices)))
    max_error = float(np.max(np.abs(prices - analytical_prices)))

    return MethodResult(
        name="Monte Carlo",
        prices=prices,
        computation_time_seconds=comp_time,
        mse=mse,
        mae=mae,
        max_error=max_error,
    )


def benchmark_finite_difference(
    config: BenchmarkConfig,
    S_test: np.ndarray,
    analytical_prices: np.ndarray,
    analytical_delta: np.ndarray,
    analytical_gamma: np.ndarray,
) -> MethodResult:
    """Benchmark finite difference methods."""
    grid = FDGrid(S_min=0.0, S_max=config.S_max, n_S=config.fd_n_S, n_t=config.fd_n_t)
    fd = FiniteDifferencePricer(r=config.r, sigma=config.sigma, grid=grid)

    start = time.time()
    greeks = fd.compute_greeks(config.K, config.T - config.test_time, option_type="call")
    comp_time = time.time() - start

    # Interpolate to test points
    S_fd = greeks["S"]
    prices = np.interp(S_test, S_fd, greeks["V"])
    delta = np.interp(S_test, S_fd, greeks["delta"])
    gamma = np.interp(S_test, S_fd, greeks["gamma"])

    # Compute errors
    mse = float(np.mean((prices - analytical_prices) ** 2))
    mae = float(np.mean(np.abs(prices - analytical_prices)))
    max_error = float(np.max(np.abs(prices - analytical_prices)))
    delta_mse = float(np.mean((delta - analytical_delta) ** 2))
    gamma_mse = float(np.mean((gamma - analytical_gamma) ** 2))

    return MethodResult(
        name="Finite Difference (CN)",
        prices=prices,
        delta=delta,
        gamma=gamma,
        computation_time_seconds=comp_time,
        mse=mse,
        mae=mae,
        max_error=max_error,
        delta_mse=delta_mse,
        gamma_mse=gamma_mse,
    )


def benchmark_pinn(
    config: BenchmarkConfig,
    S_test: np.ndarray,
    analytical_prices: np.ndarray,
    analytical_delta: np.ndarray,
    analytical_gamma: np.ndarray,
    device: torch.device,
    quiet: bool = False,
) -> MethodResult:
    """Benchmark classical PINN."""
    bs_params = BSParams(r=config.r, sigma=config.sigma, K=config.K, T=config.T)

    model = PINN(
        hidden_dims=list(config.pinn_hidden_dims),
        S_max=config.S_max,
        T_max=config.T,
        use_residual=False,
    ).to(device)

    trainer = PINNTrainer(
        model=model,
        params=bs_params,
        lr=config.pinn_lr,
    )

    # Train
    train_start = time.time()
    trainer.train(
        n_epochs=config.pinn_epochs,
        n_interior=config.pinn_n_interior,
        n_boundary=config.pinn_n_boundary,
        n_terminal=config.pinn_n_terminal,
        print_every=config.pinn_epochs // 10 if not quiet else config.pinn_epochs + 1,
    )
    train_time = time.time() - train_start

    # Evaluate
    model.eval()
    S_tensor = torch.tensor(S_test, dtype=torch.float32, device=device)
    t_tensor = torch.full_like(S_tensor, config.test_time)

    eval_start = time.time()
    with torch.no_grad():
        prices = model(S_tensor, t_tensor).cpu().numpy()

    # Get Greeks via autodiff
    greeks = model.predict_with_greeks(S_tensor, t_tensor)
    delta = greeks["delta"].cpu().numpy()
    gamma = greeks["gamma"].cpu().numpy()
    comp_time = time.time() - eval_start

    # Compute errors
    mse = float(np.mean((prices - analytical_prices) ** 2))
    mae = float(np.mean(np.abs(prices - analytical_prices)))
    max_error = float(np.max(np.abs(prices - analytical_prices)))
    delta_mse = float(np.mean((delta - analytical_delta) ** 2))
    gamma_mse = float(np.mean((gamma - analytical_gamma) ** 2))

    return MethodResult(
        name="Classical PINN",
        prices=prices,
        delta=delta,
        gamma=gamma,
        computation_time_seconds=comp_time,
        training_time_seconds=train_time,
        mse=mse,
        mae=mae,
        max_error=max_error,
        delta_mse=delta_mse,
        gamma_mse=gamma_mse,
    )


def benchmark_hybrid_pinn(
    config: BenchmarkConfig,
    S_test: np.ndarray,
    analytical_prices: np.ndarray,
    device: torch.device,
    quiet: bool = False,
) -> MethodResult:
    """Benchmark hybrid quantum-classical PINN."""
    bs_params = BSParams(r=config.r, sigma=config.sigma, K=config.K, T=config.T)

    model = HybridPINN(
        n_qubits=config.hybrid_n_qubits,
        n_layers=config.hybrid_n_layers,
        classical_hidden=config.hybrid_classical_hidden,
        S_max=config.S_max,
        T_max=config.T,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.hybrid_lr)
    loss_fn = PINNLoss(bs_params)

    # Train
    train_start = time.time()
    for epoch in range(config.hybrid_epochs):
        optimizer.zero_grad()

        S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
            n_interior=config.hybrid_n_interior,
            n_boundary=config.hybrid_n_boundary,
            n_terminal=config.hybrid_n_terminal,
            S_max=config.S_max,
            T=config.T,
        )

        losses = loss_fn(model, S_int, t_int, S_bc, t_bc, S_term)
        losses["total"].backward()
        optimizer.step()

        if not quiet and epoch % (config.hybrid_epochs // 10) == 0:
            print(f"Hybrid Epoch {epoch}: loss={losses['total'].item():.6f}")

    train_time = time.time() - train_start

    # Evaluate
    model.eval()
    S_tensor = torch.tensor(S_test, dtype=torch.float32, device=device)
    t_tensor = torch.full_like(S_tensor, config.test_time)

    eval_start = time.time()
    with torch.no_grad():
        prices = model(S_tensor, t_tensor).cpu().numpy()
    comp_time = time.time() - eval_start

    # Compute errors
    mse = float(np.mean((prices - analytical_prices) ** 2))
    mae = float(np.mean(np.abs(prices - analytical_prices)))
    max_error = float(np.max(np.abs(prices - analytical_prices)))

    return MethodResult(
        name="Hybrid PINN",
        prices=prices,
        computation_time_seconds=comp_time,
        training_time_seconds=train_time,
        mse=mse,
        mae=mae,
        max_error=max_error,
    )


def create_comparison_plots(
    S_test: np.ndarray,
    results: Dict[str, MethodResult],
    output_dir: Path,
) -> None:
    """Create comparison visualization plots."""
    set_publication_style()

    # Get analytical as reference
    analytical = results["Analytical (BS)"]

    # 1. Price comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for name, result in results.items():
        linestyle = '-' if name == "Analytical (BS)" else '--'
        alpha = 1.0 if name == "Analytical (BS)" else 0.8
        ax1.plot(S_test, result.prices, linestyle=linestyle, label=name, alpha=alpha, linewidth=2)

    ax1.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax1.set_ylabel('Option Value ($V$)', fontsize=12)
    ax1.set_title('Price Comparison Across Methods', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Errors subplot
    ax2 = axes[1]
    for name, result in results.items():
        if name != "Analytical (BS)":
            error = np.abs(result.prices - analytical.prices)
            ax2.semilogy(S_test, error + 1e-10, label=name, linewidth=1.5)

    ax2.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax2.set_ylabel('Absolute Error (log scale)', fontsize=12)
    ax2.set_title('Pricing Error vs Analytical', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "price_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Greeks comparison (for methods that compute them)
    methods_with_delta = [name for name, r in results.items() if r.delta is not None]

    if len(methods_with_delta) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Delta
        ax1 = axes[0]
        for name in methods_with_delta:
            result = results[name]
            linestyle = '-' if name == "Analytical (BS)" else '--'
            ax1.plot(S_test, result.delta, linestyle=linestyle, label=name, linewidth=2)

        ax1.set_xlabel('Spot Price ($S$)', fontsize=12)
        ax1.set_ylabel('Delta ($\\Delta$)', fontsize=12)
        ax1.set_title('Delta Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Gamma
        ax2 = axes[1]
        for name in methods_with_delta:
            result = results[name]
            if result.gamma is not None:
                linestyle = '-' if name == "Analytical (BS)" else '--'
                ax2.plot(S_test, result.gamma, linestyle=linestyle, label=name, linewidth=2)

        ax2.set_xlabel('Spot Price ($S$)', fontsize=12)
        ax2.set_ylabel('Gamma ($\\Gamma$)', fontsize=12)
        ax2.set_title('Gamma Comparison', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / "greeks_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Timing comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    method_names = [name for name in results.keys() if name != "Analytical (BS)"]

    # Computation time
    ax1 = axes[0]
    comp_times = [results[name].computation_time_seconds for name in method_names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    bars1 = ax1.bar(method_names, comp_times, color=colors)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Inference/Computation Time', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars1, comp_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.4f}s',
                ha='center', va='bottom', fontsize=9)

    # Training time (for learning methods)
    ax2 = axes[1]
    train_times = [results[name].training_time_seconds for name in method_names]
    bars2 = ax2.bar(method_names, train_times, color=colors)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Training Time (PINN methods only)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars2, train_times):
        if t > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.1f}s',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "timing_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Accuracy bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = [name for name in results.keys() if name != "Analytical (BS)"]
    mse_values = [results[name].mse for name in method_names]
    mae_values = [results[name].mae for name in method_names]

    x = np.arange(len(method_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_values, width, label='MSE', color='steelblue')
    bars2 = ax.bar(x + width/2, mae_values, width, label='MAE', color='coral')

    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Pricing Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to: {output_dir}")


def save_results(
    results: Dict[str, MethodResult],
    config: BenchmarkConfig,
    output_dir: Path,
) -> None:
    """Save benchmark results to JSON and CSV."""

    # JSON output
    json_data = {
        "config": asdict(config),
        "results": {},
    }
    for name, result in results.items():
        json_data["results"][name] = {
            "mse": result.mse,
            "mae": result.mae,
            "max_error": result.max_error,
            "delta_mse": result.delta_mse,
            "gamma_mse": result.gamma_mse,
            "computation_time_seconds": result.computation_time_seconds,
            "training_time_seconds": result.training_time_seconds,
        }

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # CSV output
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "MSE", "MAE", "Max Error",
            "Delta MSE", "Gamma MSE",
            "Computation Time (s)", "Training Time (s)"
        ])
        for name, result in results.items():
            writer.writerow([
                name,
                f"{result.mse:.8f}",
                f"{result.mae:.8f}",
                f"{result.max_error:.8f}",
                f"{result.delta_mse:.8f}",
                f"{result.gamma_mse:.8f}",
                f"{result.computation_time_seconds:.6f}",
                f"{result.training_time_seconds:.2f}",
            ])

    print(f"Results saved to: {json_path}")
    print(f"Results saved to: {csv_path}")


def print_summary_table(results: Dict[str, MethodResult]) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Header
    print(f"{'Method':<25} {'MSE':<12} {'MAE':<12} {'Max Err':<12} {'Comp Time':<12} {'Train Time':<12}")
    print("-" * 100)

    for name, result in results.items():
        comp_time = f"{result.computation_time_seconds:.4f}s"
        train_time = f"{result.training_time_seconds:.1f}s" if result.training_time_seconds > 0 else "N/A"
        print(f"{name:<25} {result.mse:<12.6f} {result.mae:<12.6f} {result.max_error:<12.6f} {comp_time:<12} {train_time:<12}")

    print("=" * 100)


def main():
    """Main benchmark function."""
    args = parse_args()
    device = setup_device(args.device)

    # Create configuration
    config = BenchmarkConfig(
        r=args.r,
        sigma=args.sigma,
        K=args.K,
        T=args.T,
        S_max=args.S_max,
        pinn_epochs=args.pinn_epochs,
        pinn_lr=args.pinn_lr,
        hybrid_epochs=args.hybrid_epochs,
        hybrid_n_qubits=args.hybrid_n_qubits,
        hybrid_n_layers=args.hybrid_n_layers,
        mc_n_paths=args.mc_paths,
        mc_seed=args.mc_seed,
        fd_n_S=args.fd_grid_size,
        fd_n_t=args.fd_grid_size * 10,
    )

    # Print configuration
    print("\n" + "=" * 60)
    print("Option Pricing Methods Benchmark")
    print("=" * 60)
    print(f"\nBlack-Scholes Parameters:")
    print(f"  r: {config.r}, σ: {config.sigma}, K: {config.K}, T: {config.T}")
    print(f"\nMethods to benchmark:")
    print(f"  - Analytical Black-Scholes")
    print(f"  - Monte Carlo ({config.mc_n_paths:,} paths)")
    print(f"  - Finite Difference ({config.fd_n_S}x{config.fd_n_t} grid)")
    print(f"  - Classical PINN ({config.pinn_epochs} epochs)")
    if not args.skip_hybrid:
        print(f"  - Hybrid PINN ({config.hybrid_epochs} epochs, {config.hybrid_n_qubits} qubits)")
    print()

    # Test points
    S_test = np.linspace(config.K * 0.5, config.K * 1.5, config.n_test_spots)

    # Run benchmarks
    results = {}

    # 1. Analytical (baseline)
    print("Running Analytical Black-Scholes...")
    results["Analytical (BS)"] = benchmark_analytical(config, S_test)
    analytical = results["Analytical (BS)"]

    # 2. Monte Carlo
    print("Running Monte Carlo simulation...")
    results["Monte Carlo"] = benchmark_monte_carlo(config, S_test, analytical.prices)

    # 3. Finite Difference
    print("Running Finite Difference...")
    results["Finite Difference (CN)"] = benchmark_finite_difference(
        config, S_test, analytical.prices, analytical.delta, analytical.gamma
    )

    # 4. Classical PINN
    print("\nTraining Classical PINN...")
    results["Classical PINN"] = benchmark_pinn(
        config, S_test, analytical.prices, analytical.delta, analytical.gamma,
        device, quiet=args.quiet
    )

    # 5. Hybrid PINN (optional)
    if not args.skip_hybrid:
        print("\nTraining Hybrid PINN (this may take a while)...")
        results["Hybrid PINN"] = benchmark_hybrid_pinn(
            config, S_test, analytical.prices, device, quiet=args.quiet
        )

    # Print summary
    print_summary_table(results)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results(results, config, output_dir)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating comparison plots...")
        create_comparison_plots(S_test, results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
