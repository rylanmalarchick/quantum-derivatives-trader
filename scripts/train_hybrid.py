#!/usr/bin/env python3
"""
Training script for Hybrid Quantum-Classical PINN on Black-Scholes.

This script trains a hybrid quantum-classical PINN that uses variational
quantum circuits for function approximation. The quantum layer is
sandwiched between classical preprocessing and postprocessing layers.

WARNING: Quantum simulation is computationally expensive. Training will be
significantly slower than the classical PINN. Consider using fewer epochs
and smaller batch sizes for initial experiments.

Usage:
    python train_hybrid.py --epochs 1000 --n_qubits 4 --n_layers 3
    python train_hybrid.py --epochs 500 --n_qubits 6 --n_layers 4 --classical_hidden 32
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum.hybrid_pinn import HybridPINN, DeepHybridPINN, QuantumResidualPINN
from classical.pinn import PINN, PINNTrainer
from pde.black_scholes import BSParams, bs_analytical
from pricing.analytical import AnalyticalPricer
from data.collocation import create_grid, generate_collocation_points
from classical.losses import PINNLoss
from utils.visualization import (
    plot_training_history,
    plot_comparison,
    plot_surface,
    plot_convergence_comparison,
    set_publication_style,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Hybrid Quantum-Classical PINN for Black-Scholes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Quantum circuit parameters
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="Number of qubits in quantum circuit",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of variational layers in quantum circuit",
    )
    parser.add_argument(
        "--circuit_type",
        type=str,
        default="hardware_efficient",
        choices=["hardware_efficient", "strongly_entangling"],
        help="Type of variational quantum circuit",
    )

    # Classical components
    parser.add_argument(
        "--classical_hidden",
        type=int,
        default=32,
        help="Hidden dimension for classical pre/post-processing",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="hybrid",
        choices=["hybrid", "deep_hybrid", "quantum_residual"],
        help="Hybrid architecture type",
    )

    # Black-Scholes parameters
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--S_max", type=float, default=200.0, help="Max spot price")

    # Training parameters (reduced defaults due to quantum overhead)
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--n_interior", type=int, default=200, help="Interior collocation points")
    parser.add_argument("--n_boundary", type=int, default=50, help="Boundary collocation points")
    parser.add_argument("--n_terminal", type=int, default=50, help="Terminal collocation points")

    # Loss weights
    parser.add_argument("--lambda_pde", type=float, default=1.0, help="PDE loss weight")
    parser.add_argument("--lambda_bc", type=float, default=10.0, help="Boundary condition loss weight")
    parser.add_argument("--lambda_ic", type=float, default=10.0, help="Terminal condition loss weight")

    # Comparison with classical
    parser.add_argument(
        "--compare_classical",
        action="store_true",
        help="Also train a classical PINN for comparison",
    )
    parser.add_argument(
        "--classical_hidden_dims",
        type=int,
        nargs="+",
        default=[64, 64, 64, 64],
        help="Hidden dims for classical comparison PINN",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/hybrid", help="Output directory")
    parser.add_argument("--print_every", type=int, default=50, help="Print frequency")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")

    # Device
    parser.add_argument("--device", type=str, default="cpu", help="Device (quantum sim is CPU-bound)")

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up computation device."""
    device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def create_hybrid_model(args: argparse.Namespace) -> torch.nn.Module:
    """Create hybrid quantum-classical model from arguments."""
    if args.architecture == "hybrid":
        model = HybridPINN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            classical_hidden=args.classical_hidden,
            S_max=args.S_max,
            T_max=args.T,
            circuit_type=args.circuit_type,
        )
    elif args.architecture == "deep_hybrid":
        model = DeepHybridPINN(
            n_qubits=args.n_qubits,
            n_quantum_layers=2,
            layers_per_quantum=args.n_layers,
            classical_hidden=args.classical_hidden,
            S_max=args.S_max,
            T_max=args.T,
        )
    else:  # quantum_residual
        model = QuantumResidualPINN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            classical_hidden=[args.classical_hidden, args.classical_hidden],
            S_max=args.S_max,
            T_max=args.T,
        )
    return model


class HybridPINNTrainer:
    """
    Trainer for Hybrid Quantum-Classical PINN.
    
    Similar to PINNTrainer but adapted for hybrid models and includes
    timing information for quantum overhead analysis.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        params: BSParams,
        lr: float = 5e-3,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ):
        self.model = model
        self.params = params
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=200, factor=0.5, min_lr=1e-6
        )

        self.history = {"total": [], "pde": [], "bc": [], "ic": [], "time_per_epoch": []}

        self.loss_fn = PINNLoss(
            params,
            lambda_pde=lambda_pde,
            lambda_bc=lambda_bc,
            lambda_ic=lambda_ic,
        )

    def train_step(
        self,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()

        losses = self.loss_fn(
            self.model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal,
        )

        losses["total"].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train(
        self,
        n_epochs: int,
        n_interior: int = 200,
        n_boundary: int = 50,
        n_terminal: int = 50,
        print_every: int = 50,
    ) -> dict[str, list[float]]:
        """Full training loop with timing."""
        S_max = self.model.S_max
        T = self.params.T

        for epoch in range(n_epochs):
            epoch_start = time.time()

            # Generate fresh collocation points
            S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
                n_interior=n_interior,
                n_boundary=n_boundary,
                n_terminal=n_terminal,
                S_max=S_max,
                T=T,
            )

            losses = self.train_step(S_int, t_int, S_bc, t_bc, S_term)

            epoch_time = time.time() - epoch_start

            # Record history
            for key in ["total", "pde", "bc", "ic"]:
                self.history[key].append(losses[key])
            self.history["time_per_epoch"].append(epoch_time)

            # Learning rate scheduling
            self.scheduler.step(losses["total"])

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: total={losses['total']:.6f}, "
                    f"pde={losses['pde']:.6f}, bc={losses['bc']:.6f}, "
                    f"ic={losses['ic']:.6f}, time={epoch_time:.2f}s"
                )

        return self.history


def evaluate_model(
    model: torch.nn.Module,
    params: BSParams,
    device: torch.device,
    n_test: int = 100,
) -> dict:
    """Evaluate trained model against analytical solution."""
    model.eval()

    S_test = torch.linspace(1.0, params.K * 2, n_test, device=device)
    t_test = torch.zeros_like(S_test)

    with torch.no_grad():
        V_model = model(S_test, t_test).cpu().numpy()

    V_analytical = bs_analytical(S_test, t_test, params).cpu().numpy()
    S_np = S_test.cpu().numpy()

    abs_error = np.abs(V_model - V_analytical)
    rel_error = np.abs(abs_error / (V_analytical + 1e-8)) * 100

    valid_mask = V_analytical > 0.1
    mean_rel_error = np.mean(rel_error[valid_mask]) if valid_mask.any() else np.nan

    metrics = {
        "mse": float(np.mean((V_model - V_analytical) ** 2)),
        "mae": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_rel_error_pct": float(mean_rel_error),
        "rmse": float(np.sqrt(np.mean((V_model - V_analytical) ** 2))),
    }

    return {
        "S": S_np,
        "V_model": V_model,
        "V_analytical": V_analytical,
        "metrics": metrics,
    }


def train_classical_for_comparison(
    args: argparse.Namespace,
    bs_params: BSParams,
    device: torch.device,
) -> tuple:
    """Train a classical PINN for comparison."""
    print("\n" + "-" * 60)
    print("Training Classical PINN for comparison...")
    print("-" * 60)

    classical_model = PINN(
        hidden_dims=args.classical_hidden_dims,
        S_max=args.S_max,
        T_max=args.T,
        use_residual=False,
    ).to(device)

    classical_trainer = PINNTrainer(
        model=classical_model,
        params=bs_params,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
    )

    start_time = time.time()
    classical_history = classical_trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_terminal=args.n_terminal,
        print_every=args.print_every,
    )
    classical_time = time.time() - start_time

    classical_eval = evaluate_model(classical_model, bs_params, device)

    print(f"\nClassical PINN trained in {classical_time:.1f}s")
    print(f"Classical MSE: {classical_eval['metrics']['mse']:.6f}")

    return classical_model, classical_history, classical_eval, classical_time


def save_results(
    args: argparse.Namespace,
    model: torch.nn.Module,
    history: dict,
    eval_results: dict,
    training_time: float,
    output_dir: Path,
    classical_results: dict = None,
) -> None:
    """Save model, training history, and evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    if args.save_checkpoint:
        checkpoint_path = output_dir / "hybrid_pinn_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "metrics": eval_results["metrics"],
            },
            checkpoint_path,
        )
        print(f"Model checkpoint saved to: {checkpoint_path}")

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    # Save evaluation metrics
    results = {
        "hybrid_pinn": eval_results["metrics"],
        "training_time_seconds": training_time,
        "quantum_config": {
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "architecture": args.architecture,
        },
    }
    if classical_results:
        results["classical_pinn"] = classical_results["eval"]["metrics"]
        results["classical_training_time_seconds"] = classical_results["time"]

    metrics_path = output_dir / "results.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {metrics_path}")


def create_plots(
    args: argparse.Namespace,
    model: torch.nn.Module,
    params: BSParams,
    history: dict,
    eval_results: dict,
    output_dir: Path,
    device: torch.device,
    classical_results: dict = None,
) -> None:
    """Create and save visualization plots."""
    set_publication_style()

    # 1. Training history plot
    fig_history = plot_training_history(history, title="Hybrid PINN Training Loss")
    fig_history.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig_history)

    # 2. Comparison plot: Hybrid vs Analytical
    fig_comparison = plot_comparison(
        eval_results["S"],
        eval_results["V_model"],
        eval_results["V_analytical"],
        title="Hybrid Quantum-Classical PINN vs Analytical (t=0)",
    )
    fig_comparison.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_comparison)

    # 3. Option price surface
    n_S, n_t = 40, 25
    S_grid, t_grid = create_grid(n_S, n_t, args.S_max, args.T, device=device)

    model.eval()
    with torch.no_grad():
        V_grid = model(S_grid, t_grid).cpu().numpy()

    S_1d = np.linspace(0, args.S_max, n_S)
    t_1d = np.linspace(0, args.T, n_t)
    V_surface = V_grid.reshape(n_t, n_S)

    fig_surface = plot_surface(
        S_1d, t_1d, V_surface,
        title="Hybrid PINN Option Price Surface",
        ylabel="Time ($t$)",
    )
    fig_surface.savefig(output_dir / "price_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig_surface)

    # 4. Convergence comparison with classical (if available)
    if classical_results:
        histories = {
            "Hybrid PINN": history,
            "Classical PINN": classical_results["history"],
        }
        fig_conv = plot_convergence_comparison(
            histories,
            metric="total",
            title="Training Convergence: Hybrid vs Classical",
        )
        fig_conv.savefig(output_dir / "convergence_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig_conv)

        # Also create a comparison of both models vs analytical
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(eval_results["S"], eval_results["V_analytical"], 'b-', linewidth=2, label='Analytical', alpha=0.9)
        ax.plot(eval_results["S"], eval_results["V_model"], 'r--', linewidth=2, label='Hybrid PINN', alpha=0.9)
        ax.plot(classical_results["eval"]["S"], classical_results["eval"]["V_model"], 'g-.', linewidth=2, label='Classical PINN', alpha=0.9)
        ax.set_xlabel('Spot Price ($S$)', fontsize=12)
        ax.set_ylabel('Option Value ($V$)', fontsize=12)
        ax.set_title('Model Comparison (t=0)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to: {output_dir}")


def main():
    """Main training function."""
    args = parse_args()
    device = setup_device(args.device)

    # Print warning about quantum simulation overhead
    print("\n" + "!" * 60)
    print("WARNING: Quantum simulation is computationally expensive!")
    print("Training will be significantly slower than classical PINN.")
    print("Consider reducing epochs/points for initial experiments.")
    print("!" * 60)

    # Print configuration
    print("\n" + "=" * 60)
    print("Hybrid Quantum-Classical PINN Training")
    print("=" * 60)
    print(f"\nQuantum Configuration:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Circuit type: {args.circuit_type}")
    print(f"  Classical hidden dim: {args.classical_hidden}")

    print(f"\nBlack-Scholes Parameters:")
    print(f"  r (risk-free rate): {args.r}")
    print(f"  σ (volatility): {args.sigma}")
    print(f"  K (strike): {args.K}")
    print(f"  T (maturity): {args.T}")

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Boundary points: {args.n_boundary}")
    print(f"  Terminal points: {args.n_terminal}")
    print()

    # Create Black-Scholes parameters
    bs_params = BSParams(r=args.r, sigma=args.sigma, K=args.K, T=args.T)

    # Create hybrid model
    model = create_hybrid_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Hybrid model parameters: {n_params:,}")

    # Estimate training time
    est_time_per_epoch = args.n_interior * 0.01 * args.n_qubits * args.n_layers
    print(f"Estimated time per epoch: ~{est_time_per_epoch:.1f}s")
    print(f"Estimated total time: ~{est_time_per_epoch * args.epochs / 60:.0f} minutes")

    # Create trainer
    trainer = HybridPINNTrainer(
        model=model,
        params=bs_params,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
    )

    # Train hybrid model
    print("\n" + "-" * 60)
    print("Starting Hybrid PINN training...")
    print("-" * 60)

    start_time = time.time()
    history = trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_terminal=args.n_terminal,
        print_every=args.print_every,
    )
    training_time = time.time() - start_time

    print("-" * 60)
    print(f"Training complete! Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print("-" * 60)

    # Evaluate hybrid model
    print("\nEvaluating hybrid model against analytical Black-Scholes...")
    eval_results = evaluate_model(model, bs_params, device)

    # Optionally train classical for comparison
    classical_results = None
    if args.compare_classical:
        classical_model, classical_history, classical_eval, classical_time = \
            train_classical_for_comparison(args, bs_params, device)
        classical_results = {
            "model": classical_model,
            "history": classical_history,
            "eval": classical_eval,
            "time": classical_time,
        }

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print("\nHybrid PINN:")
    for metric, value in eval_results["metrics"].items():
        print(f"  {metric}: {value:.6f}")
    print(f"  Training time: {training_time:.1f}s")

    if classical_results:
        print("\nClassical PINN (for comparison):")
        for metric, value in classical_results["eval"]["metrics"].items():
            print(f"  {metric}: {value:.6f}")
        print(f"  Training time: {classical_results['time']:.1f}s")
        print(f"\nSpeedup ratio (classical/hybrid): {training_time / classical_results['time']:.1f}x slower")
    print("=" * 60)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results(args, model, history, eval_results, training_time, output_dir, classical_results)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        create_plots(args, model, bs_params, history, eval_results, output_dir, device, classical_results)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
