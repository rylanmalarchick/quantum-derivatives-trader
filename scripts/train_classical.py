#!/usr/bin/env python3
"""
Training script for Classical Physics-Informed Neural Network on Black-Scholes.

This script trains a classical PINN to solve the Black-Scholes PDE for
European option pricing, validates against analytical solutions, and
saves model checkpoints and training visualization.

Usage:
    python train_classical.py --epochs 5000 --lr 1e-3 --hidden_dims 64 64 64 64
    python train_classical.py --use_residual --epochs 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classical.pinn import PINN, PINNTrainer
from classical.networks import MLP, ResidualMLP
from pde.black_scholes import BSParams, bs_analytical
from pricing.analytical import AnalyticalPricer
from data.collocation import create_grid
from utils.visualization import (
    plot_training_history,
    plot_comparison,
    plot_surface,
    set_publication_style,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Classical PINN for Black-Scholes Option Pricing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 64, 64, 64],
        help="Hidden layer dimensions for the MLP",
    )
    parser.add_argument(
        "--use_residual",
        action="store_true",
        help="Use residual MLP architecture",
    )

    # Black-Scholes parameters
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity")
    parser.add_argument("--S_max", type=float, default=200.0, help="Max spot price")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_interior", type=int, default=1000, help="Interior collocation points")
    parser.add_argument("--n_boundary", type=int, default=200, help="Boundary collocation points")
    parser.add_argument("--n_terminal", type=int, default=200, help="Terminal collocation points")

    # Loss weights
    parser.add_argument("--lambda_pde", type=float, default=1.0, help="PDE loss weight")
    parser.add_argument("--lambda_bc", type=float, default=10.0, help="Boundary condition loss weight")
    parser.add_argument("--lambda_ic", type=float, default=10.0, help="Terminal condition loss weight")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/classical", help="Output directory")
    parser.add_argument("--print_every", type=int, default=100, help="Print frequency")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', or 'auto'")

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up computation device."""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Using device: {device}")
    return device


def create_model(args: argparse.Namespace) -> PINN:
    """Create PINN model from arguments."""
    model = PINN(
        hidden_dims=args.hidden_dims,
        S_max=args.S_max,
        T_max=args.T,
        use_residual=args.use_residual,
    )
    return model


def evaluate_model(
    model: PINN,
    params: BSParams,
    device: torch.device,
    n_test: int = 100,
) -> dict:
    """Evaluate trained model against analytical solution."""
    model.eval()

    # Test grid: evaluate at t=0 (today) for various spot prices
    S_test = torch.linspace(1.0, params.K * 2, n_test, device=device)
    t_test = torch.zeros_like(S_test)

    with torch.no_grad():
        V_pinn = model(S_test, t_test).cpu().numpy()

    # Analytical solution
    V_analytical = bs_analytical(S_test, t_test, params).cpu().numpy()
    S_np = S_test.cpu().numpy()

    # Compute errors
    abs_error = np.abs(V_pinn - V_analytical)
    rel_error = np.abs(abs_error / (V_analytical + 1e-8)) * 100

    # Filter out near-zero regions for meaningful relative error
    valid_mask = V_analytical > 0.1
    mean_rel_error = np.mean(rel_error[valid_mask]) if valid_mask.any() else np.nan

    metrics = {
        "mse": float(np.mean((V_pinn - V_analytical) ** 2)),
        "mae": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "mean_rel_error_pct": float(mean_rel_error),
        "rmse": float(np.sqrt(np.mean((V_pinn - V_analytical) ** 2))),
    }

    return {
        "S": S_np,
        "V_pinn": V_pinn,
        "V_analytical": V_analytical,
        "metrics": metrics,
    }


def save_results(
    args: argparse.Namespace,
    model: PINN,
    history: dict,
    eval_results: dict,
    output_dir: Path,
) -> None:
    """Save model, training history, and evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    if args.save_checkpoint:
        checkpoint_path = output_dir / "pinn_checkpoint.pt"
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
    print(f"Training history saved to: {history_path}")

    # Save evaluation metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_results["metrics"], f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")


def create_plots(
    args: argparse.Namespace,
    model: PINN,
    params: BSParams,
    history: dict,
    eval_results: dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Create and save visualization plots."""
    set_publication_style()

    # 1. Training history plot
    fig_history = plot_training_history(history, title="Classical PINN Training Loss")
    fig_history.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig_history)
    print(f"Training history plot saved to: {output_dir / 'training_history.png'}")

    # 2. Comparison plot: PINN vs Analytical at t=0
    fig_comparison = plot_comparison(
        eval_results["S"],
        eval_results["V_pinn"],
        eval_results["V_analytical"],
        title="Classical PINN vs Analytical Black-Scholes (t=0)",
    )
    fig_comparison.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_comparison)
    print(f"Comparison plot saved to: {output_dir / 'comparison.png'}")

    # 3. Option price surface
    n_S, n_t = 50, 30
    S_grid, t_grid = create_grid(n_S, n_t, args.S_max, args.T, device=device)

    model.eval()
    with torch.no_grad():
        V_grid = model(S_grid, t_grid).cpu().numpy()

    S_1d = np.linspace(0, args.S_max, n_S)
    t_1d = np.linspace(0, args.T, n_t)
    V_surface = V_grid.reshape(n_t, n_S)

    fig_surface = plot_surface(
        S_1d, t_1d, V_surface,
        title="PINN Option Price Surface",
        ylabel="Time ($t$)",
    )
    fig_surface.savefig(output_dir / "price_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig_surface)
    print(f"Price surface plot saved to: {output_dir / 'price_surface.png'}")


def main():
    """Main training function."""
    args = parse_args()
    device = setup_device(args.device)

    # Print configuration
    print("\n" + "=" * 60)
    print("Classical PINN Training for Black-Scholes Option Pricing")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Use residual: {args.use_residual}")

    print(f"\nBlack-Scholes Parameters:")
    print(f"  r (risk-free rate): {args.r}")
    print(f"  σ (volatility): {args.sigma}")
    print(f"  K (strike): {args.K}")
    print(f"  T (maturity): {args.T}")
    print(f"  S_max: {args.S_max}")

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Boundary points: {args.n_boundary}")
    print(f"  Terminal points: {args.n_terminal}")
    print(f"  Loss weights (PDE/BC/IC): {args.lambda_pde}/{args.lambda_bc}/{args.lambda_ic}")
    print()

    # Create Black-Scholes parameters
    bs_params = BSParams(r=args.r, sigma=args.sigma, K=args.K, T=args.T)

    # Create model and move to device
    model = create_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = PINNTrainer(
        model=model,
        params=bs_params,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_bc=args.lambda_bc,
        lambda_ic=args.lambda_ic,
    )

    # Train
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)

    history = trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        n_terminal=args.n_terminal,
        print_every=args.print_every,
    )

    print("-" * 60)
    print("Training complete!")
    print("-" * 60)

    # Evaluate
    print("\nEvaluating model against analytical Black-Scholes solution...")
    eval_results = evaluate_model(model, bs_params, device)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    for metric, value in eval_results["metrics"].items():
        print(f"  {metric}: {value:.6f}")
    print("=" * 60)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results(args, model, history, eval_results, output_dir)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        create_plots(args, model, bs_params, history, eval_results, output_dir, device)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
