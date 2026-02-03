#!/usr/bin/env python
"""
Train Merton jump-diffusion PINN.

The Merton model adds Poisson jumps to geometric Brownian motion,
creating a PIDE (partial integro-differential equation).

Usage:
    python scripts/train_merton.py --epochs 2000 --eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.pde.merton import (
    MertonParams, MertonPINN, MertonPINNTrainer,
    merton_analytical_call
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Merton PINN")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n_interior", type=int, default=2000)
    parser.add_argument("--n_terminal", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_quad", type=int, default=15, help="Quadrature points for integral")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--eval", action="store_true", help="Evaluate against analytical")
    parser.add_argument("--output", type=str, default="outputs/merton")
    return parser.parse_args()


def evaluate_merton(model, params, n_points=200):
    """Compare PINN to analytical Merton formula."""
    model.eval()
    
    S_vals = np.linspace(50, 200, n_points)
    tau_vals = [0.1, 0.5, 1.0]
    
    results = {}
    
    with torch.no_grad():
        for tau in tau_vals:
            t = params.T - tau
            
            # Analytical
            analytical = merton_analytical_call(S_vals, params, tau)
            
            # PINN
            S_torch = torch.tensor(S_vals, dtype=torch.float32)
            t_torch = torch.full_like(S_torch, t)
            pinn_prices = model(S_torch, t_torch).numpy()
            
            mse = np.mean((pinn_prices - analytical) ** 2)
            mae = np.mean(np.abs(pinn_prices - analytical))
            
            results[f"tau_{tau}"] = {
                "mse": float(mse),
                "mae": float(mae),
                "max_error": float(np.max(np.abs(pinn_prices - analytical))),
            }
    
    return results


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MERTON JUMP-DIFFUSION PINN TRAINING")
    print("=" * 70)
    
    # Setup parameters
    params = MertonParams(
        r=0.05,
        sigma=0.20,
        K=100.0,
        T=1.0,
        lam=0.5,      # 0.5 jumps per year
        mu_J=-0.10,   # -10% mean jump
        sigma_J=0.15, # 15% jump volatility
    )
    
    print(f"\nMerton Parameters:")
    print(f"  r: {params.r}")
    print(f"  σ (diffusion): {params.sigma}")
    print(f"  K: {params.K}")
    print(f"  λ (jump intensity): {params.lam}")
    print(f"  μ_J (mean log-jump): {params.mu_J}")
    print(f"  σ_J (jump vol): {params.sigma_J}")
    print(f"  κ (E[J-1]): {params.kappa:.4f}")
    print(f"  Drift: {params.drift:.4f}")
    
    # Reference price at S=100
    ref_price = merton_analytical_call(100.0, params, tau=1.0)[0]
    print(f"\nAnalytical price at S=100, τ=1: ${ref_price:.4f}")
    
    # Create model
    model = MertonPINN(params, hidden_dims=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    
    # Trainer
    trainer = MertonPINNTrainer(
        model, 
        lr=args.lr,
        n_quad=args.n_quad,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Terminal points: {args.n_terminal}")
    print(f"  Quadrature points: {args.n_quad}")
    
    # Train
    print(f"\nTraining...")
    start_time = time.time()
    history = trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_terminal=args.n_terminal,
        log_every=100,
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f}s")
    
    # Save
    output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "params": vars(params),
        "args": vars(args),
    }, output_dir / "merton_checkpoint.pt")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Evaluate
    if args.eval:
        print("\nEvaluating against analytical solution...")
        results = evaluate_merton(model, params)
        
        for tau_key, metrics in results.items():
            print(f"\n{tau_key}:")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Max Error: {metrics['max_error']:.4f}")
        
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
