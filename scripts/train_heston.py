#!/usr/bin/env python
"""
Train Heston stochastic volatility PINN.

The Heston model has stochastic variance following a CIR process.
This is a 3D PDE (S, v, t).

Usage:
    python scripts/train_heston.py --epochs 3000 --eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.pde.heston import (
    HestonParams, HestonPINN, HestonPINNTrainer,
    heston_call_price
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Heston PINN")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--n_interior", type=int, default=3000)
    parser.add_argument("--n_terminal", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/heston")
    return parser.parse_args()


def evaluate_heston(model, params, n_points=50):
    """Compare PINN to semi-analytical Heston."""
    model.eval()
    
    S_vals = np.linspace(70, 150, n_points)
    v_vals = [0.02, 0.04, 0.08]  # Low, medium, high variance
    
    results = {}
    
    with torch.no_grad():
        for v in v_vals:
            analytical = np.array([
                heston_call_price(S, params._replace(v0=v) if hasattr(params, '_replace') else params, tau=1.0)
                for S in S_vals
            ])
            
            S_torch = torch.tensor(S_vals, dtype=torch.float32)
            v_torch = torch.full_like(S_torch, v)
            t_torch = torch.zeros_like(S_torch)  # t=0 means τ=T
            
            pinn_prices = model(S_torch, v_torch, t_torch).numpy()
            
            mse = np.mean((pinn_prices - analytical) ** 2)
            mae = np.mean(np.abs(pinn_prices - analytical))
            
            results[f"v_{v}"] = {
                "mse": float(mse),
                "mae": float(mae),
                "max_error": float(np.max(np.abs(pinn_prices - analytical))),
            }
    
    return results


def main():
    args = parse_args()
    
    print("=" * 70)
    print("HESTON STOCHASTIC VOLATILITY PINN TRAINING")
    print("=" * 70)
    
    params = HestonParams(
        r=0.05,
        K=100.0,
        T=1.0,
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-run variance (~20% vol)
        xi=0.3,       # Vol of vol
        rho=-0.7,     # Leverage effect
        v0=0.04,      # Initial variance
    )
    
    print(f"\nHeston Parameters:")
    print(f"  r: {params.r}")
    print(f"  K: {params.K}")
    print(f"  κ (mean reversion): {params.kappa}")
    print(f"  θ (long-run var): {params.theta} (σ≈{np.sqrt(params.theta)*100:.0f}%)")
    print(f"  ξ (vol of vol): {params.xi}")
    print(f"  ρ (correlation): {params.rho}")
    print(f"  v₀ (initial var): {params.v0} (σ₀={params.initial_vol*100:.0f}%)")
    print(f"  Feller satisfied: {params.feller_satisfied}")
    
    # Reference price
    ref_price = heston_call_price(100.0, params, tau=1.0)
    print(f"\nSemi-analytical price at S=100, τ=1: ${ref_price:.4f}")
    
    # Model
    model = HestonPINN(params, hidden_dims=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters (3D input)")
    
    trainer = HestonPINNTrainer(model, lr=args.lr)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Terminal points: {args.n_terminal}")
    
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
        "params": {k: v for k, v in vars(params).items() if not k.startswith('_')},
        "args": vars(args),
    }, output_dir / "heston_checkpoint.pt")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    if args.eval:
        print("\nEvaluating against semi-analytical solution...")
        results = evaluate_heston(model, params)
        
        for v_key, metrics in results.items():
            print(f"\n{v_key}:")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Max Error: {metrics['max_error']:.4f}")
        
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
