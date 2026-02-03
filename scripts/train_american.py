#!/usr/bin/env python
"""
Train American option PINN with early exercise.

American options can be exercised at any time, creating a free boundary problem.
We use the penalty method to enforce the obstacle constraint.

Usage:
    python scripts/train_american.py --epochs 2000 --eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.pde.american import (
    AmericanParams, AmericanPINN, AmericanPINNTrainer,
    american_put_binomial, american_call_binomial,
    find_early_exercise_boundary
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train American PINN")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n_interior", type=int, default=2000)
    parser.add_argument("--n_terminal", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64])
    parser.add_argument("--option_type", type=str, default="put", choices=["put", "call"])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/american")
    return parser.parse_args()


def evaluate_american(model, params, n_points=100):
    """Compare PINN to binomial tree."""
    model.eval()
    
    S_vals = np.linspace(50, 150, n_points)
    
    # Binomial reference prices
    binomial_fn = american_put_binomial if params.is_put else american_call_binomial
    binomial_prices = np.array([binomial_fn(S, params, n_steps=300) for S in S_vals])
    
    # PINN prices at t=0
    with torch.no_grad():
        S_torch = torch.tensor(S_vals, dtype=torch.float32)
        t_torch = torch.zeros_like(S_torch)
        pinn_prices = model(S_torch, t_torch).numpy()
    
    mse = np.mean((pinn_prices - binomial_prices) ** 2)
    mae = np.mean(np.abs(pinn_prices - binomial_prices))
    rel_error = np.mean(np.abs(pinn_prices - binomial_prices) / (binomial_prices + 1e-6)) * 100
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_error": float(np.max(np.abs(pinn_prices - binomial_prices))),
        "mean_rel_error_pct": float(rel_error),
    }


def main():
    args = parse_args()
    
    print("=" * 70)
    print("AMERICAN OPTION PINN TRAINING")
    print("=" * 70)
    
    params = AmericanParams(
        r=0.05,
        sigma=0.25,
        K=100.0,
        T=1.0,
        option_type=args.option_type,
    )
    
    print(f"\nAmerican {params.option_type.upper()} Parameters:")
    print(f"  r: {params.r}")
    print(f"  σ: {params.sigma}")
    print(f"  K: {params.K}")
    print(f"  T: {params.T}")
    
    # Reference price at S=100
    binomial_fn = american_put_binomial if params.is_put else american_call_binomial
    ref_price = binomial_fn(100.0, params, n_steps=500)
    print(f"\nBinomial reference (500 steps) at S=100: ${ref_price:.4f}")
    
    # For comparison: European price
    from src.pricing.analytical import AnalyticalPricer
    euro_pricer = AnalyticalPricer(r=params.r, sigma=params.sigma)
    euro_price = euro_pricer.black_scholes(100.0, params.K, params.T, params.option_type).item()
    print(f"European {params.option_type} at S=100: ${euro_price:.4f}")
    print(f"Early exercise premium: ${ref_price - euro_price:.4f}")
    
    # Model
    model = AmericanPINN(params, hidden_dims=args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    
    trainer = AmericanPINNTrainer(model, lr=args.lr)
    
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
        "params": vars(params),
        "args": vars(args),
    }, output_dir / "american_checkpoint.pt")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    if args.eval:
        print("\nEvaluating against binomial tree...")
        results = evaluate_american(model, params)
        
        print(f"\nPricing Accuracy:")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  Max Error: {results['max_error']:.4f}")
        print(f"  Mean Rel Error: {results['mean_rel_error_pct']:.2f}%")
        
        # Find early exercise boundary
        if params.is_put:
            print("\nEarly Exercise Boundary (S* where exercise is optimal):")
            t_values = np.array([0.0, 0.25, 0.5, 0.75, 0.9])
            boundary = find_early_exercise_boundary(model, t_values, S_range=(50, 120))
            for t, s_star in zip(t_values, boundary):
                print(f"  t={t:.2f}: S* = {s_star:.2f}")
            results["early_exercise_boundary"] = boundary.tolist()
        
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
