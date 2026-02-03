#!/usr/bin/env python
"""
Train Barrier Option PINN for down-and-out call pricing.

Barrier options are path-dependent but can be priced via PDE with
modified boundary conditions. This script trains a PINN to price
down-and-out call options.

Usage:
    python scripts/train_barrier.py --epochs 2000 --eval
    python scripts/train_barrier.py --barrier 70 --strike 100 --eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.pde.barrier import (
    BarrierParams,
    BarrierPINN,
    BarrierPINNTrainer,
    barrier_analytical_down_out_call,
    evaluate_barrier_pinn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Barrier Option PINN")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of training epochs")
    parser.add_argument("--n_interior", type=int, default=2000,
                        help="Number of interior collocation points")
    parser.add_argument("--n_terminal", type=int, default=500,
                        help="Number of terminal collocation points")
    parser.add_argument("--n_barrier", type=int, default=200,
                        help="Number of barrier boundary points")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64, 64],
                        help="Hidden layer dimensions")
    
    # Option parameters
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price K")
    parser.add_argument("--barrier", type=float, default=80.0, help="Barrier level B")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate r")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility σ")
    parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity T")
    parser.add_argument("--S_max", type=float, default=300.0, help="Max spot for PINN")
    
    # Loss weights
    parser.add_argument("--lambda_pde", type=float, default=1.0,
                        help="Weight for PDE residual loss")
    parser.add_argument("--lambda_ic", type=float, default=10.0,
                        help="Weight for terminal condition loss")
    parser.add_argument("--lambda_barrier", type=float, default=100.0,
                        help="Weight for barrier boundary loss")
    parser.add_argument("--lambda_upper", type=float, default=1.0,
                        help="Weight for upper boundary loss")
    
    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--output", type=str, default="outputs/barrier",
                        help="Output directory")
    
    return parser.parse_args()


def print_option_info(params: BarrierParams):
    """Print option parameters and reference prices."""
    print(f"\nDown-and-Out Call Option Parameters:")
    print(f"  Strike (K):     ${params.K:.2f}")
    print(f"  Barrier (B):    ${params.B:.2f}")
    print(f"  Risk-free (r):  {params.r*100:.2f}%")
    print(f"  Volatility (σ): {params.sigma*100:.2f}%")
    print(f"  Maturity (T):   {params.T:.2f} years")
    
    # Reference prices at S = 100 (t=0)
    S_ref = torch.tensor([100.0])
    tau_ref = torch.tensor([params.T])
    
    barrier_price = barrier_analytical_down_out_call(S_ref, params, tau_ref).item()
    
    # Vanilla call for comparison
    from src.pde.barrier import _bs_call_price
    vanilla_price = _bs_call_price(
        S_ref.numpy(), params.K, params.r, params.sigma, tau_ref.numpy()
    )[0]
    
    print(f"\nReference Prices at S = $100.00 (t=0):")
    print(f"  Vanilla Call:     ${vanilla_price:.4f}")
    print(f"  Down-Out Call:    ${barrier_price:.4f}")
    print(f"  Barrier Discount: {(1 - barrier_price/vanilla_price)*100:.2f}%")


def evaluate_at_spots(model, params, spots, t=0.0):
    """Evaluate model at specific spot prices."""
    model.eval()
    
    S = torch.tensor(spots, dtype=torch.float32)
    t_tensor = torch.full_like(S, t)
    tau = params.T - t
    tau_tensor = torch.full_like(S, tau)
    
    with torch.no_grad():
        pinn_prices = model(S, t_tensor).numpy()
    
    analytical_prices = barrier_analytical_down_out_call(S, params, tau_tensor).numpy()
    
    print(f"\nPrices at t={t:.2f}:")
    print(f"{'Spot':>10} {'PINN':>12} {'Analytical':>12} {'Error':>10}")
    print("-" * 46)
    
    for i, s in enumerate(spots):
        error = pinn_prices[i] - analytical_prices[i]
        print(f"{s:>10.2f} {pinn_prices[i]:>12.4f} {analytical_prices[i]:>12.4f} {error:>10.4f}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("BARRIER OPTION PINN TRAINING")
    print("Down-and-Out Call Option")
    print("=" * 70)
    
    # Create parameters
    try:
        params = BarrierParams(
            r=args.rate,
            sigma=args.sigma,
            K=args.strike,
            T=args.maturity,
            B=args.barrier,
            barrier_type="down-out-call",
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("For down-and-out call, barrier must be below strike.")
        return
    
    print_option_info(params)
    
    # Create model
    model = BarrierPINN(
        params,
        hidden_dims=args.hidden,
        S_max=args.S_max,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:")
    print(f"  Hidden layers: {args.hidden}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  S_max:         {args.S_max:.1f}")
    
    # Create trainer
    trainer = BarrierPINNTrainer(
        model,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_ic=args.lambda_ic,
        lambda_barrier=args.lambda_barrier,
        lambda_upper=args.lambda_upper,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Terminal points: {args.n_terminal}")
    print(f"  Barrier points:  {args.n_barrier}")
    print(f"  Learning rate:   {args.lr}")
    
    print(f"\nLoss Weights:")
    print(f"  λ_pde:     {args.lambda_pde}")
    print(f"  λ_ic:      {args.lambda_ic}")
    print(f"  λ_barrier: {args.lambda_barrier}")
    print(f"  λ_upper:   {args.lambda_upper}")
    
    # Train
    print(f"\n{'='*70}")
    print("Training...")
    print("=" * 70)
    
    start_time = time.time()
    history = trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_terminal=args.n_terminal,
        n_barrier=args.n_barrier,
        log_every=200,
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Final loss: {history[-1]['total']:.6f}")
    
    # Save
    output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "params": {
            "r": params.r,
            "sigma": params.sigma,
            "K": params.K,
            "T": params.T,
            "B": params.B,
            "barrier_type": params.barrier_type,
        },
        "args": vars(args),
        "train_time": train_time,
    }, output_dir / "barrier_checkpoint.pt")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nCheckpoint saved to: {output_dir / 'barrier_checkpoint.pt'}")
    
    # Evaluate
    if args.eval:
        print(f"\n{'='*70}")
        print("Evaluation")
        print("=" * 70)
        
        results = evaluate_barrier_pinn(model, params, n_points=100)
        
        print(f"\nOverall Metrics:")
        print(f"  MSE:            {results['mse']:.6f}")
        print(f"  MAE:            {results['mae']:.4f}")
        print(f"  Max Error:      {results['max_error']:.4f}")
        print(f"  Mean Rel Error: {results['mean_rel_error_pct']:.2f}%")
        
        # Sample prices at key spots
        spots = [params.B + 5, params.B + 10, 90.0, 100.0, 110.0, 120.0, 150.0]
        spots = [s for s in spots if s > params.B]  # Filter to above barrier
        evaluate_at_spots(model, params, spots)
        
        # Verify barrier condition
        print(f"\nBarrier Condition Check (S = B = {params.B}):")
        S_barrier = torch.tensor([params.B])
        t_vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]) * params.T
        
        with torch.no_grad():
            for t in t_vals:
                V = model(S_barrier, torch.tensor([t])).item()
                print(f"  V(B, t={t:.2f}) = {V:.6f}")
        
        # Save evaluation results
        with open(output_dir / "evaluation.json", "w") as f:
            # Remove large arrays for JSON
            eval_summary = {k: v for k, v in results.items() 
                          if k not in ["S_vals", "pinn_prices", "analytical_prices"]}
            json.dump(eval_summary, f, indent=2)
        
        # Save full results as numpy
        np.savez(
            output_dir / "price_comparison.npz",
            S=results["S_vals"],
            pinn=results["pinn_prices"],
            analytical=results["analytical_prices"],
        )
        
        print(f"\nEvaluation saved to: {output_dir}")
    
    print(f"\n{'='*70}")
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
