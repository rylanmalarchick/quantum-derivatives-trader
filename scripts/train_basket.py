#!/usr/bin/env python3
"""
Training script for 5-asset basket option PINN.

This demonstrates PINN capability for high-dimensional problems
where finite difference methods cannot scale.

Usage:
    python scripts/train_basket.py --epochs 5000 --n_interior 15000
    python scripts/train_basket.py --epochs 10000 --n_interior 30000 --eval
"""

import argparse
import sys
from pathlib import Path
import json
import time
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pde.basket import BasketParams, monte_carlo_basket
from src.classical.pinn_basket import BasketPINN, BasketPINNTrainer, evaluate_basket_pinn


def parse_args():
    parser = argparse.ArgumentParser(description="Train basket PINN")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--n_interior", type=int, default=15000, help="Interior collocation points")
    parser.add_argument("--n_terminal", type=int, default=8000, help="Terminal collocation points")
    parser.add_argument("--hidden_dims", type=str, default="128,128,128,128,128,128",
                        help="Hidden layer dimensions")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_pde", type=float, default=1.0, help="PDE loss weight")
    parser.add_argument("--lambda_ic", type=float, default=10.0, help="Terminal loss weight")
    parser.add_argument("--n_assets", type=int, default=5, help="Number of assets")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--n_eval", type=int, default=200, help="Evaluation test points")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    
    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"outputs/basket/{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BASKET OPTION PINN TRAINING")
    print("=" * 70)
    print()
    
    # Create basket parameters
    # Realistic 5-asset equity basket (tech-like)
    params = BasketParams(
        n_assets=args.n_assets,
        r=0.05,
        K=100.0,
        T=1.0,
        S0=np.array([100., 100., 100., 100., 100.][:args.n_assets]),
        sigma=np.array([0.20, 0.25, 0.18, 0.22, 0.20][:args.n_assets]),
        weights=np.ones(args.n_assets) / args.n_assets,  # Equal weights
    )
    
    print(f"Basket Parameters:")
    print(f"  Assets: {params.n_assets}")
    print(f"  Initial prices: {params.S0}")
    print(f"  Volatilities: {params.sigma}")
    print(f"  Weights: {params.weights}")
    print(f"  Correlation diagonal: {np.diag(params.correlation)}")
    print(f"  Strike: {params.K}")
    print(f"  Maturity: {params.T}")
    print()
    
    # Monte Carlo reference price at initial spot
    print("Computing MC reference price...")
    mc_ref = monte_carlo_basket(params, n_paths=500000, seed=42)
    print(f"  MC Price (at S0): ${mc_ref['price']:.4f} ± ${mc_ref['std_error']:.4f}")
    print(f"  95% CI: [{mc_ref['confidence_interval'][0]:.4f}, {mc_ref['confidence_interval'][1]:.4f}]")
    print()
    
    # Create model
    model = BasketPINN(
        n_assets=params.n_assets,
        hidden_dims=hidden_dims,
        S_max=params.S_max,
        T_max=params.T,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture:")
    print(f"  Input dim: {params.n_assets + 1} (S1..S{params.n_assets}, t)")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Parameters: {n_params:,}")
    print()
    
    # Create trainer
    trainer = BasketPINNTrainer(
        model=model,
        params=params,
        lr=args.lr,
        lambda_pde=args.lambda_pde,
        lambda_ic=args.lambda_ic,
        device=args.device,
    )
    
    # Train
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Interior points: {args.n_interior}")
    print(f"  Terminal points: {args.n_terminal}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda PDE: {args.lambda_pde}")
    print(f"  Lambda IC: {args.lambda_ic}")
    print()
    
    history = trainer.train(
        n_epochs=args.epochs,
        n_interior=args.n_interior,
        n_terminal=args.n_terminal,
        resample_every=100,
        log_every=100,
    )
    
    # Quick validation: predict at initial spot
    model.eval()
    S0_tensor = torch.tensor(params.S0, dtype=torch.float32).unsqueeze(0).to(args.device)
    t0_tensor = torch.zeros(1, device=args.device)
    
    with torch.no_grad():
        pinn_price = model(S0_tensor, t0_tensor).item()
    
    print()
    print("=" * 70)
    print("QUICK VALIDATION (at S0, t=0)")
    print("=" * 70)
    print(f"  PINN Price: ${pinn_price:.4f}")
    print(f"  MC Price:   ${mc_ref['price']:.4f}")
    print(f"  Error:      ${abs(pinn_price - mc_ref['price']):.4f}")
    print(f"  Rel Error:  {abs(pinn_price - mc_ref['price']) / mc_ref['price'] * 100:.2f}%")
    print()
    
    # Save checkpoint
    checkpoint_path = output_dir / "basket_pinn_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "params": {
            "n_assets": params.n_assets,
            "r": params.r,
            "K": params.K,
            "T": params.T,
            "S0": params.S0.tolist(),
            "sigma": params.sigma.tolist(),
            "weights": params.weights.tolist(),
            "correlation": params.correlation.tolist(),
        },
        "hidden_dims": hidden_dims,
        "history": history,
        "mc_reference": mc_ref,
    }, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "total": history["total"],
            "pde": history["pde"],
            "ic": history["ic"],
            "time": history["time"],
        }, f, indent=2)
    
    # Full evaluation
    if args.eval:
        print()
        print("=" * 70)
        print(f"EVALUATION (vs MC on {args.n_eval} test points)")
        print("=" * 70)
        print()
        
        metrics = evaluate_basket_pinn(model, params, n_test=args.n_eval, device=args.device)
        
        print(f"Results:")
        print(f"  MSE:              {metrics['mse']:.4f}")
        print(f"  MAE:              {metrics['mae']:.4f}")
        print(f"  Max Error:        {metrics['max_error']:.4f}")
        print(f"  Mean Rel Error:   {metrics['mean_rel_error_pct']:.2f}%")
        print(f"  Median Rel Error: {metrics['median_rel_error_pct']:.2f}%")
        
        # Save evaluation results
        eval_path = output_dir / "evaluation.json"
        with open(eval_path, "w") as f:
            json.dump({
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "max_error": metrics["max_error"],
                "mean_rel_error_pct": metrics["mean_rel_error_pct"],
                "median_rel_error_pct": metrics["median_rel_error_pct"],
                "n_test": args.n_eval,
            }, f, indent=2)
        print(f"\nSaved evaluation to: {eval_path}")
    
    print()
    print(f"All outputs saved to: {output_dir}")
    print()
    
    # Summary for resume/docs
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Basket Option PINN ({params.n_assets} assets):
  - Problem dimension: {params.n_assets + 1}D (FD infeasible)
  - Parameters: {n_params:,}
  - Training: {args.epochs} epochs, {args.n_interior} collocation points
  - Final loss: {history['total'][-1]:.4f}
  - PINN vs MC error at S0: {abs(pinn_price - mc_ref['price']) / mc_ref['price'] * 100:.2f}%
""")


if __name__ == "__main__":
    main()
