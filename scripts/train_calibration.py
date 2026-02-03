#!/usr/bin/env python3
"""
Training script for volatility surface calibration PINN.

This demonstrates the inverse problem: given option prices, infer volatility.

Usage:
    python scripts/train_calibration.py --epochs 2000
    python scripts/train_calibration.py --epochs 5000 --noise 0.01
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

from src.pde.dupire import DupireParams, generate_calibration_data, generate_synthetic_vol_surface
from src.classical.pinn_calibration import (
    VolCalibrationPINN,
    VolCalibrationTrainer,
    evaluate_calibration,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train volatility calibration PINN")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--hidden_dims", type=str, default="64,64,64",
                        help="Hidden layer dimensions")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_smooth", type=float, default=0.01, 
                        help="Smoothness regularization weight")
    parser.add_argument("--lambda_arb", type=float, default=0.1,
                        help="Arbitrage constraint weight")
    parser.add_argument("--n_strikes", type=int, default=15, help="Number of strikes")
    parser.add_argument("--n_maturities", type=int, default=10, help="Number of maturities")
    parser.add_argument("--noise", type=float, default=0.0, 
                        help="Add noise to synthetic prices (for robustness)")
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
        output_dir = PROJECT_ROOT / f"outputs/calibration/{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("VOLATILITY CALIBRATION PINN")
    print("=" * 70)
    print()
    
    # Create parameters
    params = DupireParams(
        r=0.05,
        S0=100.0,
        K_min=60.0,
        K_max=160.0,
        T_min=0.1,
        T_max=2.0,
        vol_base=0.20,
        vol_skew=-0.08,
        vol_smile=0.04,
        vol_term=-0.01,
    )
    
    print("Dupire Parameters:")
    print(f"  Spot: {params.S0}")
    print(f"  Rate: {params.r}")
    print(f"  Strike range: [{params.K_min}, {params.K_max}]")
    print(f"  Maturity range: [{params.T_min}, {params.T_max}]")
    print()
    
    print("True Vol Surface (parametric):")
    print(f"  Base vol: {params.vol_base}")
    print(f"  Skew: {params.vol_skew}")
    print(f"  Smile: {params.vol_smile}")
    print(f"  Term slope: {params.vol_term}")
    print()
    
    # Generate synthetic market data
    print(f"Generating synthetic market data ({args.n_strikes} x {args.n_maturities})...")
    surface_data = generate_synthetic_vol_surface(
        params,
        n_strikes=args.n_strikes,
        n_maturities=args.n_maturities,
        noise_std=args.noise,
        seed=42,
    )
    
    market_data = {
        "K": torch.tensor(surface_data["strikes"], dtype=torch.float32),
        "T": torch.tensor(surface_data["maturities"], dtype=torch.float32),
        "C_market": torch.tensor(surface_data["call_prices"], dtype=torch.float32),
        "IV_true": torch.tensor(surface_data["implied_vols"], dtype=torch.float32),
    }
    
    n_points = len(market_data["K"])
    print(f"  Generated {n_points} market data points")
    print(f"  Price range: [{market_data['C_market'].min():.2f}, {market_data['C_market'].max():.2f}]")
    print(f"  IV range: [{market_data['IV_true'].min():.3f}, {market_data['IV_true'].max():.3f}]")
    if args.noise > 0:
        print(f"  Added noise: {args.noise * 100:.1f}% std on IV")
    print()
    
    # Create model
    model = VolCalibrationPINN(
        params=params,
        hidden_dims=hidden_dims,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture:")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Parameters: {n_params:,}")
    print()
    
    # Create trainer
    trainer = VolCalibrationTrainer(
        model=model,
        market_data=market_data,
        lr=args.lr,
        lambda_smooth=args.lambda_smooth,
        lambda_arb=args.lambda_arb,
        device=args.device,
    )
    
    # Train
    print("Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Lambda smooth: {args.lambda_smooth}")
    print(f"  Lambda arb: {args.lambda_arb}")
    print()
    
    history = trainer.train(
        n_epochs=args.epochs,
        log_every=100,
    )
    
    # Evaluate
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print()
    
    metrics = evaluate_calibration(model, market_data, device=args.device)
    
    print("Price Fit:")
    print(f"  MSE:            {metrics['price_mse']:.6f}")
    print(f"  MAE:            {metrics['price_mae']:.4f}")
    print(f"  Max Error:      {metrics['max_price_error']:.4f}")
    print(f"  Mean Rel Error: {metrics['mean_rel_error_pct']:.2f}%")
    print()
    
    if "vol_mse" in metrics:
        print("Volatility Recovery:")
        print(f"  MSE:            {metrics['vol_mse']:.6f}")
        print(f"  MAE:            {metrics['vol_mae']:.4f}")
        print(f"  Max Error:      {metrics['max_vol_error']:.4f}")
        print(f"  Mean Error:     {metrics['mean_vol_error_pct']:.2f}%")
    print()
    
    # Save checkpoint
    checkpoint_path = output_dir / "calibration_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "params": {
            "r": params.r,
            "S0": params.S0,
            "K_min": params.K_min,
            "K_max": params.K_max,
            "T_min": params.T_min,
            "T_max": params.T_max,
            "vol_base": params.vol_base,
            "vol_skew": params.vol_skew,
            "vol_smile": params.vol_smile,
            "vol_term": params.vol_term,
        },
        "hidden_dims": hidden_dims,
        "history": history,
        "metrics": metrics,
    }, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "total": history["total"],
            "data": history["data"],
            "smooth": history["smooth"],
            "arb": history["arb"],
            "vol_mse": history["vol_mse"],
            "time": history["time"],
        }, f, indent=2)
    
    # Save evaluation
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation to: {eval_path}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Volatility Calibration PINN:
  - Inverse problem: Infer σ(K,T) from option prices
  - Market points: {n_points}
  - Parameters: {n_params:,}
  - Training: {args.epochs} epochs
  - Final data loss: {history['data'][-1]:.6f}
  - Vol recovery error: {metrics.get('mean_vol_error_pct', 'N/A'):.2f}%
  
This demonstrates real quant workflow: calibrating models to market data.
""")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
