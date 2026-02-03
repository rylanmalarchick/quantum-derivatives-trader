#!/usr/bin/env python
"""
Train Hybrid Quantum-Classical PINN for 5-Asset Basket Option.

Tests three architectures:
1. HybridBasketPINN: Classical encoder → VQC → Decoder
2. EnsembleHybridBasketPINN: Multiple VQCs for asset pairs
3. QuantumEnhancedBasketPINN: Classical + quantum residual

Usage:
    python scripts/train_hybrid_basket.py --arch hybrid --epochs 1000 --eval
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from src.pde.basket import BasketParams, basket_pde_residual, basket_payoff
from src.quantum.hybrid_basket import (
    HybridBasketPINN,
    EnsembleHybridBasketPINN,
    QuantumEnhancedBasketPINN,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Quantum Basket PINN")
    parser.add_argument("--arch", type=str, default="hybrid",
                        choices=["hybrid", "ensemble", "enhanced"],
                        help="Architecture type")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n_interior", type=int, default=500)
    parser.add_argument("--n_terminal", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    parser.add_argument("--lambda_ic", type=float, default=10.0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/hybrid_basket")
    return parser.parse_args()


def create_model(arch: str, params: BasketParams, n_qubits: int, n_layers: int):
    """Create the specified hybrid architecture."""
    if arch == "hybrid":
        return HybridBasketPINN(
            n_assets=params.n_assets,
            n_qubits=n_qubits,
            n_layers=n_layers,
            S_max=params.S_max.max(),
            T_max=params.T,
        )
    elif arch == "ensemble":
        return EnsembleHybridBasketPINN(
            n_assets=params.n_assets,
            n_qubits=n_qubits,
            n_layers=n_layers,
            S_max=params.S_max.max(),
            T_max=params.T,
        )
    elif arch == "enhanced":
        return QuantumEnhancedBasketPINN(
            n_assets=params.n_assets,
            n_qubits=n_qubits,
            n_layers=n_layers,
            classical_dims=[64, 64, 64, 64],
            S_max=params.S_max.max(),
            T_max=params.T,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def sample_lhs(n_samples: int, n_dims: int, bounds: list, seed: int = 42):
    """Latin Hypercube Sampling for high-dimensional space."""
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    scaled = qmc.scale(samples, lower, upper)
    
    return torch.tensor(scaled, dtype=torch.float32)


def compute_loss(model, params, S_int, t_int, S_term, lambda_pde, lambda_ic):
    """Compute PDE + terminal condition loss."""
    # Forward pass for interior
    S_int = S_int.requires_grad_(True)
    t_int = t_int.requires_grad_(True)
    
    V_int = model(S_int, t_int)
    
    # PDE residual
    pde_res = basket_pde_residual(V_int, S_int, t_int, params)
    pde_loss = (pde_res ** 2).mean()
    
    # Terminal condition
    t_T = torch.full((S_term.shape[0],), params.T)
    V_term = model(S_term, t_T)
    payoff = basket_payoff(S_term, params)
    ic_loss = ((V_term - payoff) ** 2).mean()
    
    total = lambda_pde * pde_loss + lambda_ic * ic_loss
    
    return {
        "total": total,
        "pde": pde_loss,
        "ic": ic_loss,
    }


def train(model, params, args):
    """Training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    history = []
    start_time = time.time()
    
    # Bounds for LHS
    bounds = [(0.5 * s, 1.5 * s) for s in params.S0] + [(0, params.T)]
    
    for epoch in range(args.epochs):
        # Sample new points each epoch
        interior = sample_lhs(args.n_interior, params.n_assets + 1, bounds, seed=epoch)
        S_int = interior[:, :params.n_assets]
        t_int = interior[:, -1]
        
        terminal = sample_lhs(args.n_terminal, params.n_assets, bounds[:-1], seed=epoch + 10000)
        S_term = terminal
        
        # Training step
        optimizer.zero_grad()
        losses = compute_loss(model, params, S_int, t_int, S_term, args.lambda_pde, args.lambda_ic)
        losses["total"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        history.append({k: v.item() for k, v in losses.items()})
        
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d}: total={losses['total'].item():.4f}, "
                  f"pde={losses['pde'].item():.4f}, ic={losses['ic'].item():.4f}, "
                  f"time={elapsed:.1f}s")
    
    return history


def evaluate(model, params, n_points=500):
    """Evaluate against MC reference."""
    model.eval()
    
    # Sample test points around S0
    rng = np.random.default_rng(999)
    S_test = params.S0 * (0.8 + 0.4 * rng.random((n_points, params.n_assets)))
    t_test = rng.random(n_points) * params.T * 0.9
    
    S_tensor = torch.tensor(S_test, dtype=torch.float32)
    t_tensor = torch.tensor(t_test, dtype=torch.float32)
    
    with torch.no_grad():
        V_pinn = model(S_tensor, t_tensor).numpy()
    
    # We don't have MC reference for arbitrary points, so evaluate at terminal
    S_term = params.S0 * (0.8 + 0.4 * rng.random((n_points, params.n_assets)))
    S_term_tensor = torch.tensor(S_term, dtype=torch.float32)
    t_T = torch.full((n_points,), params.T)
    
    with torch.no_grad():
        V_term = model(S_term_tensor, t_T).numpy()
    
    payoff = basket_payoff(S_term_tensor, params).numpy()
    
    terminal_mse = np.mean((V_term - payoff) ** 2)
    terminal_mae = np.mean(np.abs(V_term - payoff))
    
    return {
        "terminal_mse": float(terminal_mse),
        "terminal_mae": float(terminal_mae),
    }


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"HYBRID QUANTUM BASKET PINN - {args.arch.upper()}")
    print("=" * 70)
    
    params = BasketParams(n_assets=5)
    
    print(f"\nBasket Parameters:")
    print(f"  Assets: {params.n_assets}")
    print(f"  Initial prices: {params.S0}")
    print(f"  Volatilities: {params.sigma}")
    print(f"  Strike: {params.K}")
    
    # Create model
    model = create_model(args.arch, params, args.n_qubits, args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.arch}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  VQC layers: {args.n_layers}")
    
    print(f"\nTraining...")
    history = train(model, params, args)
    
    # Save
    output_dir = Path(args.output) / f"{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "arch": args.arch,
        "args": vars(args),
    }, output_dir / "checkpoint.pt")
    
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f)
    
    if args.eval:
        print("\nEvaluating...")
        results = evaluate(model, params)
        print(f"  Terminal MSE: {results['terminal_mse']:.4f}")
        print(f"  Terminal MAE: {results['terminal_mae']:.4f}")
        
        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
