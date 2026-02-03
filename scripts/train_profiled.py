#!/usr/bin/env python3
"""
Profiled Training Script: Demonstrates compute infrastructure integration.

This script shows how to integrate the compute module into training:
1. Profile the training loop to identify bottlenecks
2. Route quantum workloads to optimal backends
3. Track timing breakdown across components

This is meant as a demonstration of the infrastructure, not for production training.

Usage:
    python scripts/train_profiled.py --epochs 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compute.profiler import Profiler, benchmark, profile_section
from src.compute.router import get_router, route_circuit
from src.compute.scheduler import TaskScheduler, ComputeTask, Priority

from src.quantum.hybrid_pinn import HybridPINN
from src.classical.pinn import PINN
from src.pde.black_scholes import BSParams, bs_analytical
from src.data.collocation import generate_collocation_points
from src.classical.losses import PINNLoss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profiled PINN Training with Compute Infrastructure",
    )
    
    # Model selection
    parser.add_argument("--model", choices=["classical", "hybrid"], default="classical")
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_interior", type=int, default=200)
    
    # Black-Scholes
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.05)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/profiled")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on args."""
    if args.model == "classical":
        model = PINN(
            hidden_dims=[64, 64, 64, 64],
            S_max=200.0,
            T_max=args.T,
        )
    else:
        model = HybridPINN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            classical_hidden=32,
            S_max=200.0,
            T_max=args.T,
        )
    return model


def profile_single_epoch(model, loss_fn, optimizer, params, profiler, args):
    """Profile a single training epoch in detail."""
    
    with profiler.section("data_generation"):
        S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
            n_interior=args.n_interior,
            n_boundary=50,
            n_terminal=50,
            S_max=200.0,
            T=args.T,
        )
    
    with profiler.section("forward_pass"):
        optimizer.zero_grad()
        losses = loss_fn(model, S_int, t_int, S_bc, t_bc, S_term)
    
    with profiler.section("backward_pass"):
        losses["total"].backward()
    
    with profiler.section("optimizer_step"):
        optimizer.step()
    
    return losses["total"].item()


def run_profiled_training(args):
    """Run training with detailed profiling."""
    
    print("╔" + "═" * 58 + "╗")
    print("║" + " Profiled Training with Compute Infrastructure ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Initialize profiler
    profiler = Profiler(track_memory=True)
    
    # Check available compute resources
    print("\n📊 Compute Resource Discovery:")
    router = get_router()
    for name, backend in router.backends.items():
        caps = backend.get_capabilities()
        print(f"  • {name}: {caps.device_type.name}, {caps.num_cores} cores, "
              f"{caps.memory_gb:.1f} GB RAM")
    
    # Routing decision for quantum workload
    if args.model == "hybrid":
        decision = route_circuit(args.n_qubits, args.n_layers * 10)
        print(f"\n🔄 Routing {args.n_qubits}-qubit circuit to: "
              f"{decision.backend.get_capabilities().device_id}")
        print(f"   Estimated time per circuit: {decision.estimated_cost:.4f}s")
    
    # Setup
    print(f"\n⚙️  Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    if args.model == "hybrid":
        print(f"  Qubits: {args.n_qubits}, Layers: {args.n_layers}")
    
    # Create model and training components
    with profiler.section("setup"):
        model = create_model(args)
        params = BSParams(r=args.r, sigma=args.sigma, K=args.K, T=args.T)
        loss_fn = PINNLoss(params)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # Training loop with profiling
    print(f"\n🏋️  Training ({args.epochs} epochs):")
    print("-" * 60)
    
    history = {"loss": [], "epoch_time": []}
    
    for epoch in range(args.epochs):
        with profiler.section("epoch"):
            epoch_start = time.perf_counter()
            loss = profile_single_epoch(model, loss_fn, optimizer, params, profiler, args)
            epoch_time = time.perf_counter() - epoch_start
        
        history["loss"].append(loss)
        history["epoch_time"].append(epoch_time)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d}: loss={loss:.6f}, time={epoch_time:.3f}s")
    
    print("-" * 60)
    
    # Evaluation
    print("\n📈 Evaluating...")
    model.eval()
    S_test = torch.linspace(1.0, 200.0, 100)
    t_test = torch.zeros_like(S_test)
    
    with torch.no_grad():
        V_pred = model(S_test, t_test).numpy()
    V_true = bs_analytical(S_test, t_test, params).numpy()
    
    mse = float(np.mean((V_pred - V_true) ** 2))
    mae = float(np.mean(np.abs(V_pred - V_true)))
    
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    # Profiling report
    print("\n" + profiler.report())
    
    # Timing breakdown
    result = profiler.get_result()
    total_time = result.total_time
    
    print("\n⏱️  Timing Breakdown:")
    print("-" * 60)
    
    component_times = {}
    for name, stats in result.function_stats.items():
        if name not in ["epoch", "setup"]:
            component_times[name] = stats.duration
    
    for name, duration in sorted(component_times.items(), key=lambda x: -x[1]):
        pct = 100 * duration / total_time if total_time > 0 else 0
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        print(f"  {name:20s} {bar} {pct:5.1f}% ({duration:.2f}s)")
    
    print("-" * 60)
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Average epoch time:  {np.mean(history['epoch_time']):.4f}s")
    print(f"  Throughput:          {args.epochs / total_time:.1f} epochs/s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "final_loss": history["loss"][-1],
        "mse": mse,
        "mae": mae,
        "total_time_s": total_time,
        "avg_epoch_time_s": float(np.mean(history["epoch_time"])),
        "timing_breakdown": {
            name: {
                "duration_s": stats.duration,
                "pct": 100 * stats.duration / total_time if total_time > 0 else 0,
                "call_count": stats.call_count,
            }
            for name, stats in result.function_stats.items()
        },
    }
    
    if args.model == "hybrid":
        results["n_qubits"] = args.n_qubits
        results["n_layers"] = args.n_layers
    
    with open(output_dir / "profiled_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n✅ Done!")
    
    return results


def main():
    args = parse_args()
    run_profiled_training(args)


if __name__ == "__main__":
    main()
