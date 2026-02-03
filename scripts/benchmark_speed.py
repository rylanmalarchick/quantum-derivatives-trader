#!/usr/bin/env python3
"""
Speed benchmarking script for option pricing methods.

Compares wall-clock time for option pricing across methods:
1. PINN inference - Forward pass through trained network
2. Monte Carlo - Standard MC simulation
3. Finite Difference - Grid-based PDE solver
4. Analytical - Black-Scholes formula (baseline)

Key insight: PINNs are slow to train but fast at inference. This script
demonstrates the crossover point where PINN becomes faster than MC for
batch pricing.

Usage:
    python scripts/benchmark_speed.py
    python scripts/benchmark_speed.py --skip_pinn  # Skip PINN if no trained model
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pricing.analytical import AnalyticalPricer
from src.pricing.finite_difference import FiniteDifferencePricer, FDGrid
from src.pricing.monte_carlo import MonteCarloEngine
from src.classical.pinn import PINN
from src.pde.black_scholes import BSParams


@dataclass
class TimingResult:
    """Result from timing a single method."""
    method: str
    batch_size: int
    n_runs: int
    mean_time_ms: float
    std_time_ms: float
    throughput: float  # options per second
    extra_info: dict = field(default_factory=dict)


@dataclass
class BenchmarkParams:
    """Common parameters for all pricing methods."""
    S: float = 100.0      # Spot price
    K: float = 100.0      # Strike
    T: float = 1.0        # Time to maturity
    r: float = 0.05       # Risk-free rate
    sigma: float = 0.2    # Volatility
    S_max: float = 200.0  # Max spot for PINN normalization


def benchmark_pinn(
    model: PINN,
    S_batch: torch.Tensor,
    t_batch: torch.Tensor,
    n_runs: int = 100,
    device: torch.device = torch.device("cpu"),
) -> TimingResult:
    """
    Benchmark PINN forward pass (inference time).

    Args:
        model: Trained PINN model
        S_batch: Batch of spot prices
        t_batch: Batch of times
        n_runs: Number of timing runs
        device: Computation device

    Returns:
        TimingResult with mean/std timing
    """
    model.eval()
    S_batch = S_batch.to(device)
    t_batch = t_batch.to(device)
    batch_size = len(S_batch)

    # Warmup run
    with torch.no_grad():
        _ = model(S_batch, t_batch)
    
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(S_batch, t_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0

    return TimingResult(
        method="PINN Inference",
        batch_size=batch_size,
        n_runs=n_runs,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput=throughput,
    )


def benchmark_monte_carlo(
    params: BenchmarkParams,
    batch_size: int = 1,
    n_paths: int = 10000,
    n_runs: int = 10,
) -> TimingResult:
    """
    Benchmark Monte Carlo pricing.

    Args:
        params: Black-Scholes parameters
        batch_size: Number of options to price
        n_paths: Number of MC simulation paths
        n_runs: Number of timing runs

    Returns:
        TimingResult with mean/std timing
    """
    mc = MonteCarloEngine(r=params.r, sigma=params.sigma, seed=42)
    
    def call_payoff(S_T: float) -> float:
        return max(S_T - params.K, 0.0)

    # Generate batch of spot prices around ATM
    S_batch = np.linspace(params.K * 0.8, params.K * 1.2, batch_size)

    # Warmup run
    for S0 in S_batch:
        mc.price_european(call_payoff, S0=float(S0), T=params.T, n_paths=n_paths)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for S0 in S_batch:
            mc.price_european(call_payoff, S0=float(S0), T=params.T, n_paths=n_paths)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0

    return TimingResult(
        method=f"Monte Carlo ({n_paths:,} paths)",
        batch_size=batch_size,
        n_runs=n_runs,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput=throughput,
        extra_info={"n_paths": n_paths},
    )


def benchmark_finite_difference(
    params: BenchmarkParams,
    batch_size: int = 1,
    n_S: int = 100,
    n_t: int = 1000,
    n_runs: int = 10,
) -> TimingResult:
    """
    Benchmark Finite Difference PDE solver.

    Args:
        params: Black-Scholes parameters
        batch_size: Number of spot prices to evaluate (interpolated from grid)
        n_S: Number of spatial grid points
        n_t: Number of time grid points
        n_runs: Number of timing runs

    Returns:
        TimingResult with mean/std timing
    """
    grid = FDGrid(S_min=0.0, S_max=params.S_max, n_S=n_S, n_t=n_t)
    fd = FiniteDifferencePricer(r=params.r, sigma=params.sigma, grid=grid)

    # Generate batch of spot prices to evaluate
    S_batch = np.linspace(params.K * 0.8, params.K * 1.2, batch_size)

    # Warmup run
    S_grid, V_grid = fd.price_crank_nicolson(K=params.K, T=params.T, option_type="call")
    _ = np.interp(S_batch, S_grid, V_grid)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        S_grid, V_grid = fd.price_crank_nicolson(K=params.K, T=params.T, option_type="call")
        # Interpolate to get prices at specific spots
        _ = np.interp(S_batch, S_grid, V_grid)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0

    return TimingResult(
        method=f"Finite Difference ({n_S}x{n_t} grid)",
        batch_size=batch_size,
        n_runs=n_runs,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput=throughput,
        extra_info={"n_S": n_S, "n_t": n_t},
    )


def benchmark_analytical(
    params: BenchmarkParams,
    batch_size: int = 1,
    n_runs: int = 1000,
) -> TimingResult:
    """
    Benchmark analytical Black-Scholes formula.

    Args:
        params: Black-Scholes parameters
        batch_size: Number of options to price
        n_runs: Number of timing runs

    Returns:
        TimingResult with mean/std timing
    """
    pricer = AnalyticalPricer(r=params.r, sigma=params.sigma)

    # Generate batch of spot prices
    S_batch = np.linspace(params.K * 0.8, params.K * 1.2, batch_size)

    # Warmup run
    _ = pricer.black_scholes(S_batch, params.K, params.T)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pricer.black_scholes(S_batch, params.K, params.T)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0

    return TimingResult(
        method="Analytical (BS)",
        batch_size=batch_size,
        n_runs=n_runs,
        mean_time_ms=mean_time,
        std_time_ms=std_time,
        throughput=throughput,
    )


def create_or_load_pinn(
    params: BenchmarkParams,
    model_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    train_epochs: int = 1000,
) -> Optional[PINN]:
    """
    Load a trained PINN or create and train a new one.

    Args:
        params: Black-Scholes parameters
        model_path: Path to saved model weights
        device: Computation device
        train_epochs: Training epochs if creating new model

    Returns:
        Trained PINN model or None if loading failed and skip training
    """
    model = PINN(
        hidden_dims=[64, 64, 64, 64],
        S_max=params.S_max,
        T_max=params.T,
        use_residual=False,
    ).to(device)

    # Try to load existing model
    if model_path and model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded PINN model from {model_path}")
            return model
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")

    # Train a new model if no saved model found
    print(f"\nTraining PINN model ({train_epochs} epochs)...")
    print("Note: Training time is NOT included in inference benchmark.\n")

    from src.classical.pinn import PINNTrainer
    
    bs_params = BSParams(r=params.r, sigma=params.sigma, K=params.K, T=params.T)
    trainer = PINNTrainer(model=model, params=bs_params, lr=1e-3)
    
    trainer.train(
        n_epochs=train_epochs,
        n_interior=1000,
        n_boundary=200,
        n_terminal=200,
        print_every=train_epochs // 5,
    )

    return model


def run_batch_scaling_benchmark(
    params: BenchmarkParams,
    pinn_model: Optional[PINN],
    device: torch.device,
    batch_sizes: list[int] = [1, 100, 1000, 10000],
) -> dict[str, list[TimingResult]]:
    """
    Run benchmarks across different batch sizes.

    Returns:
        Dictionary mapping method names to lists of TimingResult
    """
    results = {
        "Analytical": [],
        "PINN": [],
        "Monte Carlo": [],
        "Finite Difference": [],
    }

    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")

        # Analytical
        result = benchmark_analytical(params, batch_size=batch_size, n_runs=1000)
        results["Analytical"].append(result)
        print(f"Analytical: {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")

        # PINN
        if pinn_model is not None:
            S_batch = torch.linspace(params.K * 0.8, params.K * 1.2, batch_size)
            t_batch = torch.zeros(batch_size)
            result = benchmark_pinn(pinn_model, S_batch, t_batch, n_runs=100, device=device)
            results["PINN"].append(result)
            print(f"PINN: {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")
        else:
            results["PINN"].append(None)
            print("PINN: skipped (no model)")

        # Monte Carlo (fewer runs for larger batches due to cost)
        n_runs = 10 if batch_size <= 100 else 3
        result = benchmark_monte_carlo(
            params, batch_size=batch_size, n_paths=10000, n_runs=n_runs
        )
        results["Monte Carlo"].append(result)
        print(f"Monte Carlo: {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")

        # Finite Difference
        n_runs = 10 if batch_size <= 1000 else 5
        result = benchmark_finite_difference(
            params, batch_size=batch_size, n_S=100, n_t=1000, n_runs=n_runs
        )
        results["Finite Difference"].append(result)
        print(f"Finite Difference: {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")

    return results


def run_mc_paths_scaling_benchmark(
    params: BenchmarkParams,
    n_paths_list: list[int] = [1000, 10000, 100000],
) -> list[TimingResult]:
    """
    Benchmark MC with varying number of paths.
    """
    results = []
    print("\n--- Monte Carlo: Varying number of paths ---")

    for n_paths in n_paths_list:
        result = benchmark_monte_carlo(params, batch_size=10, n_paths=n_paths, n_runs=10)
        results.append(result)
        print(f"MC ({n_paths:,} paths): {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")

    return results


def run_fd_grid_scaling_benchmark(
    params: BenchmarkParams,
    grid_configs: list[tuple[int, int]] = [(50, 500), (100, 1000), (200, 2000)],
) -> list[TimingResult]:
    """
    Benchmark FD with varying grid sizes.
    """
    results = []
    print("\n--- Finite Difference: Varying grid sizes ---")

    for n_S, n_t in grid_configs:
        result = benchmark_finite_difference(
            params, batch_size=10, n_S=n_S, n_t=n_t, n_runs=10
        )
        results.append(result)
        print(f"FD ({n_S}x{n_t}): {result.mean_time_ms:.4f} +/- {result.std_time_ms:.4f} ms")

    return results


def print_summary_table(
    batch_results: dict[str, list[TimingResult]],
    batch_sizes: list[int],
) -> None:
    """Print formatted summary table of benchmark results."""
    print("\n" + "=" * 100)
    print("SPEED BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    header = f"{'Method':<25}"
    for bs in batch_sizes:
        header += f" {'Batch=' + str(bs):>15}"
    print(header)
    print("-" * 100)

    for method, results in batch_results.items():
        row = f"{method:<25}"
        for result in results:
            if result is None:
                row += f" {'N/A':>15}"
            else:
                row += f" {result.mean_time_ms:>12.4f} ms"
        print(row)

    print("=" * 100)

    # Throughput table
    print("\n" + "=" * 100)
    print("THROUGHPUT (options/second)")
    print("=" * 100)

    header = f"{'Method':<25}"
    for bs in batch_sizes:
        header += f" {'Batch=' + str(bs):>15}"
    print(header)
    print("-" * 100)

    for method, results in batch_results.items():
        row = f"{method:<25}"
        for result in results:
            if result is None:
                row += f" {'N/A':>15}"
            else:
                row += f" {result.throughput:>13,.0f}"
        print(row)

    print("=" * 100)


def create_benchmark_plots(
    batch_results: dict[str, list[TimingResult]],
    batch_sizes: list[int],
    mc_results: list[TimingResult],
    fd_results: list[TimingResult],
    output_dir: Path,
) -> None:
    """Create visualization plots for benchmark results."""
    plt.style.use('default')
    
    # 1. Time vs Batch Size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors = {'Analytical': 'green', 'PINN': 'blue', 'Monte Carlo': 'red', 'Finite Difference': 'orange'}
    
    for method, results in batch_results.items():
        valid_sizes = []
        valid_times = []
        valid_stds = []
        for i, result in enumerate(results):
            if result is not None:
                valid_sizes.append(batch_sizes[i])
                valid_times.append(result.mean_time_ms)
                valid_stds.append(result.std_time_ms)
        
        if valid_sizes:
            ax1.errorbar(valid_sizes, valid_times, yerr=valid_stds, 
                        label=method, marker='o', color=colors.get(method, 'gray'),
                        linewidth=2, markersize=8, capsize=3)

    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Execution Time vs Batch Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Throughput comparison bar chart (for largest batch)
    ax2 = axes[1]
    methods = []
    throughputs = []
    bar_colors = []
    
    for method, results in batch_results.items():
        if results[-1] is not None:  # Last batch size (largest)
            methods.append(method)
            throughputs.append(results[-1].throughput)
            bar_colors.append(colors.get(method, 'gray'))

    bars = ax2.bar(methods, throughputs, color=bar_colors)
    ax2.set_ylabel('Throughput (options/second)', fontsize=12)
    ax2.set_title(f'Throughput at Batch Size = {batch_sizes[-1]}', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, tp in zip(bars, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{tp:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "speed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. MC paths scaling
    if mc_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        n_paths = [r.extra_info.get("n_paths", 0) for r in mc_results]
        times = [r.mean_time_ms for r in mc_results]
        stds = [r.std_time_ms for r in mc_results]
        
        ax.errorbar(n_paths, times, yerr=stds, marker='o', color='red', 
                   linewidth=2, markersize=8, capsize=3)
        ax.set_xlabel('Number of MC Paths', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Monte Carlo: Time vs Number of Paths', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / "mc_paths_scaling.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. FD grid scaling
    if fd_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        grid_labels = [f"{r.extra_info.get('n_S', 0)}x{r.extra_info.get('n_t', 0)}" for r in fd_results]
        times = [r.mean_time_ms for r in fd_results]
        stds = [r.std_time_ms for r in fd_results]
        
        x = np.arange(len(grid_labels))
        ax.bar(x, times, yerr=stds, color='orange', capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(grid_labels)
        ax.set_xlabel('Grid Size (n_S x n_t)', fontsize=12)
        ax.set_ylabel('Time (ms)', fontsize=12)
        ax.set_title('Finite Difference: Time vs Grid Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(output_dir / "fd_grid_scaling.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to: {output_dir}")


def save_results_json(
    batch_results: dict[str, list[TimingResult]],
    batch_sizes: list[int],
    mc_results: list[TimingResult],
    fd_results: list[TimingResult],
    params: BenchmarkParams,
    output_path: Path,
) -> None:
    """Save all benchmark results to JSON."""
    data = {
        "parameters": asdict(params),
        "batch_sizes": batch_sizes,
        "batch_scaling": {},
        "mc_paths_scaling": [],
        "fd_grid_scaling": [],
    }

    # Batch scaling results
    for method, results in batch_results.items():
        data["batch_scaling"][method] = []
        for result in results:
            if result is not None:
                data["batch_scaling"][method].append({
                    "batch_size": result.batch_size,
                    "mean_time_ms": result.mean_time_ms,
                    "std_time_ms": result.std_time_ms,
                    "throughput": result.throughput,
                    "n_runs": result.n_runs,
                })
            else:
                data["batch_scaling"][method].append(None)

    # MC paths scaling
    for result in mc_results:
        data["mc_paths_scaling"].append({
            "n_paths": result.extra_info.get("n_paths", 0),
            "mean_time_ms": result.mean_time_ms,
            "std_time_ms": result.std_time_ms,
            "throughput": result.throughput,
        })

    # FD grid scaling
    for result in fd_results:
        data["fd_grid_scaling"].append({
            "n_S": result.extra_info.get("n_S", 0),
            "n_t": result.extra_info.get("n_t", 0),
            "mean_time_ms": result.mean_time_ms,
            "std_time_ms": result.std_time_ms,
            "throughput": result.throughput,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_crossover_analysis(
    batch_results: dict[str, list[TimingResult]],
    batch_sizes: list[int],
) -> None:
    """Analyze and print the crossover point where PINN beats MC."""
    print("\n" + "=" * 70)
    print("CROSSOVER ANALYSIS: PINN vs Monte Carlo")
    print("=" * 70)

    pinn_results = batch_results.get("PINN", [])
    mc_results = batch_results.get("Monte Carlo", [])

    if not any(r is not None for r in pinn_results):
        print("PINN results not available - skipping crossover analysis.")
        return

    crossover_found = False
    for i, batch_size in enumerate(batch_sizes):
        pinn_result = pinn_results[i] if i < len(pinn_results) else None
        mc_result = mc_results[i] if i < len(mc_results) else None

        if pinn_result is None or mc_result is None:
            continue

        pinn_time = pinn_result.mean_time_ms
        mc_time = mc_result.mean_time_ms
        speedup = mc_time / pinn_time if pinn_time > 0 else 0

        if speedup >= 1 and not crossover_found:
            print(f"\nCrossover point found at batch size = {batch_size}")
            crossover_found = True

        print(f"Batch {batch_size:>6}: PINN = {pinn_time:>8.4f} ms, MC = {mc_time:>8.4f} ms, "
              f"PINN speedup = {speedup:>6.2f}x")

    print()
    print("Key Insight: PINNs have higher upfront training cost, but once trained,")
    print("inference is very fast. For large batch pricing (risk calculations,")
    print("portfolio valuation), PINN inference can be orders of magnitude faster")
    print("than running Monte Carlo simulations for each option.")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Speed Benchmark for Option Pricing Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--skip_pinn", action="store_true",
                       help="Skip PINN benchmarks (if no trained model available)")
    parser.add_argument("--pinn_epochs", type=int, default=1000,
                       help="Training epochs for PINN if no saved model")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved PINN model weights")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks",
                       help="Output directory for results")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--device", type=str, default="auto",
                       help="Computation device (auto, cpu, cuda)")

    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("\n" + "=" * 70)
    print("OPTION PRICING SPEED BENCHMARK")
    print("=" * 70)
    print(f"\nDevice: {device}")

    # Benchmark parameters
    params = BenchmarkParams()
    print(f"\nBlack-Scholes Parameters:")
    print(f"  S = {params.S}, K = {params.K}, T = {params.T}")
    print(f"  r = {params.r}, sigma = {params.sigma}")

    # Load or create PINN model
    pinn_model = None
    if not args.skip_pinn:
        model_path = Path(args.model_path) if args.model_path else None
        try:
            pinn_model = create_or_load_pinn(
                params, model_path, device, train_epochs=args.pinn_epochs
            )
        except Exception as e:
            print(f"\nWarning: Could not create/load PINN model: {e}")
            print("Continuing without PINN benchmarks.\n")
    else:
        print("\nSkipping PINN benchmarks as requested.")

    # Define batch sizes to test
    batch_sizes = [1, 100, 1000, 10000]

    # Run batch scaling benchmarks
    print("\n" + "=" * 70)
    print("BATCH SCALING BENCHMARK")
    print("=" * 70)
    batch_results = run_batch_scaling_benchmark(params, pinn_model, device, batch_sizes)

    # Run MC paths scaling
    mc_results = run_mc_paths_scaling_benchmark(params, [1000, 10000, 100000])

    # Run FD grid scaling
    fd_results = run_fd_grid_scaling_benchmark(params, [(50, 500), (100, 1000), (200, 2000)])

    # Print summary
    print_summary_table(batch_results, batch_sizes)

    # Crossover analysis
    print_crossover_analysis(batch_results, batch_sizes)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results_json(
        batch_results, batch_sizes, mc_results, fd_results, params,
        output_dir / "speed_benchmark.json"
    )

    # Generate plots
    if not args.no_plots:
        print("\nGenerating benchmark plots...")
        create_benchmark_plots(batch_results, batch_sizes, mc_results, fd_results, output_dir)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
