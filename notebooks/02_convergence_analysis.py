#!/usr/bin/env python3
"""
Convergence Analysis: Classical vs Hybrid PINN

This script/notebook analyzes and compares the convergence behavior of:
1. Classical PINN (MLP-based)
2. Hybrid Quantum-Classical PINN (VQC-based)

Produces publication-quality plots for:
- Training loss convergence
- Final accuracy comparison
- Computational cost analysis
- Scaling behavior

Can be run as a script or opened in Jupyter with: jupytext --to notebook 02_convergence_analysis.py

Usage:
    python notebooks/02_convergence_analysis.py
"""

# %% Imports
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root
NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization import set_publication_style


# %% Load results from training runs
def load_training_results(output_dir: Path) -> dict:
    """Load training results from output directory."""
    results = {}
    
    # Load training history
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            results["history"] = json.load(f)
    
    # Load evaluation metrics
    metrics_path = output_dir / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)
    
    # Alternative: results.json for hybrid
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            if "hybrid_pinn" in data:
                results["metrics"] = data["hybrid_pinn"]
                results["training_time"] = data.get("training_time_seconds", 0)
                results["quantum_config"] = data.get("quantum_config", {})
    
    return results


def find_latest_run(base_dir: Path) -> Path:
    """Find the most recent run in an output directory."""
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    return runs[0] if runs else None


# %% Analysis functions
def compute_convergence_rate(losses: list, window: int = 50) -> np.ndarray:
    """Compute moving average convergence rate (improvement per epoch)."""
    losses = np.array(losses)
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    rate = -np.diff(np.log(smoothed + 1e-10))  # Log-scale improvement
    return rate


def compute_efficiency_metric(losses: list, times: list) -> np.ndarray:
    """Compute loss improvement per second."""
    losses = np.array(losses)
    times = np.array(times)
    cumulative_time = np.cumsum(times)
    
    # Improvement rate
    improvement = losses[0] - losses
    efficiency = improvement / cumulative_time
    return efficiency


# %% Main analysis
def run_analysis():
    """Run convergence analysis on available results."""
    set_publication_style()
    
    print("=" * 60)
    print("Convergence Analysis: Classical vs Hybrid PINN")
    print("=" * 60)
    
    # Find results
    classical_dir = PROJECT_ROOT / "outputs" / "classical"
    hybrid_dir = PROJECT_ROOT / "outputs" / "hybrid"
    
    results = {}
    
    if classical_dir.exists():
        latest_classical = find_latest_run(classical_dir)
        if latest_classical:
            results["Classical PINN"] = load_training_results(latest_classical)
            print(f"✓ Found classical results: {latest_classical.name}")
    
    if hybrid_dir.exists():
        latest_hybrid = find_latest_run(hybrid_dir)
        if latest_hybrid:
            results["Hybrid PINN"] = load_training_results(latest_hybrid)
            print(f"✓ Found hybrid results: {latest_hybrid.name}")
    
    if not results:
        print("❌ No training results found!")
        print("   Run training first:")
        print("   python scripts/train_classical.py --epochs 1000")
        print("   python scripts/train_hybrid.py --epochs 500")
        return
    
    # Print summary
    print("\n" + "-" * 60)
    print("Results Summary:")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"\n{name}:")
        if "history" in data:
            losses = data["history"].get("total", [])
            print(f"  Epochs: {len(losses)}")
            print(f"  Final loss: {losses[-1]:.6f}" if losses else "  No loss data")
        if "metrics" in data:
            metrics = data["metrics"]
            print(f"  MSE: {metrics.get('mse', 'N/A'):.6f}" if 'mse' in metrics else "")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.6f}" if 'mae' in metrics else "")
            print(f"  Mean rel. error: {metrics.get('mean_rel_error_pct', 'N/A'):.2f}%" 
                  if 'mean_rel_error_pct' in metrics else "")
        if "training_time" in data:
            print(f"  Training time: {data['training_time']:.1f}s")
    
    # Create plots if we have multiple results
    if len(results) >= 1:
        create_convergence_plots(results)


def create_convergence_plots(results: dict):
    """Create comparison plots."""
    print("\n" + "-" * 60)
    print("Generating Plots...")
    print("-" * 60)
    
    output_dir = PROJECT_ROOT / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Training Loss Convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"Classical PINN": "#2ecc71", "Hybrid PINN": "#e74c3c"}
    
    # Linear scale
    ax1 = axes[0]
    for name, data in results.items():
        if "history" in data and "total" in data["history"]:
            losses = data["history"]["total"]
            ax1.plot(losses, label=name, color=colors.get(name, "blue"), linewidth=2, alpha=0.9)
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.set_title("Training Loss (Linear Scale)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2 = axes[1]
    for name, data in results.items():
        if "history" in data and "total" in data["history"]:
            losses = data["history"]["total"]
            ax2.semilogy(losses, label=name, color=colors.get(name, "blue"), linewidth=2, alpha=0.9)
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Total Loss (log scale)", fontsize=12)
    ax2.set_title("Training Loss (Log Scale)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "convergence_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {output_dir / 'convergence_comparison.png'}")
    
    # Figure 2: Loss Components
    if any("history" in d and "pde" in d["history"] for d in results.values()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        components = ["pde", "bc", "ic"]
        titles = ["PDE Residual", "Boundary Condition", "Terminal Condition"]
        
        for ax, comp, title in zip(axes, components, titles):
            for name, data in results.items():
                if "history" in data and comp in data["history"]:
                    losses = data["history"][comp]
                    ax.semilogy(losses, label=name, color=colors.get(name, "blue"), 
                               linewidth=2, alpha=0.9)
            
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / "loss_components.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'loss_components.png'}")
    
    # Figure 3: Accuracy Comparison (bar chart)
    if all("metrics" in d for d in results.values()):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        metrics = ["mse", "mae", "mean_rel_error_pct"]
        titles = ["Mean Squared Error", "Mean Absolute Error", "Mean Relative Error (%)"]
        
        for ax, metric, title in zip(axes, metrics, titles):
            names = []
            values = []
            for name, data in results.items():
                if metric in data["metrics"]:
                    names.append(name.replace(" PINN", ""))
                    values.append(data["metrics"][metric])
            
            bars = ax.bar(names, values, color=[colors.get(f"{n} PINN", "gray") for n in names])
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        fig.savefig(output_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'accuracy_comparison.png'}")
    
    # Figure 4: Time per epoch (if available)
    if any("time_per_epoch" in d.get("history", {}) for d in results.values()):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        for name, data in results.items():
            if "history" in data and "time_per_epoch" in data["history"]:
                times = data["history"]["time_per_epoch"]
                ax.plot(times, label=name, color=colors.get(name, "blue"), linewidth=2, alpha=0.9)
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Time per Epoch (s)", fontsize=12)
        ax.set_title("Computational Cost per Epoch", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / "time_per_epoch.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'time_per_epoch.png'}")
    
    print(f"\nAll plots saved to: {output_dir}")


# %% Entry point
if __name__ == "__main__":
    run_analysis()
