#!/usr/bin/env python
"""
Convergence Analysis: Classical PINN vs Quantum-Classical Hybrid

This script generates publication-quality plots comparing:
1. Training convergence (loss vs epochs)
2. Accuracy vs number of collocation points
3. Accuracy vs network capacity
4. Classical vs quantum expressivity

Run with: python notebooks/01_convergence_analysis.py
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_training_history(path: str) -> dict:
    """Load training history from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_training_curves(classical_path: str, hybrid_path: str = None, save_path: str = None):
    """Plot training loss curves for classical and optionally hybrid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Load classical history
    classical = load_training_history(classical_path)
    epochs = range(len(classical['total_loss']))
    
    # Plot total loss
    axes[0].semilogy(epochs, classical['total_loss'], 'b-', label='Classical PINN', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot component losses
    axes[1].semilogy(epochs, classical['pde_loss'], 'r-', label='PDE Residual', linewidth=2)
    axes[1].semilogy(epochs, classical['bc_loss'], 'g-', label='Boundary', linewidth=2)
    axes[1].semilogy(epochs, classical['ic_loss'], 'b-', label='Terminal', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('Loss Decomposition')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # If hybrid available, plot comparison
    if hybrid_path and Path(hybrid_path).exists():
        hybrid = load_training_history(hybrid_path)
        hybrid_epochs = range(len(hybrid['total_loss']))
        axes[2].semilogy(epochs, classical['total_loss'], 'b-', label='Classical', linewidth=2)
        axes[2].semilogy(hybrid_epochs, hybrid['total_loss'], 'r--', label='Quantum Hybrid', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Total Loss')
        axes[2].set_title('Classical vs Quantum')
        axes[2].legend()
    else:
        # Show convergence rate
        window = 50
        if len(classical['total_loss']) > window:
            smoothed = np.convolve(classical['total_loss'], np.ones(window)/window, mode='valid')
            axes[2].semilogy(range(window-1, len(epochs)), smoothed, 'b-', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Smoothed Loss')
            axes[2].set_title(f'Convergence (window={window})')
    
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_error_vs_collocation(save_path: str = None):
    """
    Plot error vs number of collocation points.
    
    This demonstrates how PINN accuracy scales with sampling density.
    """
    # TODO: Run experiments with varying collocation points
    # For now, create placeholder with expected behavior
    
    n_points = [100, 250, 500, 1000, 2000, 5000]
    mse_expected = [1000, 500, 250, 100, 50, 25]  # Placeholder - replace with actual
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(n_points, mse_expected, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('Number of Collocation Points')
    ax.set_ylabel('MSE vs Analytical')
    ax.set_title('Convergence: Error vs Sampling Density')
    ax.grid(True, which='both', alpha=0.3)
    
    # Add reference line for O(1/N) convergence
    x_ref = np.array([100, 5000])
    y_ref = 50000 / x_ref
    ax.loglog(x_ref, y_ref, 'k--', alpha=0.5, label=r'$O(1/N)$')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_quantum_expressivity(save_path: str = None):
    """
    Compare expressivity of classical vs quantum networks.
    
    Measures: ability to represent complex functions with fewer parameters.
    """
    # TODO: Implement actual expressivity comparison
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Parameter efficiency
    params_classical = [1000, 5000, 10000, 20000]
    mse_classical = [500, 200, 100, 80]
    
    params_quantum = [100, 200, 400, 800]  # Fewer params due to VQC
    mse_quantum = [400, 180, 90, 70]  # Hypothetical
    
    axes[0].loglog(params_classical, mse_classical, 'b-o', label='Classical MLP', markersize=8)
    axes[0].loglog(params_quantum, mse_quantum, 'r-s', label='Quantum Hybrid', markersize=8)
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Parameter Efficiency')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    
    # Training time
    qubits = [2, 4, 6, 8]
    time_per_epoch = [0.1, 0.5, 2.0, 15.0]  # Exponential scaling
    
    axes[1].semilogy(qubits, time_per_epoch, 'r-o', markersize=10, linewidth=2)
    axes[1].set_xlabel('Number of Qubits')
    axes[1].set_ylabel('Time per Epoch (s)')
    axes[1].set_title('Quantum Simulation Cost: O(2^n)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    output_dir = Path('outputs/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find most recent classical training
    classical_dirs = sorted(Path('outputs/classical').glob('*'))
    if classical_dirs:
        latest = classical_dirs[-1]
        history_path = latest / 'training_history.json'
        
        if history_path.exists():
            print(f"Loading: {history_path}")
            plot_training_curves(
                str(history_path),
                save_path=str(output_dir / 'training_curves.png')
            )
    
    # Placeholder analyses
    plot_error_vs_collocation(save_path=str(output_dir / 'convergence_collocation.png'))
    plot_quantum_expressivity(save_path=str(output_dir / 'quantum_expressivity.png'))
    
    print("\nAnalysis complete. Outputs in:", output_dir)
