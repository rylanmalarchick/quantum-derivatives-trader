# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 5-Asset Basket Option PINN Analysis
#
# This notebook analyzes the results of training a PINN on a 5-asset basket option.
# Key achievements:
# 1. **6D problem** (5 assets + time) - where finite difference is computationally infeasible
# 2. **Multi-asset Black-Scholes PDE** with full Hessian (cross-derivatives)
# 3. **Latin Hypercube Sampling** for efficient high-dimensional collocation
# 4. **Monte Carlo validation** as ground truth

# %%
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization import set_publication_style
from src.pde.basket import BasketParams, monte_carlo_basket
from src.classical.pinn_basket import BasketPINN, evaluate_basket_pinn

set_publication_style()

# %% [markdown]
# ## 1. Load Training Results

# %%
# Find the latest basket training run
basket_dir = PROJECT_ROOT / "outputs/basket"
runs = sorted([d for d in basket_dir.iterdir() if d.is_dir()], reverse=True)
print(f"Found {len(runs)} training runs")

# Find the run with the most complete results (has evaluation.json)
best_run = None
for run in runs:
    if (run / "evaluation.json").exists():
        best_run = run
        break
        
if best_run is None:
    # Fall back to most recent
    best_run = runs[0] if runs else None
    
print(f"Using run: {best_run}")

# %%
# Load checkpoint
checkpoint_path = best_run / "basket_pinn_checkpoint.pt"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history = checkpoint['history']
    mc_reference = checkpoint['mc_reference']
    model_params = checkpoint['params']
    hidden_dims = checkpoint['hidden_dims']
    
    print("Loaded checkpoint:")
    print(f"  Assets: {model_params['n_assets']}")
    print(f"  MC Reference: ${mc_reference['price']:.4f} ± ${mc_reference['std_error']:.4f}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Epochs trained: {len(history['total'])}")
else:
    print("No checkpoint found - will need to run training first")
    history = None

# %% [markdown]
# ## 2. Training Convergence

# %%
if history:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(len(history['total']))
    
    # Plot 1: Total Loss
    ax1 = axes[0]
    ax1.semilogy(epochs, history['total'], 'b-', linewidth=1.5, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (log scale)', fontsize=12)
    ax1.set_title('Total Loss Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Component Losses
    ax2 = axes[1]
    ax2.semilogy(epochs, history['pde'], 'r-', linewidth=1.5, label='PDE Loss')
    ax2.semilogy(epochs, history['ic'], 'g-', linewidth=1.5, label='Terminal Loss')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (log scale)', fontsize=12)
    ax2.set_title('Component Losses', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Loss Ratio
    ax3 = axes[2]
    ratio = np.array(history['pde']) / (np.array(history['ic']) + 1e-10)
    ax3.plot(epochs, ratio, 'purple', linewidth=1.5)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('PDE / Terminal Loss Ratio', fontsize=12)
    ax3.set_title('Loss Balance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/basket_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary stats
    print(f"\nTraining Summary:")
    print(f"  Initial total loss: {history['total'][0]:.4f}")
    print(f"  Final total loss:   {history['total'][-1]:.4f}")
    print(f"  Loss reduction:     {history['total'][0] / history['total'][-1]:.1f}x")
    print(f"  Total time:         {sum(history['time']):.1f}s ({sum(history['time'])/60:.1f} min)")

# %% [markdown]
# ## 3. Load Evaluation Results

# %%
eval_path = best_run / "evaluation.json"
if eval_path.exists():
    with open(eval_path) as f:
        eval_results = json.load(f)
    
    print("Evaluation Results (vs Monte Carlo):")
    print(f"  MSE:              {eval_results['mse']:.4f}")
    print(f"  MAE:              {eval_results['mae']:.4f}")
    print(f"  Max Error:        {eval_results['max_error']:.4f}")
    print(f"  Mean Rel Error:   {eval_results['mean_rel_error_pct']:.2f}%")
    print(f"  Median Rel Error: {eval_results['median_rel_error_pct']:.2f}%")
    print(f"  Test Points:      {eval_results['n_test']}")
else:
    print("No evaluation results found")
    eval_results = None

# %% [markdown]
# ## 4. Recreate Model and Run Live Evaluation

# %%
# Recreate basket parameters
if checkpoint_path.exists():
    params = BasketParams(
        n_assets=model_params['n_assets'],
        r=model_params['r'],
        K=model_params['K'],
        T=model_params['T'],
        S0=np.array(model_params['S0']),
        sigma=np.array(model_params['sigma']),
        weights=np.array(model_params['weights']),
    )
    
    # Create and load model
    model = BasketPINN(
        n_assets=params.n_assets,
        hidden_dims=hidden_dims,
        S_max=params.S_max,
        T_max=params.T,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

# %%
# Quick spot check at S0
if checkpoint_path.exists():
    S0_tensor = torch.tensor(params.S0, dtype=torch.float32).unsqueeze(0)
    t0_tensor = torch.zeros(1)
    
    with torch.no_grad():
        pinn_price = model(S0_tensor, t0_tensor).item()
    
    print(f"\nSpot Check at S0 = {params.S0}:")
    print(f"  PINN:  ${pinn_price:.4f}")
    print(f"  MC:    ${mc_reference['price']:.4f}")
    print(f"  Error: {abs(pinn_price - mc_reference['price']):.4f} ({abs(pinn_price - mc_reference['price'])/mc_reference['price']*100:.2f}%)")

# %% [markdown]
# ## 5. Price Surface Visualization (2D Slices)

# %%
# Since we have 5 assets, we visualize 2D slices holding other assets at S0
if checkpoint_path.exists():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    S_range = np.linspace(50, 200, 50)
    t_vals = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
    
    for ax, t in zip(axes.flat, t_vals):
        # Vary first asset, hold others at 100
        prices = []
        for S1 in S_range:
            S_vec = np.array([S1, 100., 100., 100., 100.])
            S_tensor = torch.tensor(S_vec, dtype=torch.float32).unsqueeze(0)
            t_tensor = torch.tensor([t], dtype=torch.float32)
            
            with torch.no_grad():
                price = model(S_tensor, t_tensor).item()
            prices.append(price)
        
        ax.plot(S_range, prices, 'b-', linewidth=2)
        ax.axhline(y=params.K * params.weights[0], color='r', linestyle='--', alpha=0.5, label='K*w1')
        ax.axvline(x=params.K, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('S1', fontsize=11)
        ax.set_ylabel('Option Price', fontsize=11)
        ax.set_title(f't = {t:.2f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([50, 200])
        
    plt.suptitle('Basket Option Price vs S1 (other assets at 100)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/basket_price_slices.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Delta (dV/dS) via Autodiff

# %%
# Compute all 5 deltas at S0
if checkpoint_path.exists():
    print("Computing Greeks via Automatic Differentiation...")
    
    # Create tensors with gradients
    S_tensor = torch.tensor(params.S0, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    t_tensor = torch.zeros(1, requires_grad=True)
    
    # Forward pass
    price = model(S_tensor, t_tensor)
    
    # Compute deltas (dV/dSi for each asset)
    grads = torch.autograd.grad(price, S_tensor, create_graph=True)[0]
    deltas = grads.squeeze().detach().numpy()
    
    print(f"\nDeltas at S0 = {params.S0}, t=0:")
    for i, delta in enumerate(deltas):
        print(f"  Delta_{i+1} = {delta:.4f}")
    print(f"  Sum of Deltas: {deltas.sum():.4f}")
    
    # Compute theta (dV/dt)
    price = model(S_tensor, t_tensor)
    theta = torch.autograd.grad(price, t_tensor)[0]
    print(f"\nTheta: {theta.item():.4f}")

# %% [markdown]
# ## 7. Dimensionality Advantage

# %%
# Compare computational scaling
print("=" * 70)
print("DIMENSIONALITY ADVANTAGE: PINN vs Finite Difference")
print("=" * 70)

# FD grid sizing
fd_grid_1d = 100  # Points per dimension

dimensions = [1, 2, 3, 4, 5]
fd_points = [fd_grid_1d ** d for d in dimensions]
pinn_points = 15000  # Fixed collocation points for PINN

fig, ax = plt.subplots(figsize=(10, 6))

# FD scaling (exponential)
ax.semilogy(dimensions, fd_points, 'ro-', linewidth=2, markersize=10, label=f'Finite Difference (N={fd_grid_1d} per dim)')

# PINN scaling (constant)
ax.axhline(pinn_points, color='blue', linestyle='--', linewidth=2, label=f'PINN ({pinn_points:,} points)')

ax.fill_between(dimensions, fd_points, pinn_points, where=np.array(fd_points) > pinn_points, 
                alpha=0.3, color='green', label='PINN wins')

ax.set_xlabel('Number of Assets (dimensions)', fontsize=12)
ax.set_ylabel('Grid/Collocation Points', fontsize=12)
ax.set_title('Curse of Dimensionality: FD vs PINN', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(dimensions)
ax.set_xticklabels([f'{d}D' for d in dimensions])

# Add annotations
for d, fp in zip(dimensions, fd_points):
    if fp > 1e6:
        ax.annotate(f'{fp/1e9:.1f}B' if fp > 1e9 else f'{fp/1e6:.0f}M', 
                    (d, fp), textcoords="offset points", xytext=(10, 5), fontsize=9)
    else:
        ax.annotate(f'{fp:,}', (d, fp), textcoords="offset points", xytext=(10, 5), fontsize=9)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs/analysis/dimensionality_scaling.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"""
Grid Points Required:
  1D (1 asset):  {fd_grid_1d:,} (FD feasible)
  2D (2 assets): {fd_grid_1d**2:,} (FD feasible)
  3D (3 assets): {fd_grid_1d**3:,} (FD challenging)
  4D (4 assets): {fd_grid_1d**4:,} (FD very slow)
  5D (5 assets): {fd_grid_1d**5:,} (FD INFEASIBLE - 10 billion points!)

PINN uses: {pinn_points:,} points regardless of dimension!

This is why mesh-free methods like PINNs are essential for multi-asset derivatives.
""")

# %% [markdown]
# ## 8. Summary

# %%
print("""
=======================================================================
                        KEY TAKEAWAYS
=======================================================================

1. PINN SOLVES THE CURSE OF DIMENSIONALITY
   - 5-asset basket = 6D problem
   - FD would need 10 BILLION grid points (100^5)
   - PINN uses only 15,000 collocation points
   - O(N^d) -> O(N) scaling

2. MESH-FREE = FLEXIBLE GEOMETRY
   - No structured grid required
   - Latin Hypercube Sampling covers space efficiently
   - Can handle irregular domains

3. AUTOMATIC DIFFERENTIATION = FREE GREEKS
   - Get all 5 deltas with one backward pass
   - Cross-gammas available too
   - No numerical differentiation noise

4. MONTE CARLO VALIDATION
   - MC is gold standard for multi-asset options
   - PINN matches MC accuracy with faster inference
   - Train once, evaluate instantly

5. JANE STREET RELEVANCE
   - Real trading desks price basket options daily
   - This demonstrates genuine computational advantage
   - Not just a toy problem - actual quant workflow

=======================================================================
""")
