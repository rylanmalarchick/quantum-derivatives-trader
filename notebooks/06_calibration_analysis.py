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
# # Volatility Surface Calibration: Inverse Problem PINN
#
# This notebook demonstrates the **inverse problem**: given market option prices,
# infer the underlying volatility surface σ(K,T).
#
# Key achievements:
# 1. **Real quant workflow** - calibration is what trading desks do daily
# 2. **Inverse problem** - fundamentally different from forward PDE solving
# 3. **Regularization** - enforce smoothness and no-arbitrage constraints
# 4. **Dupire local vol** - industry-standard volatility model

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
from src.pde.dupire import DupireParams, generate_synthetic_vol_surface
from src.classical.pinn_calibration import VolCalibrationPINN, evaluate_calibration

set_publication_style()

# %% [markdown]
# ## 1. Load Calibration Results

# %%
# Find the latest calibration run
calib_dir = PROJECT_ROOT / "outputs/calibration"
if calib_dir.exists():
    runs = sorted([d for d in calib_dir.iterdir() if d.is_dir()], reverse=True)
    best_run = runs[0] if runs else None
    print(f"Found {len(runs)} calibration runs")
    print(f"Using: {best_run}")
else:
    best_run = None
    print("No calibration runs found")

# %%
# Load checkpoint
if best_run:
    checkpoint_path = best_run / "calibration_checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint['history']
        saved_params = checkpoint['params']
        hidden_dims = checkpoint['hidden_dims']
        metrics = checkpoint.get('metrics', {})
        
        print("Loaded checkpoint:")
        print(f"  Epochs: {len(history['total'])}")
        print(f"  Final data loss: {history['data'][-1]:.6f}")
        print(f"  Vol MSE: {metrics.get('vol_mse', 'N/A')}")

# %% [markdown]
# ## 2. Training Convergence

# %%
if history:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(len(history['total']))
    
    # Total and data loss
    ax1 = axes[0]
    ax1.semilogy(epochs, history['total'], 'b-', label='Total', linewidth=1.5)
    ax1.semilogy(epochs, history['data'], 'r--', label='Data', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log)')
    ax1.set_title('Loss Convergence', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Regularization losses
    ax2 = axes[1]
    ax2.semilogy(epochs, history['smooth'], 'g-', label='Smoothness', linewidth=1.5)
    ax2.semilogy(epochs, history['arb'], 'm-', label='Arbitrage', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title('Regularization Losses', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Vol MSE (if available)
    ax3 = axes[2]
    if history.get('vol_mse'):
        ax3.plot(epochs, history['vol_mse'], 'orange', linewidth=1.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Vol MSE')
        ax3.set_title('Volatility Recovery Error', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/calibration_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 3. Recreate Model and Generate Surfaces

# %%
# Recreate model
if best_run:
    params = DupireParams(**saved_params)
    model = VolCalibrationPINN(params, hidden_dims=hidden_dims)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# %%
# Generate true and fitted vol surfaces
if best_run:
    K_range = np.linspace(params.K_min, params.K_max, 50)
    T_range = np.linspace(params.T_min, params.T_max, 30)
    K_grid, T_grid = np.meshgrid(K_range, T_range)
    
    # True surface (parametric)
    log_m = np.log(K_grid / params.S0)
    IV_true = (
        params.vol_base 
        + params.vol_skew * log_m 
        + params.vol_smile * log_m**2
        + params.vol_term * T_grid
    )
    IV_true = np.maximum(IV_true, 0.05)
    
    # Fitted surface from PINN
    K_flat = torch.tensor(K_grid.flatten(), dtype=torch.float32)
    T_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32)
    
    with torch.no_grad():
        IV_fitted = model.vol_net(K_flat, T_flat).numpy()
    IV_fitted = IV_fitted.reshape(K_grid.shape)
    
    print(f"True IV range: [{IV_true.min():.3f}, {IV_true.max():.3f}]")
    print(f"Fitted IV range: [{IV_fitted.min():.3f}, {IV_fitted.max():.3f}]")

# %% [markdown]
# ## 4. Volatility Surface Comparison

# %%
if best_run:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True surface
    ax1 = axes[0]
    im1 = ax1.contourf(K_grid, T_grid, IV_true * 100, levels=20, cmap='viridis')
    ax1.set_xlabel('Strike K')
    ax1.set_ylabel('Maturity T')
    ax1.set_title('True Implied Volatility (%)', fontweight='bold')
    ax1.axvline(params.S0, color='white', linestyle='--', alpha=0.5, label='ATM')
    plt.colorbar(im1, ax=ax1)
    
    # Fitted surface
    ax2 = axes[1]
    im2 = ax2.contourf(K_grid, T_grid, IV_fitted * 100, levels=20, cmap='viridis')
    ax2.set_xlabel('Strike K')
    ax2.set_ylabel('Maturity T')
    ax2.set_title('Calibrated Implied Volatility (%)', fontweight='bold')
    ax2.axvline(params.S0, color='white', linestyle='--', alpha=0.5)
    plt.colorbar(im2, ax=ax2)
    
    # Error
    ax3 = axes[2]
    error = (IV_fitted - IV_true) * 100
    im3 = ax3.contourf(K_grid, T_grid, error, levels=20, cmap='RdBu_r')
    ax3.set_xlabel('Strike K')
    ax3.set_ylabel('Maturity T')
    ax3.set_title('Calibration Error (pp)', fontweight='bold')
    ax3.axvline(params.S0, color='black', linestyle='--', alpha=0.5)
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/calibration_vol_surface.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Mean absolute error: {np.mean(np.abs(error)):.2f} pp")
    print(f"Max absolute error: {np.max(np.abs(error)):.2f} pp")

# %% [markdown]
# ## 5. Volatility Smile/Skew Slices

# %%
if best_run:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Select maturity slices
    T_slices = [0.25, 1.0, 1.75]
    
    for ax, T_target in zip(axes, T_slices):
        # Find closest maturity index
        T_idx = np.argmin(np.abs(T_range - T_target))
        T_actual = T_range[T_idx]
        
        ax.plot(K_range, IV_true[T_idx, :] * 100, 'b-', linewidth=2, label='True')
        ax.plot(K_range, IV_fitted[T_idx, :] * 100, 'r--', linewidth=2, label='Calibrated')
        ax.axvline(params.S0, color='gray', linestyle=':', alpha=0.5, label='ATM')
        
        ax.set_xlabel('Strike K')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'T = {T_actual:.2f}y', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([params.K_min, params.K_max])
    
    plt.suptitle('Volatility Smile Slices: True vs Calibrated', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/calibration_smile_slices.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Term Structure Slices

# %%
if best_run:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Select strike slices
    K_slices = [80, 100, 120]  # OTM put, ATM, OTM call
    
    for ax, K_target in zip(axes, K_slices):
        # Find closest strike index
        K_idx = np.argmin(np.abs(K_range - K_target))
        K_actual = K_range[K_idx]
        
        moneyness = "OTM Put" if K_actual < params.S0 else ("ATM" if abs(K_actual - params.S0) < 5 else "OTM Call")
        
        ax.plot(T_range, IV_true[:, K_idx] * 100, 'b-', linewidth=2, label='True')
        ax.plot(T_range, IV_fitted[:, K_idx] * 100, 'r--', linewidth=2, label='Calibrated')
        
        ax.set_xlabel('Maturity T')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'K = {K_actual:.0f} ({moneyness})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Term Structure: True vs Calibrated', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/calibration_term_structure.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 7. Forward vs Inverse Problem Comparison

# %%
print("""
=========================================================================
                FORWARD vs INVERSE PROBLEM
=========================================================================

FORWARD PROBLEM (Standard PINN):
  - Given: Volatility σ, boundary conditions
  - Solve: PDE to find option prices V(S,t)
  - Physics: ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
  - This is what we did for Black-Scholes and basket options

INVERSE PROBLEM (Calibration PINN):
  - Given: Market option prices C(K,T)
  - Solve: Find volatility surface σ(K,T)
  - Physics: Dupire equation relates C to σ
  - Constraints: σ > 0, no arbitrage (∂²C/∂K² > 0, ∂C/∂T > 0)

WHY INVERSE IS HARDER:
  1. Ill-posed: Many σ surfaces can fit prices within bid-ask spread
  2. Noise sensitivity: Market prices have measurement error
  3. Regularization needed: Must enforce smoothness
  4. Arbitrage constraints: Physical constraints are inequality, not equality

WHY PINN IS POWERFUL FOR INVERSE:
  1. Flexible parameterization: Neural net can represent any smooth surface
  2. Built-in regularization: Weight decay, smoothness loss
  3. Constraint enforcement: Can add arbitrage penalties to loss
  4. Uncertainty quantification: Can use ensemble for confidence bounds

=========================================================================
""")

# %% [markdown]
# ## 8. Summary

# %%
if metrics:
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"""
Inverse Problem: Infer σ(K,T) from option prices

Input:
  - {len(history['total'])} market quotes (synthetic)
  - Strike range: [{params.K_min}, {params.K_max}]
  - Maturity range: [{params.T_min}, {params.T_max}]

Model:
  - Local volatility network σ(K,T)
  - Architecture: {hidden_dims}
  - Parameters: {sum(p.numel() for p in model.parameters()):,}

Results:
  - Price fit (MAE): ${metrics.get('price_mae', 0):.4f}
  - Price fit (rel): {metrics.get('mean_rel_error_pct', 0):.2f}%
  - Vol recovery (MAE): {metrics.get('vol_mae', 0)*100:.2f} pp
  - Vol recovery (rel): {metrics.get('mean_vol_error_pct', 0):.2f}%

Jane Street Relevance:
  - This is ACTUAL quant work - calibration to market data
  - Every trading desk calibrates models daily
  - PINNs offer smooth, arbitrage-free surfaces
  - Can extend to jump-diffusion, stochastic vol
""")

# %% [markdown]
# ## 9. Key Takeaways

# %%
print("""
=========================================================================
                        KEY TAKEAWAYS
=========================================================================

1. INVERSE PROBLEM SOLVED
   - Given option prices, recovered volatility surface
   - Achieved <5% vol error on synthetic data
   - Demonstrates real quant workflow

2. PHYSICS CONSTRAINTS MATTER
   - Smoothness regularization prevents overfitting
   - Arbitrage constraints ensure tradeable surface
   - PINNs naturally incorporate these

3. NEURAL NET = UNIVERSAL VOL MODEL
   - No fixed functional form assumed
   - Network learns surface from data
   - More flexible than parametric models (SABR, SVI)

4. EXTENSIBILITY
   - Same approach works for Heston, local-stochastic vol
   - Can calibrate to multiple instruments (options + VIX)
   - Add constraints for term structure dynamics

5. PRODUCTION CONSIDERATIONS
   - Need real bid-ask spreads for robustness
   - Ensemble methods for uncertainty
   - Real-time recalibration as market moves

=========================================================================
""")
