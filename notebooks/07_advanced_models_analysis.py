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
# # Advanced Options Models: Merton, Heston, and American PINNs
#
# This notebook analyzes three advanced PINN models:
# 1. **Merton Jump-Diffusion**: Asset prices with discontinuous jumps
# 2. **Heston Stochastic Volatility**: Time-varying variance with mean reversion
# 3. **American Options**: Early exercise with free boundary problem
#
# Each model extends Black-Scholes to capture real market dynamics.

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

# Import model components
from src.pde.merton import MertonParams, merton_analytical_call, MertonPINN
from src.pde.heston import HestonParams, heston_call_price, HestonPINN
from src.pde.american import AmericanParams, american_put_binomial, AmericanPINN, find_early_exercise_boundary

set_publication_style()

# %% [markdown]
# ## 1. Load Trained Checkpoints
#
# Find the latest training runs for each model. Handle cases where training 
# is still in progress or hasn't been run yet.

# %%
def find_latest_run(model_dir: Path) -> Path | None:
    """Find the latest timestamped run directory."""
    if not model_dir.exists():
        return None
    runs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    return runs[0] if runs else None


def load_checkpoint(checkpoint_path: Path) -> dict | None:
    """Load a PyTorch checkpoint safely."""
    if checkpoint_path.exists():
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return None


# Find latest runs
merton_run = find_latest_run(PROJECT_ROOT / "outputs/merton")
heston_run = find_latest_run(PROJECT_ROOT / "outputs/heston")
american_run = find_latest_run(PROJECT_ROOT / "outputs/american")

print("Model Training Status:")
print(f"  Merton:   {'✓ ' + str(merton_run.name) if merton_run else '✗ Not trained'}")
print(f"  Heston:   {'✓ ' + str(heston_run.name) if heston_run else '✗ Not trained'}")
print(f"  American: {'✓ ' + str(american_run.name) if american_run else '✗ Not trained'}")

# %% [markdown]
# ---
# # Part 1: Merton Jump-Diffusion Model
# ---
#
# The Merton model adds Poisson-distributed jumps to geometric Brownian motion:
# $$dS/S = (r - \lambda\kappa) dt + \sigma dW + (J-1) dN$$
#
# Key parameters:
# - λ (lambda): Jump intensity per year
# - μ_J: Mean of log-jump size
# - σ_J: Volatility of log-jump size

# %%
# Load Merton checkpoint
merton_checkpoint = None
merton_model = None

if merton_run:
    checkpoint_path = merton_run / "merton_checkpoint.pt"
    merton_checkpoint = load_checkpoint(checkpoint_path)
    
    if merton_checkpoint:
        merton_params = MertonParams(**merton_checkpoint['params'])
        merton_model = MertonPINN(
            params=merton_params,
            hidden_dims=merton_checkpoint.get('hidden_dims', [64, 64, 64, 64]),
        )
        merton_model.load_state_dict(merton_checkpoint['model_state_dict'])
        merton_model.eval()
        
        print("Merton Model Loaded:")
        print(f"  Parameters: {sum(p.numel() for p in merton_model.parameters()):,}")
        print(f"  Jump intensity (λ): {merton_params.lam}")
        print(f"  Jump mean (μ_J): {merton_params.mu_J}")
        print(f"  Jump vol (σ_J): {merton_params.sigma_J}")
        print(f"  κ = E[J-1]: {merton_params.kappa:.4f}")
    else:
        print("No Merton checkpoint found")
else:
    print("Merton model not trained yet")

# %% [markdown]
# ### 1.1 PINN vs Analytical Prices

# %%
if merton_model is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    S_range = np.linspace(50, 200, 100)
    tau = merton_params.T  # Time to maturity
    
    # Analytical prices (Merton's formula)
    V_analytical = merton_analytical_call(S_range, merton_params, tau)
    
    # PINN prices
    S_tensor = torch.tensor(S_range, dtype=torch.float32)
    t_tensor = torch.zeros_like(S_tensor)  # t=0 means tau=T
    
    with torch.no_grad():
        V_pinn = merton_model(S_tensor, t_tensor).numpy()
    
    # Plot 1: Price comparison
    ax1 = axes[0]
    ax1.plot(S_range, V_analytical, 'b-', linewidth=2, label='Analytical')
    ax1.plot(S_range, V_pinn, 'r--', linewidth=2, label='PINN')
    ax1.axvline(merton_params.K, color='gray', linestyle=':', alpha=0.5, label=f'K={merton_params.K}')
    ax1.set_xlabel('Spot Price S')
    ax1.set_ylabel('Call Option Price')
    ax1.set_title('Merton: PINN vs Analytical', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2 = axes[1]
    error = V_pinn - V_analytical
    ax2.plot(S_range, error, 'g-', linewidth=2)
    ax2.fill_between(S_range, 0, error, alpha=0.3, color='green')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Spot Price S')
    ax2.set_ylabel('Error (PINN - Analytical)')
    ax2.set_title(f'Pricing Error (MAE: ${np.mean(np.abs(error)):.4f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Relative error
    ax3 = axes[2]
    rel_error = np.abs(error) / np.maximum(V_analytical, 0.01) * 100
    ax3.semilogy(S_range, rel_error + 0.001, 'm-', linewidth=2)
    ax3.set_xlabel('Spot Price S')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title(f'Relative Error (Mean: {np.mean(rel_error):.2f}%)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/merton_pinn_vs_analytical.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Merton Pricing Accuracy:")
    print(f"  MAE:  ${np.mean(np.abs(error)):.4f}")
    print(f"  RMSE: ${np.sqrt(np.mean(error**2)):.4f}")
    print(f"  Max Error: ${np.max(np.abs(error)):.4f}")

# %% [markdown]
# ### 1.2 Volatility Smile from Jumps
#
# Jumps create implied volatility smiles/skews that Black-Scholes cannot capture.
# The negative jump mean (μ_J < 0) creates a skew (higher IV for low strikes).

# %%
if merton_model is not None:
    from scipy.stats import norm
    from scipy.optimize import brentq
    
    def black_scholes_call(S, K, T, r, sigma):
        """Standard Black-Scholes call price."""
        if T < 1e-10:
            return max(S - K, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    def implied_vol(price, S, K, T, r):
        """Invert Black-Scholes to get implied volatility."""
        if price < 1e-10:
            return np.nan
        try:
            return brentq(lambda sig: black_scholes_call(S, K, T, r, sig) - price, 0.01, 3.0)
        except ValueError:
            return np.nan
    
    # Compute implied vols for different strikes
    S0 = 100.0
    K_range = np.linspace(70, 140, 30)
    tau = merton_params.T
    
    iv_merton = []
    iv_bs = []
    
    for K in K_range:
        # Merton price and implied vol
        params_K = MertonParams(
            r=merton_params.r, sigma=merton_params.sigma, K=K, T=tau,
            lam=merton_params.lam, mu_J=merton_params.mu_J, sigma_J=merton_params.sigma_J
        )
        price_merton = merton_analytical_call(np.array([S0]), params_K, tau)[0]
        iv = implied_vol(price_merton, S0, K, tau, merton_params.r)
        iv_merton.append(iv)
        
        # BS reference (flat vol)
        iv_bs.append(merton_params.sigma)
    
    iv_merton = np.array(iv_merton)
    iv_bs = np.array(iv_bs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    log_moneyness = np.log(K_range / S0)
    
    ax.plot(log_moneyness, iv_merton * 100, 'b-', linewidth=2.5, label='Merton (with jumps)', marker='o', markersize=4)
    ax.axhline(merton_params.sigma * 100, color='r', linestyle='--', linewidth=2, label='Black-Scholes (flat)')
    
    ax.set_xlabel('Log-Moneyness ln(K/S)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title('Volatility Smile from Merton Jumps', fontsize=14, fontweight='bold')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='ATM')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/merton_vol_smile.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Volatility Smile Effect:")
    print(f"  ATM IV: {iv_merton[len(iv_merton)//2]*100:.2f}%")
    print(f"  OTM Put IV (K=80): {iv_merton[np.argmin(np.abs(K_range-80))]*100:.2f}%")
    print(f"  OTM Call IV (K=120): {iv_merton[np.argmin(np.abs(K_range-120))]*100:.2f}%")
    print(f"  Smile skew: {(iv_merton[0] - iv_merton[-1])*100:.2f}pp")

# %% [markdown]
# ### 1.3 Jump Parameter Sensitivity

# %%
if merton_model is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    S_range = np.linspace(50, 200, 100)
    S0_ref = 100.0
    
    # Sensitivity to λ (jump intensity)
    ax1 = axes[0]
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        params = MertonParams(
            r=0.05, sigma=0.20, K=100, T=1.0,
            lam=lam, mu_J=-0.10, sigma_J=0.15
        )
        prices = merton_analytical_call(S_range, params, params.T)
        label = f'λ={lam}' + (' (BS)' if lam == 0 else '')
        ax1.plot(S_range, prices, linewidth=2, label=label)
    
    ax1.set_xlabel('Spot Price S')
    ax1.set_ylabel('Call Price')
    ax1.set_title('Effect of Jump Intensity (λ)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sensitivity to μ_J (mean jump)
    ax2 = axes[1]
    for mu_J in [-0.30, -0.15, 0.0, 0.15]:
        params = MertonParams(
            r=0.05, sigma=0.20, K=100, T=1.0,
            lam=0.5, mu_J=mu_J, sigma_J=0.15
        )
        prices = merton_analytical_call(S_range, params, params.T)
        ax2.plot(S_range, prices, linewidth=2, label=f'μ_J={mu_J}')
    
    ax2.set_xlabel('Spot Price S')
    ax2.set_ylabel('Call Price')
    ax2.set_title('Effect of Jump Mean (μ_J)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sensitivity to σ_J (jump volatility)
    ax3 = axes[2]
    for sigma_J in [0.05, 0.15, 0.30, 0.50]:
        params = MertonParams(
            r=0.05, sigma=0.20, K=100, T=1.0,
            lam=0.5, mu_J=-0.10, sigma_J=sigma_J
        )
        prices = merton_analytical_call(S_range, params, params.T)
        ax3.plot(S_range, prices, linewidth=2, label=f'σ_J={sigma_J}')
    
    ax3.set_xlabel('Spot Price S')
    ax3.set_ylabel('Call Price')
    ax3.set_title('Effect of Jump Volatility (σ_J)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/merton_param_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ---
# # Part 2: Heston Stochastic Volatility Model
# ---
#
# The Heston model allows volatility to vary randomly:
# $$dS = rS dt + \sqrt{v} S dW_S$$
# $$dv = \kappa(\theta - v) dt + \xi \sqrt{v} dW_v$$
# $$dW_S \cdot dW_v = \rho \, dt$$
#
# Key parameters:
# - κ (kappa): Mean reversion speed
# - θ (theta): Long-run variance
# - ξ (xi): Volatility of variance (vol-of-vol)
# - ρ (rho): Correlation (typically negative for equities)

# %%
# Load Heston checkpoint (if available)
heston_checkpoint = None
heston_model = None

if heston_run:
    checkpoint_path = heston_run / "heston_checkpoint.pt"
    heston_checkpoint = load_checkpoint(checkpoint_path)
    
    if heston_checkpoint:
        heston_params = HestonParams(**heston_checkpoint['params'])
        heston_model = HestonPINN(
            params=heston_params,
            hidden_dims=heston_checkpoint.get('hidden_dims', [64, 64, 64, 64]),
        )
        heston_model.load_state_dict(heston_checkpoint['model_state_dict'])
        heston_model.eval()
        
        print("Heston Model Loaded:")
        print(f"  Feller condition satisfied: {heston_params.feller_satisfied}")
    else:
        print("No Heston checkpoint found")
else:
    print("Heston model not trained yet - using analytical formulas for demo")
    # Create default params for demonstration
    heston_params = HestonParams()

# %% [markdown]
# ### 2.1 Heston Price Surface (S, v)
#
# The Heston model prices options in 3D: (Spot, Variance, Time).
# Here we visualize a slice at t=0 (current time).

# %%
# Use analytical Heston prices for visualization (works even without trained PINN)
fig = plt.figure(figsize=(14, 6))

S_range = np.linspace(60, 160, 30)
v_range = np.linspace(0.01, 0.16, 25)  # Variance (not vol!)
S_grid, v_grid = np.meshgrid(S_range, v_range)

# Compute analytical Heston prices
prices = np.zeros_like(S_grid)
for i in range(len(v_range)):
    for j in range(len(S_range)):
        params_ij = HestonParams(
            r=heston_params.r, K=heston_params.K, T=heston_params.T,
            kappa=heston_params.kappa, theta=heston_params.theta,
            xi=heston_params.xi, rho=heston_params.rho, v0=v_range[i]
        )
        try:
            prices[i, j] = heston_call_price(S_range[j], params_ij, heston_params.T)
        except:
            prices[i, j] = np.nan

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(S_grid, np.sqrt(v_grid)*100, prices, cmap='viridis', alpha=0.9)
ax1.set_xlabel('Spot Price S')
ax1.set_ylabel('Volatility σ (%)')
ax1.set_zlabel('Call Price')
ax1.set_title('Heston Price Surface', fontweight='bold')
ax1.view_init(elev=25, azim=-60)

# 2D contour
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(S_grid, np.sqrt(v_grid)*100, prices, levels=20, cmap='viridis')
ax2.set_xlabel('Spot Price S')
ax2.set_ylabel('Volatility σ (%)')
ax2.set_title('Heston Price Contours', fontweight='bold')
ax2.axhline(np.sqrt(heston_params.theta)*100, color='white', linestyle='--', alpha=0.7, label='θ^0.5')
ax2.axvline(heston_params.K, color='white', linestyle=':', alpha=0.7, label='K')
plt.colorbar(contour, ax=ax2, label='Call Price')

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs/analysis/heston_price_surface.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.2 Implied Volatility Smile from Heston

# %%
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes call price."""
    if T < 1e-10:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol_from_heston(S0, K, T, heston_params):
    """Get implied vol from Heston price."""
    params = HestonParams(
        r=heston_params.r, K=K, T=T,
        kappa=heston_params.kappa, theta=heston_params.theta,
        xi=heston_params.xi, rho=heston_params.rho, v0=heston_params.v0
    )
    try:
        price = heston_call_price(S0, params, T)
        if price < 1e-10:
            return np.nan
        iv = brentq(lambda sig: black_scholes_call(S0, K, T, heston_params.r, sig) - price, 0.01, 3.0)
        return iv
    except:
        return np.nan

# Compute implied vols
S0 = 100.0
K_range = np.linspace(70, 140, 25)
T_vals = [0.25, 0.5, 1.0, 2.0]

fig, ax = plt.subplots(figsize=(10, 6))

for T in T_vals:
    ivs = [implied_vol_from_heston(S0, K, T, heston_params) for K in K_range]
    ivs = np.array(ivs) * 100
    log_m = np.log(K_range / S0)
    ax.plot(log_m, ivs, linewidth=2, marker='o', markersize=3, label=f'T={T}y')

ax.axhline(np.sqrt(heston_params.theta) * 100, color='gray', linestyle='--', 
           alpha=0.7, label=f'Long-run vol (θ^0.5={np.sqrt(heston_params.theta)*100:.1f}%)')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Log-Moneyness ln(K/S)', fontsize=12)
ax.set_ylabel('Implied Volatility (%)', fontsize=12)
ax.set_title(f'Heston Volatility Smile (ρ={heston_params.rho})', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs/analysis/heston_vol_smile.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Heston Model Parameters:")
print(f"  κ (mean reversion): {heston_params.kappa}")
print(f"  θ (long-run var):   {heston_params.theta} (σ={np.sqrt(heston_params.theta)*100:.1f}%)")
print(f"  ξ (vol-of-vol):     {heston_params.xi}")
print(f"  ρ (correlation):    {heston_params.rho}")
print(f"  Feller satisfied:   {heston_params.feller_satisfied}")

# %% [markdown]
# ### 2.3 Heston vs Black-Scholes at Different Variance Levels

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

S_range = np.linspace(50, 200, 100)
v_levels = [0.01, 0.04, 0.16]  # Low, medium, high variance
vol_levels = np.sqrt(np.array(v_levels)) * 100

for ax, v0, vol_label in zip(axes, v_levels, ['Low (10%)', 'Medium (20%)', 'High (40%)']):
    # Heston prices
    heston_prices = []
    for S in S_range:
        params = HestonParams(
            r=heston_params.r, K=100, T=1.0,
            kappa=heston_params.kappa, theta=heston_params.theta,
            xi=heston_params.xi, rho=heston_params.rho, v0=v0
        )
        try:
            heston_prices.append(heston_call_price(S, params, 1.0))
        except:
            heston_prices.append(np.nan)
    
    heston_prices = np.array(heston_prices)
    
    # Black-Scholes prices (using same initial vol)
    sigma = np.sqrt(v0)
    bs_prices = np.array([black_scholes_call(S, 100, 1.0, heston_params.r, sigma) for S in S_range])
    
    ax.plot(S_range, heston_prices, 'b-', linewidth=2, label='Heston')
    ax.plot(S_range, bs_prices, 'r--', linewidth=2, label='Black-Scholes')
    ax.axvline(100, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Spot Price S')
    ax.set_ylabel('Call Price')
    ax.set_title(f'Variance Level: {vol_label}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Heston vs Black-Scholes at Different Volatility Levels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "outputs/analysis/heston_vs_bs.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# # Part 3: American Options with Early Exercise
# ---
#
# American options can be exercised anytime before expiry. The key challenges:
# 1. **Free boundary problem**: Exercise region changes with time
# 2. **Early exercise premium**: American put > European put
# 3. **Complementarity condition**: Either hold or exercise is optimal

# %%
# Load American checkpoint
american_checkpoint = None
american_model = None

if american_run:
    checkpoint_path = american_run / "american_checkpoint.pt"
    american_checkpoint = load_checkpoint(checkpoint_path)
    
    if american_checkpoint:
        american_params = AmericanParams(**american_checkpoint['params'])
        american_model = AmericanPINN(
            params=american_params,
            hidden_dims=american_checkpoint.get('hidden_dims', [64, 64, 64, 64]),
        )
        american_model.load_state_dict(american_checkpoint['model_state_dict'])
        american_model.eval()
        
        print("American Model Loaded:")
        print(f"  Option type: {american_params.option_type}")
        print(f"  Strike: ${american_params.K}")
        print(f"  Maturity: {american_params.T} years")
        print(f"  Parameters: {sum(p.numel() for p in american_model.parameters()):,}")
    else:
        print("No American checkpoint found")
else:
    print("American model not trained yet")
    american_params = AmericanParams()

# %% [markdown]
# ### 3.1 PINN vs Binomial Tree Reference

# %%
if american_model is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    S_range = np.linspace(50, 200, 100)
    
    # Binomial tree prices (reference)
    V_binomial = np.array([american_put_binomial(S, american_params) for S in S_range])
    
    # PINN prices
    S_tensor = torch.tensor(S_range, dtype=torch.float32)
    t_tensor = torch.zeros_like(S_tensor)
    
    with torch.no_grad():
        V_pinn = american_model(S_tensor, t_tensor).numpy()
    
    # Plot 1: Price comparison
    ax1 = axes[0]
    ax1.plot(S_range, V_binomial, 'b-', linewidth=2, label='Binomial Tree')
    ax1.plot(S_range, V_pinn, 'r--', linewidth=2, label='PINN')
    payoff = np.maximum(american_params.K - S_range, 0)
    ax1.plot(S_range, payoff, 'k:', linewidth=1.5, alpha=0.7, label='Intrinsic Value')
    ax1.axvline(american_params.K, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Spot Price S')
    ax1.set_ylabel('Put Option Price')
    ax1.set_title('American Put: PINN vs Binomial', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2 = axes[1]
    error = V_pinn - V_binomial
    ax2.plot(S_range, error, 'g-', linewidth=2)
    ax2.fill_between(S_range, 0, error, alpha=0.3, color='green')
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Spot Price S')
    ax2.set_ylabel('Error (PINN - Binomial)')
    ax2.set_title(f'Pricing Error (MAE: ${np.mean(np.abs(error)):.4f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time value
    ax3 = axes[2]
    time_value_pinn = V_pinn - payoff
    time_value_binomial = V_binomial - payoff
    ax3.plot(S_range, time_value_binomial, 'b-', linewidth=2, label='Binomial')
    ax3.plot(S_range, time_value_pinn, 'r--', linewidth=2, label='PINN')
    ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Spot Price S')
    ax3.set_ylabel('Time Value (V - Intrinsic)')
    ax3.set_title('Time Value Component', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/american_pinn_vs_binomial.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"American Put Pricing Accuracy:")
    print(f"  MAE:  ${np.mean(np.abs(error)):.4f}")
    print(f"  RMSE: ${np.sqrt(np.mean(error**2)):.4f}")

# %% [markdown]
# ### 3.2 Early Exercise Boundary S*(t)
#
# The early exercise boundary separates the continuation region (hold) from the
# exercise region (exercise immediately). For American puts:
# - If S < S*(t): Exercise immediately
# - If S > S*(t): Continue holding

# %%
if american_model is not None:
    t_values = np.linspace(0, american_params.T * 0.99, 50)
    
    # Find boundary using PINN
    boundaries = find_early_exercise_boundary(
        american_model, t_values, 
        S_range=(50, 150), n_search=200
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(t_values, boundaries, 0, alpha=0.3, color='red', label='Exercise Region')
    ax.fill_between(t_values, boundaries, 200, alpha=0.3, color='green', label='Continuation Region')
    ax.plot(t_values, boundaries, 'b-', linewidth=2.5, label='S*(t) Boundary')
    ax.axhline(american_params.K, color='gray', linestyle='--', alpha=0.7, label=f'Strike K={american_params.K}')
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Spot Price S', fontsize=12)
    ax.set_title('Early Exercise Boundary for American Put', fontsize=14, fontweight='bold')
    ax.set_xlim([0, american_params.T])
    ax.set_ylim([50, 150])
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Exercise\nimmediately', xy=(0.1, boundaries[5] - 15), fontsize=10, color='red')
    ax.annotate('Continue\nholding', xy=(0.1, boundaries[5] + 20), fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/american_exercise_boundary.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Early Exercise Boundary:")
    print(f"  S*(0) = ${boundaries[0]:.2f} (at inception)")
    print(f"  S*(T/2) = ${boundaries[len(boundaries)//2]:.2f} (mid-life)")
    print(f"  Boundary approaches K={american_params.K} as t→T")

# %% [markdown]
# ### 3.3 American Put Premium over European Put
#
# The early exercise feature has value. American put is always worth at least 
# as much as European put.

# %%
def european_put_bs(S, K, T, r, sigma):
    """Black-Scholes European put price."""
    if T < 1e-10:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

if american_model is not None or True:  # Always show this analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    S_range = np.linspace(50, 150, 100)
    params = american_params if american_model else AmericanParams()
    
    # American put prices (binomial)
    V_american = np.array([american_put_binomial(S, params) for S in S_range])
    
    # European put prices
    V_european = np.array([european_put_bs(S, params.K, params.T, params.r, params.sigma) for S in S_range])
    
    # Early exercise premium
    premium = V_american - V_european
    
    ax1 = axes[0]
    ax1.plot(S_range, V_american, 'b-', linewidth=2, label='American Put')
    ax1.plot(S_range, V_european, 'r--', linewidth=2, label='European Put')
    intrinsic = np.maximum(params.K - S_range, 0)
    ax1.plot(S_range, intrinsic, 'k:', linewidth=1.5, alpha=0.7, label='Intrinsic')
    ax1.axvline(params.K, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Spot Price S')
    ax1.set_ylabel('Put Option Price')
    ax1.set_title('American vs European Put', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(S_range, premium, 'purple', linewidth=2.5)
    ax2.fill_between(S_range, 0, premium, alpha=0.3, color='purple')
    ax2.set_xlabel('Spot Price S')
    ax2.set_ylabel('Premium ($)')
    ax2.set_title(f'Early Exercise Premium (max: ${premium.max():.2f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs/analysis/american_vs_european.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Early Exercise Premium Analysis:")
    print(f"  Maximum premium: ${premium.max():.2f}")
    print(f"  Premium at S=80: ${premium[np.argmin(np.abs(S_range-80))]:.2f}")
    print(f"  Premium at S=100: ${premium[np.argmin(np.abs(S_range-100))]:.2f}")
    print(f"  Premium at S=120: ${premium[np.argmin(np.abs(S_range-120))]:.2f}")

# %% [markdown]
# ---
# # Part 4: Model Comparison
# ---
#
# Compare all three advanced models in terms of:
# 1. Accuracy metrics (MSE, MAE)
# 2. Training complexity
# 3. When to use each model

# %%
print("=" * 75)
print("MODEL COMPARISON SUMMARY")
print("=" * 75)

# Collect metrics from checkpoints
model_metrics = {}

if merton_checkpoint:
    history = merton_checkpoint.get('history', {})
    if isinstance(history, list):
        final_loss = history[-1]['total'] if history else float('nan')
    else:
        final_loss = history.get('total', [float('nan')])[-1] if history.get('total') else float('nan')
    
    model_metrics['Merton'] = {
        'Epochs': len(history) if isinstance(history, list) else len(history.get('total', [])),
        'Final Loss': final_loss,
        'Parameters': sum(p.numel() for p in merton_model.parameters()) if merton_model else 0,
        'Input Dim': '2D (S, t)',
        'PDE Type': 'PIDE (integro-differential)',
    }

if heston_checkpoint:
    history = heston_checkpoint.get('history', {})
    if isinstance(history, list):
        final_loss = history[-1]['total'] if history else float('nan')
    else:
        final_loss = history.get('total', [float('nan')])[-1] if history.get('total') else float('nan')
    
    model_metrics['Heston'] = {
        'Epochs': len(history) if isinstance(history, list) else len(history.get('total', [])),
        'Final Loss': final_loss,
        'Parameters': sum(p.numel() for p in heston_model.parameters()) if heston_model else 0,
        'Input Dim': '3D (S, v, t)',
        'PDE Type': 'PDE (2nd order, 2 spatial)',
    }

if american_checkpoint:
    history = american_checkpoint.get('history', {})
    if isinstance(history, list):
        final_loss = history[-1]['total'] if history else float('nan')
    else:
        final_loss = history.get('total', [float('nan')])[-1] if history.get('total') else float('nan')
    
    model_metrics['American'] = {
        'Epochs': len(history) if isinstance(history, list) else len(history.get('total', [])),
        'Final Loss': final_loss,
        'Parameters': sum(p.numel() for p in american_model.parameters()) if american_model else 0,
        'Input Dim': '2D (S, t)',
        'PDE Type': 'Free boundary (obstacle)',
    }

# Print comparison table
if model_metrics:
    print(f"\n{'Model':<12} {'Epochs':<10} {'Final Loss':<12} {'Parameters':<12} {'Input':<15} {'PDE Type':<25}")
    print("-" * 86)
    for model, metrics in model_metrics.items():
        print(f"{model:<12} {metrics['Epochs']:<10} {metrics['Final Loss']:<12.4f} {metrics['Parameters']:<12,} {metrics['Input Dim']:<15} {metrics['PDE Type']:<25}")
else:
    print("\nNo trained models found. Run training scripts first.")

# %% [markdown]
# ### When to Use Each Model

# %%
print("""
=========================================================================
                    PRACTICAL GUIDANCE
=========================================================================

MERTON JUMP-DIFFUSION
  Use when:
    ✓ Pricing short-dated options (jumps matter more)
    ✓ Tail risk hedging (crash scenarios)
    ✓ Fitting observed volatility skew
    ✓ Credit-sensitive derivatives
  Characteristics:
    - Captures sudden price movements
    - Creates volatility smile/skew
    - Analytical solution available (semi-closed form)
  
HESTON STOCHASTIC VOLATILITY
  Use when:
    ✓ Pricing long-dated options (vol persistence matters)
    ✓ Variance trading (VIX products)
    ✓ Path-dependent payoffs (barriers, Asians)
    ✓ Building consistent vol surface
  Characteristics:
    - Mean-reverting volatility
    - Negative correlation → negative skew
    - 2D spatial problem (harder numerically)

AMERICAN OPTIONS
  Use when:
    ✓ Single-stock options (most listed options are American)
    ✓ High dividend yield assets
    ✓ Deep ITM puts (early exercise valuable)
    ✓ Computing early exercise boundaries
  Characteristics:
    - Free boundary problem
    - No closed-form solution
    - Premium over European increases with dividends/rates

=========================================================================
                    PINN ADVANTAGES
=========================================================================

1. MESH-FREE: No grid discretization → flexible domains
2. DIFFERENTIABLE: Greeks via autodiff → no finite difference noise
3. TRANSFER LEARNING: Pre-trained networks for similar problems
4. INVERSE PROBLEMS: Calibrate parameters from market prices
5. UNCERTAINTY: Ensemble methods for confidence bounds

=========================================================================
""")

# %% [markdown]
# ### Load Evaluation Metrics (if available)

# %%
# Load evaluation JSONs if they exist
eval_summary = {}

if american_run:
    eval_path = american_run / "evaluation.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_summary['American'] = json.load(f)

# Print evaluation summary
if eval_summary:
    print("\nEvaluation Metrics (vs Reference):")
    print("-" * 50)
    for model, metrics in eval_summary.items():
        print(f"\n{model}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

# %% [markdown]
# ## Summary

# %%
print("""
=======================================================================
                         KEY TAKEAWAYS
=======================================================================

1. MERTON ADDS JUMPS
   - PIDE instead of PDE (integral term for jumps)
   - Creates volatility skew from asymmetric jumps
   - Semi-analytical formula enables fast validation

2. HESTON ADDS STOCHASTIC VOL
   - 3D problem: (S, v, t)
   - Mean reversion → term structure of vol
   - Negative ρ → leverage effect → skew

3. AMERICAN ADDS EARLY EXERCISE
   - Free boundary problem (obstacle constraint)
   - Penalty method in PINN loss
   - Can extract exercise boundary S*(t)

4. PINN APPROACH SCALES
   - Same architecture handles all three models
   - Just change the PDE residual computation
   - Greeks available for all via autodiff

5. JANE STREET RELEVANCE
   - Real desks use these models daily
   - Merton for equity tail risk
   - Heston for variance products
   - American for listed options

=======================================================================
""")
