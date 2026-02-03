"""
Dupire Local Volatility Model.

The Dupire equation relates the local volatility function σ(K,T) to 
European call prices C(K,T):

    σ²(K,T) = 2 * [∂C/∂T + rK∂C/∂K] / [K² ∂²C/∂K²]

This is a FORWARD problem: given σ(K,T), price options.

The INVERSE problem (calibration): given market prices C(K,T), find σ(K,T).

PINNs excel at inverse problems because we can:
1. Parameterize σ(K,T) as a neural network
2. Fit observed option prices as data loss
3. Enforce PDE constraints via physics loss
4. Regularize the volatility surface for smoothness
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable
from scipy.stats import norm


@dataclass
class DupireParams:
    """Parameters for Dupire local volatility model."""
    r: float = 0.05              # Risk-free rate
    q: float = 0.0               # Dividend yield
    S0: float = 100.0            # Current spot price
    
    # Strike/maturity grid bounds
    K_min: float = 50.0
    K_max: float = 200.0
    T_min: float = 0.05          # Minimum maturity (avoid singularity)
    T_max: float = 2.0           # Maximum maturity
    
    # For synthetic data generation
    vol_base: float = 0.20       # Base implied volatility
    vol_skew: float = -0.10      # Skew (negative for equity)
    vol_smile: float = 0.05      # Smile curvature
    vol_term: float = -0.02      # Term structure slope


def generate_synthetic_vol_surface(
    params: DupireParams,
    n_strikes: int = 20,
    n_maturities: int = 10,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic implied volatility surface for testing.
    
    Uses a parametric SABR-like model:
        IV(K,T) = vol_base + vol_skew * log(K/S0) + vol_smile * log(K/S0)² + vol_term * T
    
    Args:
        params: Dupire parameters
        n_strikes: Number of strikes
        n_maturities: Number of maturities
        noise_std: Add Gaussian noise (for robustness testing)
        seed: Random seed
        
    Returns:
        Dictionary with strikes, maturities, implied_vols, call_prices
    """
    rng = np.random.default_rng(seed)
    
    # Create grid
    strikes = np.linspace(params.K_min, params.K_max, n_strikes)
    maturities = np.linspace(params.T_min, params.T_max, n_maturities)
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    K_flat = K_grid.flatten()
    T_flat = T_grid.flatten()
    
    # Log-moneyness
    log_m = np.log(K_flat / params.S0)
    
    # Parametric implied vol surface
    iv = (
        params.vol_base 
        + params.vol_skew * log_m 
        + params.vol_smile * log_m**2
        + params.vol_term * T_flat
    )
    
    # Ensure volatility is positive
    iv = np.maximum(iv, 0.05)
    
    # Add noise if requested
    if noise_std > 0:
        iv = iv + rng.normal(0, noise_std, iv.shape)
        iv = np.maximum(iv, 0.05)
    
    # Compute Black-Scholes call prices
    call_prices = black_scholes_call(
        S=params.S0,
        K=K_flat,
        T=T_flat,
        r=params.r,
        sigma=iv,
    )
    
    return {
        "strikes": K_flat,
        "maturities": T_flat,
        "implied_vols": iv,
        "call_prices": call_prices,
        "K_grid": K_grid,
        "T_grid": T_grid,
        "IV_grid": iv.reshape(K_grid.shape),
    }


def black_scholes_call(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes call price."""
    tau = np.maximum(T, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    call = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return call


def black_scholes_call_torch(
    S: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Black-Scholes call price in PyTorch (differentiable)."""
    tau = torch.clamp(T, min=1e-10)
    sqrt_tau = torch.sqrt(tau)
    
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    
    # Use PyTorch's normal CDF
    normal = torch.distributions.Normal(0, 1)
    N_d1 = normal.cdf(d1)
    N_d2 = normal.cdf(d2)
    
    call = S * N_d1 - K * torch.exp(-r * tau) * N_d2
    return call


def dupire_local_vol_analytical(
    C: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
) -> torch.Tensor:
    """
    Compute local volatility from call prices using Dupire formula.
    
    σ²_local(K,T) = 2 * [∂C/∂T + rK∂C/∂K] / [K² ∂²C/∂K²]
    
    Requires C to have gradients enabled for K and T.
    
    Args:
        C: Call prices (with grad)
        K: Strike prices (with grad)
        T: Maturities (with grad)
        r: Risk-free rate
        
    Returns:
        Local variance σ²(K,T)
    """
    # First derivatives
    dC_dT = torch.autograd.grad(
        C.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    dC_dK = torch.autograd.grad(
        C.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Second derivative
    d2C_dK2 = torch.autograd.grad(
        dC_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Dupire formula (local variance)
    numerator = 2 * (dC_dT + r * K * dC_dK)
    denominator = K**2 * d2C_dK2
    
    # Regularize to avoid division by zero
    local_var = numerator / (denominator + 1e-8)
    
    # Clamp to reasonable range [0.01, 4.0] for variance (vol 10% to 200%)
    local_var = torch.clamp(local_var, min=0.01, max=4.0)
    
    return local_var


class LocalVolNetwork(nn.Module):
    """
    Neural network that outputs local volatility σ(K, T).
    
    The network learns the volatility surface directly.
    Input: (K, T) normalized
    Output: σ(K, T) > 0
    """
    
    def __init__(
        self,
        hidden_dims: list[int] = [64, 64, 64],
        K_range: tuple[float, float] = (50, 200),
        T_range: tuple[float, float] = (0.05, 2.0),
    ):
        super().__init__()
        
        self.K_min, self.K_max = K_range
        self.T_min, self.T_max = T_range
        
        # Build MLP
        layers = []
        in_dim = 2  # (K, T)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # Ensure σ > 0
        
        self.network = nn.Sequential(*layers)
        
        # Initialize to output reasonable vol (~0.2)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize to output vol around 0.2."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        
        # Adjust last layer to output ~0.2
        last_linear = [m for m in self.network if isinstance(m, nn.Linear)][-1]
        last_linear.bias.data.fill_(-1.5)  # softplus(-1.5) ≈ 0.2
    
    def forward(self, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute local volatility.
        
        Args:
            K: Strike prices, shape (batch,)
            T: Maturities, shape (batch,)
            
        Returns:
            Local volatility σ(K,T), shape (batch,)
        """
        # Normalize inputs to [-1, 1]
        K_norm = 2 * (K - self.K_min) / (self.K_max - self.K_min) - 1
        T_norm = 2 * (T - self.T_min) / (self.T_max - self.T_min) - 1
        
        x = torch.stack([K_norm, T_norm], dim=-1)
        sigma = self.network(x).squeeze(-1)
        
        # Clamp to reasonable range
        sigma = torch.clamp(sigma, min=0.05, max=1.5)
        
        return sigma


class CallPriceNetwork(nn.Module):
    """
    Neural network that outputs call prices C(K, T).
    
    For the inverse problem, we learn both:
    - Call price surface C(K, T) 
    - Local vol σ(K, T) is derived via Dupire formula
    """
    
    def __init__(
        self,
        hidden_dims: list[int] = [64, 64, 64, 64],
        S0: float = 100.0,
        K_range: tuple[float, float] = (50, 200),
        T_range: tuple[float, float] = (0.05, 2.0),
    ):
        super().__init__()
        
        self.S0 = S0
        self.K_min, self.K_max = K_range
        self.T_min, self.T_max = T_range
        
        # Build MLP
        layers = []
        in_dim = 2  # (K, T)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize for stable training."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Compute call price.
        
        Args:
            K: Strike prices, shape (batch,)
            T: Maturities, shape (batch,)
            
        Returns:
            Call prices C(K,T), shape (batch,)
        """
        # Normalize inputs
        K_norm = 2 * (K - self.K_min) / (self.K_max - self.K_min) - 1
        T_norm = 2 * (T - self.T_min) / (self.T_max - self.T_min) - 1
        
        x = torch.stack([K_norm, T_norm], dim=-1)
        
        # Raw network output
        raw = self.network(x).squeeze(-1)
        
        # Enforce call price constraints:
        # 1. C ≥ max(S0 - K*exp(-rT), 0) (no-arbitrage lower bound)
        # 2. C ≤ S0 (upper bound for call)
        # Use sigmoid to interpolate within bounds
        
        # Simple approach: softplus for positivity, scaled by S0
        C = torch.nn.functional.softplus(raw) * self.S0 / 10
        
        return C


def compute_dupire_pde_residual(
    model: nn.Module,
    K: torch.Tensor,
    T: torch.Tensor,
    r: float,
) -> torch.Tensor:
    """
    Compute Dupire PDE residual for the inverse problem.
    
    The forward Dupire PDE is:
        ∂C/∂T = ½σ²(K,T)K² ∂²C/∂K² - rK ∂C/∂K
        
    Residual should be zero if the call price surface is arbitrage-free
    and consistent with local vol.
    
    Args:
        model: Network outputting call prices
        K: Strikes (requires grad)
        T: Maturities (requires grad)
        r: Risk-free rate
        
    Returns:
        PDE residual
    """
    # Ensure gradients
    K = K.requires_grad_(True)
    T = T.requires_grad_(True)
    
    # Call prices from network
    C = model(K, T)
    
    # Compute derivatives
    dC_dT = torch.autograd.grad(
        C.sum(), T, create_graph=True, retain_graph=True
    )[0]
    
    dC_dK = torch.autograd.grad(
        C.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    d2C_dK2 = torch.autograd.grad(
        dC_dK.sum(), K, create_graph=True, retain_graph=True
    )[0]
    
    # Local variance from Dupire formula
    numerator = 2 * (dC_dT + r * K * dC_dK)
    denominator = K**2 * d2C_dK2 + 1e-8
    local_var = numerator / denominator
    
    # PDE residual: ∂C/∂T - ½σ²K²∂²C/∂K² + rK∂C/∂K = 0
    residual = dC_dT - 0.5 * local_var * K**2 * d2C_dK2 + r * K * dC_dK
    
    return residual, local_var


def generate_calibration_data(
    params: DupireParams,
    n_points: int = 100,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Generate synthetic market data for calibration testing.
    
    Returns:
        Dictionary with K, T, C_market, IV_true tensors
    """
    surface = generate_synthetic_vol_surface(
        params,
        n_strikes=int(np.sqrt(n_points)),
        n_maturities=int(np.sqrt(n_points)),
        noise_std=0.0,
        seed=seed,
    )
    
    return {
        "K": torch.tensor(surface["strikes"], dtype=torch.float32),
        "T": torch.tensor(surface["maturities"], dtype=torch.float32),
        "C_market": torch.tensor(surface["call_prices"], dtype=torch.float32),
        "IV_true": torch.tensor(surface["implied_vols"], dtype=torch.float32),
    }
