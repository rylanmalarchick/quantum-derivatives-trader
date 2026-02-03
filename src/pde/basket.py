"""
Multi-asset basket option PDE.

Extends Black-Scholes to N correlated assets. The basket option has payoff:
    max(Σ wᵢ·Sᵢ(T) - K, 0)

The N-dimensional Black-Scholes PDE is:
    ∂V/∂t + Σᵢ rSᵢ∂V/∂Sᵢ + ½ΣᵢΣⱼ ρᵢⱼσᵢσⱼSᵢSⱼ∂²V/∂Sᵢ∂Sⱼ - rV = 0

This is a (N+1)-dimensional PDE that cannot be solved by finite differences
for N > 3 due to the curse of dimensionality.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import qmc


@dataclass
class BasketParams:
    """Parameters for multi-asset basket option."""
    n_assets: int = 5
    r: float = 0.05                          # Risk-free rate
    K: float = 100.0                         # Strike price
    T: float = 1.0                           # Time to maturity
    
    # Per-asset parameters (defaults for 5 assets)
    S0: np.ndarray = field(default_factory=lambda: np.array([100., 100., 100., 100., 100.]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.25, 0.18, 0.22, 0.20]))
    weights: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    
    # Correlation matrix (default: moderate positive correlation)
    correlation: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        self.S0 = np.atleast_1d(self.S0)
        self.sigma = np.atleast_1d(self.sigma)
        self.weights = np.atleast_1d(self.weights)
        
        # Validate dimensions
        assert len(self.S0) == self.n_assets
        assert len(self.sigma) == self.n_assets
        assert len(self.weights) == self.n_assets
        assert np.isclose(self.weights.sum(), 1.0), "Weights must sum to 1"
        
        # Default correlation: moderate positive (realistic for equity basket)
        if self.correlation is None:
            self.correlation = self._default_correlation()
        
        # Validate correlation matrix
        assert self.correlation.shape == (self.n_assets, self.n_assets)
        assert np.allclose(self.correlation, self.correlation.T), "Correlation must be symmetric"
        assert np.all(np.linalg.eigvals(self.correlation) > -1e-10), "Correlation must be PSD"
    
    def _default_correlation(self) -> np.ndarray:
        """Create realistic equity correlation matrix."""
        # Off-diagonal correlations between 0.3 and 0.7
        rho = np.eye(self.n_assets)
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                # Slightly different correlations for realism
                rho[i, j] = 0.4 + 0.05 * ((i + j) % 4)
                rho[j, i] = rho[i, j]
        return rho
    
    @property
    def covariance(self) -> np.ndarray:
        """Compute covariance matrix: Σᵢⱼ = ρᵢⱼσᵢσⱼ."""
        return self.correlation * np.outer(self.sigma, self.sigma)
    
    @property 
    def S_max(self) -> np.ndarray:
        """Upper bounds for each asset (for normalization)."""
        return self.S0 * 2.0  # 2x initial price as upper bound
    
    @property
    def S_min(self) -> np.ndarray:
        """Lower bounds for each asset."""
        return self.S0 * 0.2  # 20% of initial as lower bound


def basket_payoff(S: torch.Tensor, params: BasketParams) -> torch.Tensor:
    """
    Compute basket option payoff.
    
    Args:
        S: Asset prices, shape (batch, n_assets)
        params: Basket parameters
        
    Returns:
        Payoff values, shape (batch,)
    """
    weights = torch.tensor(params.weights, dtype=S.dtype, device=S.device)
    basket_value = (S * weights).sum(dim=-1)
    return torch.relu(basket_value - params.K)


def basket_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor, 
    t: torch.Tensor,
    params: BasketParams,
) -> torch.Tensor:
    """
    Compute the N-dimensional Black-Scholes PDE residual.
    
    The PDE is:
        ∂V/∂t + Σᵢ rSᵢ∂V/∂Sᵢ + ½ΣᵢΣⱼ Σᵢⱼ SᵢSⱼ ∂²V/∂Sᵢ∂Sⱼ - rV = 0
        
    where Σᵢⱼ = ρᵢⱼσᵢσⱼ is the covariance.
    
    Args:
        V: Option values, shape (batch,) - requires grad
        S: Asset prices, shape (batch, n_assets) - requires grad
        t: Time values, shape (batch,) - requires grad
        params: Basket parameters
        
    Returns:
        PDE residual, shape (batch,)
    """
    batch_size = S.shape[0]
    n_assets = params.n_assets
    r = params.r
    
    # Covariance matrix
    cov = torch.tensor(params.covariance, dtype=S.dtype, device=S.device)
    
    # Compute ∂V/∂t
    dV_dt = torch.autograd.grad(
        V.sum(), t, create_graph=True, retain_graph=True
    )[0]
    
    # Compute first derivatives ∂V/∂Sᵢ
    dV_dS = torch.autograd.grad(
        V.sum(), S, create_graph=True, retain_graph=True
    )[0]  # Shape: (batch, n_assets)
    
    # Drift term: Σᵢ rSᵢ∂V/∂Sᵢ
    drift = r * (S * dV_dS).sum(dim=-1)
    
    # Compute second derivatives ∂²V/∂Sᵢ∂Sⱼ
    # This is the expensive part - we need the full Hessian
    diffusion = torch.zeros(batch_size, dtype=V.dtype, device=V.device)
    
    for i in range(n_assets):
        # ∂²V/∂Sᵢ∂Sⱼ for all j
        d2V_dSi_dSj = torch.autograd.grad(
            dV_dS[:, i].sum(), S, create_graph=True, retain_graph=True
        )[0]  # Shape: (batch, n_assets)
        
        for j in range(n_assets):
            # Add contribution: ½ Σᵢⱼ SᵢSⱼ ∂²V/∂Sᵢ∂Sⱼ
            diffusion = diffusion + 0.5 * cov[i, j] * S[:, i] * S[:, j] * d2V_dSi_dSj[:, j]
    
    # Discount term: -rV
    discount = -r * V
    
    # PDE residual (should be zero for correct solution)
    residual = dV_dt + drift + diffusion + discount
    
    return residual


def generate_basket_collocation_lhs(
    params: BasketParams,
    n_interior: int = 20000,
    n_terminal: int = 10000,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Generate collocation points using Latin Hypercube Sampling.
    
    LHS provides better coverage of high-dimensional spaces than
    uniform random sampling.
    
    Args:
        params: Basket parameters
        n_interior: Number of interior (PDE) points
        n_terminal: Number of terminal (payoff) points
        seed: Random seed
        
    Returns:
        Dictionary with S_int, t_int, S_term tensors
    """
    n_assets = params.n_assets
    S_min = params.S_min
    S_max = params.S_max
    
    # Latin Hypercube Sampler for interior points (n_assets + 1 dimensions: S1..Sn, t)
    sampler_int = qmc.LatinHypercube(d=n_assets + 1, seed=seed)
    samples_int = sampler_int.random(n=n_interior)
    
    # Scale to domain
    S_int = np.zeros((n_interior, n_assets))
    for i in range(n_assets):
        S_int[:, i] = S_min[i] + samples_int[:, i] * (S_max[i] - S_min[i])
    t_int = samples_int[:, -1] * params.T  # t in [0, T)
    
    # Terminal points (only S dimensions, t = T)
    sampler_term = qmc.LatinHypercube(d=n_assets, seed=seed + 1)
    samples_term = sampler_term.random(n=n_terminal)
    
    S_term = np.zeros((n_terminal, n_assets))
    for i in range(n_assets):
        S_term[:, i] = S_min[i] + samples_term[:, i] * (S_max[i] - S_min[i])
    
    return {
        "S_int": torch.tensor(S_int, dtype=torch.float32),
        "t_int": torch.tensor(t_int, dtype=torch.float32),
        "S_term": torch.tensor(S_term, dtype=torch.float32),
    }


def monte_carlo_basket(
    params: BasketParams,
    S0: Optional[np.ndarray] = None,
    n_paths: int = 100000,
    seed: int = 42,
) -> dict:
    """
    Price basket option using Monte Carlo simulation.
    
    Simulates correlated GBM paths for all assets and computes
    discounted expected payoff.
    
    Args:
        params: Basket parameters
        S0: Initial prices (defaults to params.S0)
        n_paths: Number of simulation paths
        seed: Random seed
        
    Returns:
        Dictionary with price, std_error, confidence_interval
    """
    if S0 is None:
        S0 = params.S0
    
    rng = np.random.default_rng(seed)
    n_assets = params.n_assets
    
    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(params.correlation)
    
    # Generate correlated terminal prices
    Z = rng.standard_normal((n_paths, n_assets))
    Z_corr = Z @ L.T
    
    # GBM terminal values: S_T = S_0 * exp((r - σ²/2)T + σ√T Z)
    drift = (params.r - 0.5 * params.sigma**2) * params.T
    vol = params.sigma * np.sqrt(params.T)
    
    S_T = S0 * np.exp(drift + vol * Z_corr)
    
    # Basket value at maturity
    basket_T = (S_T * params.weights).sum(axis=1)
    
    # Payoff and discounting
    payoffs = np.maximum(basket_T - params.K, 0)
    discounted = np.exp(-params.r * params.T) * payoffs
    
    price = np.mean(discounted)
    std_error = np.std(discounted) / np.sqrt(n_paths)
    
    return {
        "price": float(price),
        "std_error": float(std_error),
        "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
        "n_paths": n_paths,
    }


def monte_carlo_basket_greeks(
    params: BasketParams,
    S0: Optional[np.ndarray] = None,
    n_paths: int = 100000,
    bump: float = 0.01,
    seed: int = 42,
) -> dict:
    """
    Compute basket option price and Greeks using bump-and-revalue.
    
    Returns:
        Dictionary with price, deltas (n_assets), gammas (n_assets x n_assets)
    """
    if S0 is None:
        S0 = params.S0.copy()
    
    # Base price
    base = monte_carlo_basket(params, S0, n_paths, seed)
    
    # Delta for each asset: ∂V/∂Sᵢ
    deltas = np.zeros(params.n_assets)
    for i in range(params.n_assets):
        S_up = S0.copy()
        S_up[i] *= (1 + bump)
        S_down = S0.copy()
        S_down[i] *= (1 - bump)
        
        price_up = monte_carlo_basket(params, S_up, n_paths, seed)["price"]
        price_down = monte_carlo_basket(params, S_down, n_paths, seed)["price"]
        
        deltas[i] = (price_up - price_down) / (2 * bump * S0[i])
    
    # Gamma (diagonal only for now): ∂²V/∂Sᵢ²
    gammas = np.zeros(params.n_assets)
    for i in range(params.n_assets):
        S_up = S0.copy()
        S_up[i] *= (1 + bump)
        S_down = S0.copy()
        S_down[i] *= (1 - bump)
        
        price_up = monte_carlo_basket(params, S_up, n_paths, seed)["price"]
        price_down = monte_carlo_basket(params, S_down, n_paths, seed)["price"]
        
        gammas[i] = (price_up - 2 * base["price"] + price_down) / (bump * S0[i])**2
    
    return {
        "price": base["price"],
        "std_error": base["std_error"],
        "deltas": deltas,
        "gammas": gammas,
    }
