"""
Merton Jump-Diffusion Model.

The asset price follows:
    dS/S = (r - λκ) dt + σ dW + (J-1) dN

where:
    - λ: jump intensity (Poisson process N)
    - J: jump size multiplier (lognormal: log(J) ~ N(μ_J, σ_J²))
    - κ = E[J-1] = exp(μ_J + σ_J²/2) - 1

The pricing PIDE (partial integro-differential equation):
    ∂V/∂t + (r - λκ)S ∂V/∂S + ½σ²S² ∂²V/∂S² - rV 
    + λ ∫ [V(SJ, t) - V(S, t)] g(J) dJ = 0

For PINNs, we approximate the integral term numerically.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from scipy.stats import norm


@dataclass
class MertonParams:
    """Merton jump-diffusion parameters."""
    r: float = 0.05           # Risk-free rate
    sigma: float = 0.20       # Diffusion volatility
    K: float = 100.0          # Strike price
    T: float = 1.0            # Time to maturity
    
    # Jump parameters
    lam: float = 0.5          # Jump intensity (λ per year)
    mu_J: float = -0.10       # Mean of log-jump size
    sigma_J: float = 0.15     # Std of log-jump size
    
    @property
    def kappa(self) -> float:
        """Expected relative jump size: κ = E[J-1]."""
        return np.exp(self.mu_J + 0.5 * self.sigma_J**2) - 1
    
    @property
    def drift(self) -> float:
        """Risk-neutral drift: r - λκ."""
        return self.r - self.lam * self.kappa


def merton_analytical_call(
    S: np.ndarray,
    params: MertonParams,
    tau: float,
    n_terms: int = 50,
) -> np.ndarray:
    """
    Merton's analytical formula for European call.
    
    The price is a weighted sum of Black-Scholes prices:
        C = Σ_{n=0}^∞ [e^{-λ'τ} (λ'τ)^n / n!] * BS(S, σ_n, r_n)
    
    where:
        λ' = λ(1 + κ)
        σ_n² = σ² + n σ_J² / τ
        r_n = r - λκ + n log(1+κ) / τ
    
    Args:
        S: Spot prices
        params: Merton parameters
        tau: Time to maturity
        n_terms: Number of terms in series (truncation)
        
    Returns:
        Call option prices
    """
    if tau < 1e-10:
        return np.maximum(S - params.K, 0)
    
    S = np.atleast_1d(S)
    
    # Special case: λ=0 reduces to Black-Scholes
    if params.lam < 1e-10:
        d1 = (np.log(S / params.K) + (params.r + 0.5 * params.sigma**2) * tau) / (params.sigma * np.sqrt(tau))
        d2 = d1 - params.sigma * np.sqrt(tau)
        return S * norm.cdf(d1) - params.K * np.exp(-params.r * tau) * norm.cdf(d2)
    
    lam_prime = params.lam * (1 + params.kappa)
    
    price = np.zeros_like(S, dtype=np.float64)
    
    for n in range(n_terms):
        # Poisson weight: e^{-λ'τ} (λ'τ)^n / n!
        if n == 0:
            log_weight = -lam_prime * tau
        else:
            log_weight = -lam_prime * tau + n * np.log(lam_prime * tau) - np.sum(np.log(np.arange(1, n+1)))
        weight = np.exp(log_weight)
        
        if weight < 1e-15:
            break
        
        # Adjusted parameters for n jumps
        sigma_n = np.sqrt(params.sigma**2 + n * params.sigma_J**2 / tau)
        r_n = params.r - params.lam * params.kappa + n * np.log(1 + params.kappa) / tau
        
        # Black-Scholes price with adjusted parameters
        d1 = (np.log(S / params.K) + (r_n + 0.5 * sigma_n**2) * tau) / (sigma_n * np.sqrt(tau))
        d2 = d1 - sigma_n * np.sqrt(tau)
        
        bs_price = S * norm.cdf(d1) - params.K * np.exp(-r_n * tau) * norm.cdf(d2)
        price += weight * bs_price
    
    return price


def merton_pide_residual(
    V_fn: Callable,
    S: torch.Tensor,
    t: torch.Tensor,
    params: MertonParams,
    n_quad: int = 20,
) -> torch.Tensor:
    """
    Compute the PIDE residual for Merton jump-diffusion.
    
    The PIDE:
        ∂V/∂t + (r - λκ)S ∂V/∂S + ½σ²S² ∂²V/∂S² - rV 
        + λ ∫ [V(SJ, t) - V(S, t)] g(J) dJ = 0
    
    The integral is computed using Gauss-Hermite quadrature.
    
    Args:
        V_fn: Function V(S, t) -> value (the PINN)
        S: Spot prices [batch]
        t: Times [batch]
        params: Merton parameters
        n_quad: Quadrature points for integral
        
    Returns:
        PIDE residual [batch]
    """
    S = S.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Evaluate V at (S, t)
    V = V_fn(S, t)
    
    # Compute derivatives
    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
    dV_dt = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]
    
    # Diffusion part (standard Black-Scholes with drift adjustment)
    diffusion = (
        dV_dt
        + params.drift * S * dV_dS
        + 0.5 * params.sigma**2 * S**2 * d2V_dS2
        - params.r * V
    )
    
    # Jump integral: E[V(SJ, t) - V(S, t)] where log(J) ~ N(μ_J, σ_J²)
    # Use Gauss-Hermite quadrature: ∫ f(x) e^{-x²} dx ≈ Σ w_i f(x_i)
    # Transform: if z ~ N(0,1), then J = exp(μ_J + σ_J * z)
    
    # Gauss-Hermite points and weights
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    x_gh = torch.tensor(x_gh, dtype=S.dtype, device=S.device)
    w_gh = torch.tensor(w_gh, dtype=S.dtype, device=S.device)
    
    # Transform to lognormal: z = x * sqrt(2), so J = exp(μ_J + σ_J * x * sqrt(2))
    sqrt2 = np.sqrt(2)
    
    jump_integral = torch.zeros_like(V)
    
    for i in range(n_quad):
        z = x_gh[i] * sqrt2
        J = torch.exp(torch.tensor(params.mu_J + params.sigma_J * z.item(), device=S.device))
        S_jumped = S * J
        V_jumped = V_fn(S_jumped, t)
        
        # Weight includes the 1/sqrt(π) from Gauss-Hermite
        jump_integral = jump_integral + w_gh[i] * (V_jumped - V) / np.sqrt(np.pi)
    
    # Full PIDE residual
    residual = diffusion + params.lam * jump_integral
    
    return residual


class MertonPINN(nn.Module):
    """
    Physics-Informed Neural Network for Merton jump-diffusion.
    
    Architecture: MLP with input normalization.
    """
    
    def __init__(
        self,
        params: MertonParams,
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        S_max: float = 300.0,
    ):
        super().__init__()
        self.params = params
        self.S_max = S_max
        
        # Build MLP
        layers = []
        in_dim = 2  # (S, t)
        
        act_fn = nn.Tanh() if activation == "tanh" else nn.GELU()
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.tensor(50.0))
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: (S, t) -> V."""
        # Normalize inputs
        S_norm = S / self.S_max
        t_norm = t / self.params.T
        
        x = torch.stack([S_norm, t_norm], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        raw = self.network(x).squeeze(-1)
        
        # Ensure positive output with soft-plus
        V = torch.nn.functional.softplus(raw) * self.output_scale
        
        return V
    
    def terminal_condition(self, S: torch.Tensor) -> torch.Tensor:
        """Call payoff at maturity."""
        return torch.maximum(S - self.params.K, torch.zeros_like(S))


class MertonPINNTrainer:
    """Trainer for Merton PINN with PIDE residual."""
    
    def __init__(
        self,
        model: MertonPINN,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        n_quad: int = 15,
    ):
        self.model = model
        self.params = model.params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.n_quad = n_quad
        self.history = []
    
    def compute_loss(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
    ) -> dict:
        """Compute PIDE + IC loss."""
        
        # PIDE residual at interior points
        def V_fn(S, t):
            return self.model(S, t)
        
        pde_residual = merton_pide_residual(
            V_fn, S_int, t_int, self.params, n_quad=self.n_quad
        )
        pde_loss = (pde_residual ** 2).mean()
        
        # Terminal condition
        V_term = self.model(S_term, torch.full_like(S_term, self.params.T))
        payoff = self.model.terminal_condition(S_term)
        ic_loss = ((V_term - payoff) ** 2).mean()
        
        total_loss = self.lambda_pde * pde_loss + self.lambda_ic * ic_loss
        
        return {
            "total": total_loss,
            "pde": pde_loss,
            "ic": ic_loss,
        }
    
    def train_step(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()
        losses = self.compute_loss(S_int, t_int, S_term)
        losses["total"].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        n_epochs: int,
        n_interior: int = 2000,
        n_terminal: int = 500,
        S_max: float = 300.0,
        log_every: int = 100,
    ) -> list:
        """Full training loop."""
        import time
        
        device = next(self.model.parameters()).device
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Sample points
            S_int = torch.rand(n_interior, device=device) * S_max
            t_int = torch.rand(n_interior, device=device) * self.params.T
            S_term = torch.rand(n_terminal, device=device) * S_max
            
            losses = self.train_step(S_int, t_int, S_term)
            self.history.append(losses)
            
            if epoch % log_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: total={losses['total']:.4f}, "
                      f"pde={losses['pde']:.4f}, ic={losses['ic']:.4f}, "
                      f"time={elapsed:.1f}s")
        
        return self.history
