"""
Heston Stochastic Volatility Model.

The asset and variance follow:
    dS = rS dt + √v S dW_S
    dv = κ(θ - v) dt + ξ √v dW_v
    
    dW_S · dW_v = ρ dt

Parameters:
    κ (kappa): Mean reversion speed
    θ (theta): Long-run variance
    ξ (xi/vol_of_vol): Volatility of variance
    ρ (rho): Correlation between asset and variance
    v_0: Initial variance

The Feller condition (2κθ > ξ²) ensures variance stays positive.

The pricing PDE (2D + time = 3D):
    ∂V/∂t + ½vS² ∂²V/∂S² + ρξvS ∂²V/∂S∂v + ½ξ²v ∂²V/∂v² 
    + rS ∂V/∂S + κ(θ-v) ∂V/∂v - rV = 0
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from scipy.integrate import quad
from scipy.special import factorial


@dataclass
class HestonParams:
    """Heston model parameters."""
    r: float = 0.05           # Risk-free rate
    K: float = 100.0          # Strike price
    T: float = 1.0            # Time to maturity
    
    # Variance process parameters
    kappa: float = 2.0        # Mean reversion speed
    theta: float = 0.04       # Long-run variance (σ² ~ 20% vol)
    xi: float = 0.3           # Vol of vol
    rho: float = -0.7         # Correlation (negative for equity)
    v0: float = 0.04          # Initial variance
    
    @property
    def feller_satisfied(self) -> bool:
        """Check Feller condition: 2κθ > ξ²."""
        return 2 * self.kappa * self.theta > self.xi ** 2
    
    @property
    def initial_vol(self) -> float:
        """Initial volatility σ_0 = √v_0."""
        return np.sqrt(self.v0)


def heston_characteristic_fn(
    u: complex,
    tau: float,
    params: HestonParams,
) -> complex:
    """
    Heston characteristic function for log-price.
    
    φ(u) = E[exp(iu·log(S_T/S_0))]
    
    Used for semi-analytical pricing via Fourier inversion.
    """
    kappa, theta, xi, rho, v0, r = (
        params.kappa, params.theta, params.xi, params.rho, params.v0, params.r
    )
    
    # Complex intermediate values
    d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
    
    # Avoid numerical issues
    exp_d_tau = np.exp(-d * tau)
    
    C = r * 1j * u * tau + (kappa * theta / xi**2) * (
        (kappa - rho * xi * 1j * u - d) * tau 
        - 2 * np.log((1 - g * exp_d_tau) / (1 - g))
    )
    
    D = ((kappa - rho * xi * 1j * u - d) / xi**2) * (
        (1 - exp_d_tau) / (1 - g * exp_d_tau)
    )
    
    return np.exp(C + D * v0)


def heston_call_price(
    S: float,
    params: HestonParams,
    tau: float,
    n_points: int = 100,
) -> float:
    """
    Semi-analytical Heston call price via Fourier inversion.
    
    C = S·P_1 - K·e^{-rτ}·P_2
    
    where P_1, P_2 are computed via characteristic function.
    """
    if tau < 1e-10:
        return max(S - params.K, 0)
    
    K = params.K
    r = params.r
    
    def integrand_P1(u):
        phi = heston_characteristic_fn(u - 1j, tau, params)
        phi_0 = heston_characteristic_fn(-1j, tau, params)
        return np.real(np.exp(-1j * u * np.log(K/S)) * phi / (1j * u * phi_0))
    
    def integrand_P2(u):
        phi = heston_characteristic_fn(u, tau, params)
        return np.real(np.exp(-1j * u * np.log(K/S)) * phi / (1j * u))
    
    # Numerical integration
    P1 = 0.5 + (1/np.pi) * quad(integrand_P1, 0, n_points, limit=200)[0]
    P2 = 0.5 + (1/np.pi) * quad(integrand_P2, 0, n_points, limit=200)[0]
    
    return S * P1 - K * np.exp(-r * tau) * P2


def heston_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    params: HestonParams,
) -> torch.Tensor:
    """
    Compute the Heston PDE residual.
    
    The 2D PDE:
        ∂V/∂t + ½vS² ∂²V/∂S² + ρξvS ∂²V/∂S∂v + ½ξ²v ∂²V/∂v² 
        + rS ∂V/∂S + κ(θ-v) ∂V/∂v - rV = 0
    
    Args:
        V: Option value [batch]
        S: Spot price [batch]
        v: Variance [batch]
        t: Time [batch]
        params: Heston parameters
        
    Returns:
        PDE residual [batch]
    """
    # First derivatives
    dV_dt = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
    dV_dv = torch.autograd.grad(V.sum(), v, create_graph=True)[0]
    
    # Second derivatives
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]
    d2V_dv2 = torch.autograd.grad(dV_dv.sum(), v, create_graph=True)[0]
    
    # Cross derivative
    d2V_dSdv = torch.autograd.grad(dV_dS.sum(), v, create_graph=True)[0]
    
    # PDE terms
    residual = (
        dV_dt
        + 0.5 * v * S**2 * d2V_dS2                    # Diffusion in S
        + params.rho * params.xi * v * S * d2V_dSdv  # Cross term
        + 0.5 * params.xi**2 * v * d2V_dv2           # Diffusion in v
        + params.r * S * dV_dS                        # Drift in S
        + params.kappa * (params.theta - v) * dV_dv  # Mean reversion in v
        - params.r * V                                # Discounting
    )
    
    return residual


class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network for Heston model.
    
    Input: (S, v, t) - 3D
    Output: V (option price)
    """
    
    def __init__(
        self,
        params: HestonParams,
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        S_max: float = 300.0,
        v_max: float = 1.0,
    ):
        super().__init__()
        self.params = params
        self.S_max = S_max
        self.v_max = v_max
        
        # Build MLP
        layers = []
        in_dim = 3  # (S, v, t)
        
        act_fn = nn.Tanh() if activation == "tanh" else nn.GELU()
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self.output_scale = nn.Parameter(torch.tensor(50.0))
    
    def forward(
        self, 
        S: torch.Tensor, 
        v: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: (S, v, t) -> V."""
        # Normalize inputs
        S_norm = S / self.S_max
        v_norm = v / self.v_max
        t_norm = t / self.params.T
        
        x = torch.stack([S_norm, v_norm, t_norm], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        raw = self.network(x).squeeze(-1)
        V = torch.nn.functional.softplus(raw) * self.output_scale
        
        return V
    
    def terminal_condition(self, S: torch.Tensor) -> torch.Tensor:
        """Call payoff at maturity."""
        return torch.maximum(S - self.params.K, torch.zeros_like(S))


class HestonPINNTrainer:
    """Trainer for Heston PINN."""
    
    def __init__(
        self,
        model: HestonPINN,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_bc: float = 1.0,
    ):
        self.model = model
        self.params = model.params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.history = []
    
    def sample_variance(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample variance from reasonable distribution (avoid v=0)."""
        # Sample from distribution concentrated around theta
        v = torch.abs(torch.randn(n, device=device) * 0.5 * self.params.theta + self.params.theta)
        v = torch.clamp(v, min=0.001, max=self.model.v_max)
        return v
    
    def compute_loss(
        self,
        S_int: torch.Tensor,
        v_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
        v_term: torch.Tensor,
    ) -> dict:
        """Compute PDE + IC + BC loss."""
        
        S_int = S_int.requires_grad_(True)
        v_int = v_int.requires_grad_(True)
        t_int = t_int.requires_grad_(True)
        
        # PDE residual
        V_int = self.model(S_int, v_int, t_int)
        pde_residual = heston_pde_residual(V_int, S_int, v_int, t_int, self.params)
        pde_loss = (pde_residual ** 2).mean()
        
        # Terminal condition
        t_T = torch.full_like(S_term, self.params.T)
        V_term = self.model(S_term, v_term, t_T)
        payoff = self.model.terminal_condition(S_term)
        ic_loss = ((V_term - payoff) ** 2).mean()
        
        # Boundary: V(0, v, t) = 0
        S_zero = torch.zeros_like(S_int[:100])
        V_zero = self.model(S_zero, v_int[:100], t_int[:100])
        bc_loss = (V_zero ** 2).mean()
        
        total_loss = (
            self.lambda_pde * pde_loss 
            + self.lambda_ic * ic_loss 
            + self.lambda_bc * bc_loss
        )
        
        return {
            "total": total_loss,
            "pde": pde_loss,
            "ic": ic_loss,
            "bc": bc_loss,
        }
    
    def train_step(
        self,
        S_int: torch.Tensor,
        v_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
        v_term: torch.Tensor,
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()
        losses = self.compute_loss(S_int, v_int, t_int, S_term, v_term)
        losses["total"].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train(
        self,
        n_epochs: int,
        n_interior: int = 3000,
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
            v_int = self.sample_variance(n_interior, device)
            t_int = torch.rand(n_interior, device=device) * self.params.T
            
            S_term = torch.rand(n_terminal, device=device) * S_max
            v_term = self.sample_variance(n_terminal, device)
            
            losses = self.train_step(S_int, v_int, t_int, S_term, v_term)
            self.history.append(losses)
            
            if epoch % log_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: total={losses['total']:.4f}, "
                      f"pde={losses['pde']:.4f}, ic={losses['ic']:.4f}, "
                      f"bc={losses['bc']:.4f}, time={elapsed:.1f}s")
        
        return self.history
