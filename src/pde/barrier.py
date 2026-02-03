"""
Barrier Options PDE for path-dependent option pricing.

Down-and-Out Call:
- Standard call option that becomes worthless if spot S ever touches barrier B
- Constraint: B < S₀ < K (barrier below initial spot, which is below strike for OTM)
- PDE is Black-Scholes in the region B < S < ∞
- Boundary condition at barrier: V(B, t) = 0 (knocked out)
- Terminal condition: V(S, T) = max(S - K, 0) for S > B

Analytical formula using reflection principle:
    V_do = V_bs(S) - (S/B)^(1-2r/σ²) * V_bs(B²/S)

where V_bs is the standard Black-Scholes call price.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple
from scipy.stats import norm


@dataclass
class BarrierParams:
    """Parameters for barrier options."""
    r: float = 0.05           # Risk-free rate
    sigma: float = 0.20       # Volatility
    K: float = 100.0          # Strike price
    T: float = 1.0            # Time to maturity
    B: float = 80.0           # Barrier level (B < S₀ for down-and-out)
    barrier_type: Literal["down-out-call"] = "down-out-call"
    
    def __post_init__(self):
        """Validate parameters."""
        if self.barrier_type == "down-out-call":
            if self.B >= self.K:
                raise ValueError(f"For down-out-call, barrier B={self.B} must be < strike K={self.K}")
        if self.B <= 0:
            raise ValueError(f"Barrier B={self.B} must be positive")
        if self.sigma <= 0:
            raise ValueError(f"Volatility sigma={self.sigma} must be positive")


def barrier_payoff(S: torch.Tensor, params: BarrierParams) -> torch.Tensor:
    """
    Compute terminal payoff for barrier option.
    
    For down-and-out call: max(S - K, 0) if S > B, else 0 (knocked out).
    
    Args:
        S: Spot price tensor
        params: Barrier option parameters
        
    Returns:
        Payoff tensor, accounting for knockout
    """
    if params.barrier_type == "down-out-call":
        # Standard call payoff
        call_payoff = torch.maximum(S - params.K, torch.zeros_like(S))
        # Zero out if at or below barrier (knocked out)
        knockout_mask = S <= params.B
        call_payoff = torch.where(knockout_mask, torch.zeros_like(call_payoff), call_payoff)
        return call_payoff
    else:
        raise NotImplementedError(f"Barrier type {params.barrier_type} not implemented")


def barrier_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor, 
    t: torch.Tensor,
    params: BarrierParams,
) -> torch.Tensor:
    """
    Compute Black-Scholes PDE residual for barrier option.
    
    The PDE is the same as standard Black-Scholes in the region B < S < ∞:
        ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
    
    The barrier condition is enforced separately via boundary loss.
    
    Args:
        V: Option value tensor (must be connected to S, t via autograd)
        S: Spot price tensor (requires_grad=True)
        t: Time tensor (requires_grad=True)
        params: Barrier option parameters
        
    Returns:
        PDE residual tensor
    """
    # First derivatives
    dV_dt = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0]
    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True, retain_graph=True)[0]
    
    # Second derivative
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True, retain_graph=True)[0]
    
    # Black-Scholes PDE residual
    residual = (
        dV_dt
        + 0.5 * params.sigma**2 * S**2 * d2V_dS2
        + params.r * S * dV_dS
        - params.r * V
    )
    
    return residual


def barrier_boundary_loss(
    model: nn.Module,
    t: torch.Tensor,
    params: BarrierParams,
) -> torch.Tensor:
    """
    Compute barrier boundary condition loss.
    
    For down-and-out: V(B, t) = 0 for all t.
    
    Args:
        model: PINN model that takes (S, t) as input
        t: Time tensor at which to enforce boundary
        params: Barrier option parameters
        
    Returns:
        Mean squared error at barrier boundary
    """
    # Evaluate at barrier
    S_barrier = torch.full_like(t, params.B)
    V_at_barrier = model(S_barrier, t)
    
    # Should be zero (knocked out)
    return (V_at_barrier ** 2).mean()


def _bs_call_price(S: np.ndarray, K: float, r: float, sigma: float, tau: np.ndarray) -> np.ndarray:
    """
    Standard Black-Scholes call price (numpy implementation).
    
    Args:
        S: Spot price array
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        tau: Time to maturity array
        
    Returns:
        Call option price array
    """
    # Handle edge cases
    tau = np.maximum(tau, 1e-10)
    S = np.maximum(S, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return price


def barrier_analytical_down_out_call(
    S: torch.Tensor,
    params: BarrierParams,
    tau: torch.Tensor,
) -> torch.Tensor:
    """
    Analytical price for down-and-out call using reflection principle.
    
    Formula:
        V_do = V_bs(S) - (S/B)^(1-2r/σ²) * V_bs(B²/S)
    
    where V_bs is the standard Black-Scholes call price.
    
    This uses the reflection principle: the knocked-out paths are equivalent
    to paths that crossed the barrier, which can be computed using image charges.
    
    Args:
        S: Spot price tensor
        params: Barrier option parameters (must be down-out-call)
        tau: Time to maturity tensor (T - t)
        
    Returns:
        Down-and-out call price tensor
    """
    if params.barrier_type != "down-out-call":
        raise ValueError(f"This function is for down-out-call, got {params.barrier_type}")
    
    S_np = S.detach().cpu().numpy()
    tau_np = tau.detach().cpu().numpy()
    
    # Ensure arrays
    S_np = np.atleast_1d(S_np)
    tau_np = np.atleast_1d(tau_np)
    
    # Broadcast if needed
    if S_np.shape != tau_np.shape:
        S_np, tau_np = np.broadcast_arrays(S_np, tau_np)
    
    r = params.r
    sigma = params.sigma
    K = params.K
    B = params.B
    
    # Exponent for reflection formula
    # lambda = (1 - 2r/σ²) = 1 - 2r/σ²
    lam = 1 - 2 * r / (sigma**2)
    
    # Standard BS call price at S
    V_bs_S = _bs_call_price(S_np, K, r, sigma, tau_np)
    
    # Reflected spot: S' = B²/S
    S_reflected = B**2 / np.maximum(S_np, 1e-10)
    
    # BS call price at reflected spot
    V_bs_reflected = _bs_call_price(S_reflected, K, r, sigma, tau_np)
    
    # Reflection coefficient: (S/B)^lambda
    reflection_coeff = (S_np / B) ** lam
    
    # Down-and-out call price
    V_do = V_bs_S - reflection_coeff * V_bs_reflected
    
    # Apply knockout: V = 0 if S <= B
    V_do = np.where(S_np <= B, 0.0, V_do)
    
    # Ensure non-negative (numerical precision)
    V_do = np.maximum(V_do, 0.0)
    
    return torch.tensor(V_do, dtype=S.dtype, device=S.device)


class BarrierPINN(nn.Module):
    """
    Physics-Informed Neural Network for barrier option pricing.
    
    Learns to satisfy:
    1. Black-Scholes PDE in region B < S < S_max
    2. Barrier boundary: V(B, t) = 0
    3. Upper boundary: V(S_max, t) → S - Ke^{-rτ}
    4. Terminal condition: V(S, T) = max(S - K, 0) for S > B
    """
    
    def __init__(
        self,
        params: BarrierParams,
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        S_max: float = 300.0,
    ):
        """
        Initialize barrier PINN.
        
        Args:
            params: Barrier option parameters
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("tanh" or "gelu")
            S_max: Maximum spot price for normalization
        """
        super().__init__()
        self.params = params
        self.S_max = S_max
        
        # Build network
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard barrier constraint.
        
        The output is multiplied by (S - B) to enforce V(B, t) = 0 exactly.
        
        Args:
            S: Spot price tensor
            t: Time tensor
            
        Returns:
            Option value tensor
        """
        # Ensure 1D
        S = S.view(-1)
        t = t.view(-1)
        
        # Normalize inputs
        S_norm = S / self.S_max
        t_norm = t / self.params.T
        
        # Stack and forward
        x = torch.stack([S_norm, t_norm], dim=-1)
        raw = self.network(x).squeeze(-1)
        
        # Non-negative via softplus
        positive_output = torch.nn.functional.softplus(raw) * self.output_scale
        
        # Hard barrier constraint: multiply by (S - B) / scale
        # This ensures V(B, t) = 0 exactly
        barrier_factor = torch.relu(S - self.params.B) / (self.S_max - self.params.B)
        
        V = positive_output * barrier_factor
        
        return V
    
    def terminal_condition(self, S: torch.Tensor) -> torch.Tensor:
        """Compute terminal payoff for barrier option."""
        return barrier_payoff(S, self.params)


class BarrierPINNTrainer:
    """
    Trainer for Barrier PINN with barrier boundary enforcement.
    
    Loss components:
    1. PDE residual loss (interior points above barrier)
    2. Terminal condition loss
    3. Barrier boundary loss: V(B, t) = 0
    4. Upper boundary loss: V(S_max, t) ≈ S - Ke^{-rτ}
    """
    
    def __init__(
        self,
        model: BarrierPINN,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_barrier: float = 100.0,
        lambda_upper: float = 1.0,
    ):
        """
        Initialize trainer.
        
        Args:
            model: BarrierPINN model
            lr: Learning rate
            lambda_pde: Weight for PDE residual loss
            lambda_ic: Weight for terminal/initial condition loss
            lambda_barrier: Weight for barrier boundary loss
            lambda_upper: Weight for upper boundary loss
        """
        self.model = model
        self.params = model.params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_barrier = lambda_barrier
        self.lambda_upper = lambda_upper
        self.history = []
    
    def compute_loss(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
        t_barrier: torch.Tensor,
    ) -> dict:
        """
        Compute all loss components.
        
        Args:
            S_int: Interior spot prices (above barrier)
            t_int: Interior times
            S_term: Terminal spot prices
            t_barrier: Times for barrier boundary
            
        Returns:
            Dictionary of loss components
        """
        # Enable gradients
        S_int = S_int.requires_grad_(True)
        t_int = t_int.requires_grad_(True)
        
        # Forward pass
        V = self.model(S_int, t_int)
        
        # PDE residual loss
        residual = barrier_pde_residual(V, S_int, t_int, self.params)
        pde_loss = (residual ** 2).mean()
        
        # Terminal condition loss
        t_T = torch.full_like(S_term, self.params.T)
        V_term = self.model(S_term, t_T)
        payoff_term = self.model.terminal_condition(S_term)
        ic_loss = ((V_term - payoff_term) ** 2).mean()
        
        # Barrier boundary loss: V(B, t) = 0
        barrier_loss = barrier_boundary_loss(self.model, t_barrier, self.params)
        
        # Upper boundary loss: V(S_max, t) ≈ S_max - K*exp(-r*tau)
        S_upper = torch.full_like(t_barrier, self.model.S_max)
        tau_upper = self.params.T - t_barrier
        V_upper = self.model(S_upper, t_barrier)
        target_upper = S_upper - self.params.K * torch.exp(-self.params.r * tau_upper)
        upper_loss = ((V_upper - target_upper) ** 2).mean()
        
        total_loss = (
            self.lambda_pde * pde_loss
            + self.lambda_ic * ic_loss
            + self.lambda_barrier * barrier_loss
            + self.lambda_upper * upper_loss
        )
        
        return {
            "total": total_loss,
            "pde": pde_loss,
            "ic": ic_loss,
            "barrier": barrier_loss,
            "upper": upper_loss,
        }
    
    def train_step(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
        t_barrier: torch.Tensor,
    ) -> dict:
        """Single training step."""
        self.optimizer.zero_grad()
        losses = self.compute_loss(S_int, t_int, S_term, t_barrier)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def generate_collocation_points(
        self,
        n_interior: int,
        n_terminal: int,
        n_barrier: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for training.
        
        Interior points are sampled in [B+ε, S_max] × [0, T).
        Terminal points are sampled in [B+ε, S_max].
        Barrier points are sampled at S=B across [0, T].
        
        Args:
            n_interior: Number of interior points
            n_terminal: Number of terminal points
            n_barrier: Number of barrier boundary points
            device: Torch device
            
        Returns:
            (S_int, t_int, S_term, t_barrier)
        """
        B = self.params.B
        S_max = self.model.S_max
        T = self.params.T
        
        # Interior points: above barrier
        eps = 1.0  # Small buffer above barrier
        S_int = torch.rand(n_interior, device=device) * (S_max - B - eps) + B + eps
        t_int = torch.rand(n_interior, device=device) * T * 0.99  # Avoid t=T
        
        # Terminal points: above barrier
        S_term = torch.rand(n_terminal, device=device) * (S_max - B - eps) + B + eps
        
        # Barrier boundary times
        t_barrier = torch.rand(n_barrier, device=device) * T
        
        return S_int, t_int, S_term, t_barrier
    
    def train(
        self,
        n_epochs: int,
        n_interior: int = 2000,
        n_terminal: int = 500,
        n_barrier: int = 200,
        log_every: int = 100,
    ) -> list:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of training epochs
            n_interior: Number of interior collocation points per epoch
            n_terminal: Number of terminal collocation points per epoch
            n_barrier: Number of barrier boundary points per epoch
            log_every: Log every N epochs
            
        Returns:
            Training history
        """
        import time
        
        device = next(self.model.parameters()).device
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Generate fresh collocation points each epoch
            S_int, t_int, S_term, t_barrier = self.generate_collocation_points(
                n_interior, n_terminal, n_barrier, device
            )
            
            losses = self.train_step(S_int, t_int, S_term, t_barrier)
            self.history.append(losses)
            
            if epoch % log_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: total={losses['total']:.4f}, "
                      f"pde={losses['pde']:.4f}, ic={losses['ic']:.4f}, "
                      f"barrier={losses['barrier']:.6f}, time={elapsed:.1f}s")
        
        return self.history


def evaluate_barrier_pinn(
    model: BarrierPINN,
    params: BarrierParams,
    n_points: int = 100,
) -> dict:
    """
    Evaluate barrier PINN against analytical solution.
    
    Args:
        model: Trained BarrierPINN
        params: Barrier option parameters
        n_points: Number of evaluation points
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Spot prices above barrier
    S_vals = np.linspace(params.B + 1, model.S_max * 0.8, n_points)
    tau = params.T  # Evaluate at t=0
    
    # Analytical prices
    S_torch = torch.tensor(S_vals, dtype=torch.float32)
    tau_torch = torch.full_like(S_torch, tau)
    analytical_prices = barrier_analytical_down_out_call(S_torch, params, tau_torch).numpy()
    
    # PINN prices
    with torch.no_grad():
        t_torch = torch.zeros_like(S_torch)
        pinn_prices = model(S_torch, t_torch).numpy()
    
    # Metrics
    mse = np.mean((pinn_prices - analytical_prices) ** 2)
    mae = np.mean(np.abs(pinn_prices - analytical_prices))
    max_error = np.max(np.abs(pinn_prices - analytical_prices))
    
    # Relative error (avoid division by zero)
    mask = analytical_prices > 0.1
    if mask.any():
        rel_error = np.mean(np.abs(pinn_prices[mask] - analytical_prices[mask]) / analytical_prices[mask]) * 100
    else:
        rel_error = 0.0
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_error": float(max_error),
        "mean_rel_error_pct": float(rel_error),
        "S_vals": S_vals.tolist(),
        "pinn_prices": pinn_prices.tolist(),
        "analytical_prices": analytical_prices.tolist(),
    }
