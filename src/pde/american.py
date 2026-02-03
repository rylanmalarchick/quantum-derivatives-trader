"""
American Options with Early Exercise.

American options can be exercised at any time before expiry.
The pricing problem becomes a free boundary problem (obstacle problem).

The option value must satisfy:
    1. V ≥ payoff(S)           (no-arbitrage: worth at least intrinsic)
    2. LV ≤ 0                  (PDE inequality in continuation region)
    3. (V - payoff) · LV = 0   (complementarity: exactly one is tight)

where L is the Black-Scholes operator.

The early exercise boundary S*(t) separates:
    - Continuation region: S > S*(t) for put, S < S*(t) for call
    - Exercise region: exercise immediately for payoff

PINN Approach:
    - Penalize violations of the obstacle constraint
    - Use smooth approximation (penalty method)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from scipy.stats import norm


@dataclass 
class AmericanParams:
    """American option parameters."""
    r: float = 0.05           # Risk-free rate
    sigma: float = 0.20       # Volatility
    K: float = 100.0          # Strike price
    T: float = 1.0            # Time to maturity
    option_type: str = "put"  # "put" or "call"
    
    @property
    def is_put(self) -> bool:
        return self.option_type.lower() == "put"


def american_payoff(S: torch.Tensor, params: AmericanParams) -> torch.Tensor:
    """Compute payoff for American option."""
    if params.is_put:
        return torch.maximum(params.K - S, torch.zeros_like(S))
    else:
        return torch.maximum(S - params.K, torch.zeros_like(S))


def bs_pde_operator(
    V: torch.Tensor,
    S: torch.Tensor,
    t: torch.Tensor,
    params: AmericanParams,
) -> torch.Tensor:
    """
    Apply Black-Scholes differential operator: LV
    
    LV = ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S - rV
    
    For continuation region: LV = 0
    For exercise region: LV < 0
    """
    dV_dt = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]
    
    LV = (
        dV_dt
        + 0.5 * params.sigma**2 * S**2 * d2V_dS2
        + params.r * S * dV_dS
        - params.r * V
    )
    
    return LV


def american_put_binomial(
    S0: float,
    params: AmericanParams,
    n_steps: int = 500,
) -> float:
    """
    Price American put using binomial tree (reference implementation).
    
    Cox-Ross-Rubinstein model.
    """
    dt = params.T / n_steps
    u = np.exp(params.sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(params.r * dt) - d) / (u - d)
    
    # Terminal values
    S = S0 * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)
    V = np.maximum(params.K - S, 0)
    
    # Backward induction with early exercise
    for i in range(n_steps - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)
        V_cont = np.exp(-params.r * dt) * (p * V[1:] + (1 - p) * V[:-1])
        V_exercise = np.maximum(params.K - S, 0)
        V = np.maximum(V_cont, V_exercise)
    
    return V[0]


def american_call_binomial(
    S0: float,
    params: AmericanParams,
    n_steps: int = 500,
) -> float:
    """
    Price American call using binomial tree.
    
    Note: For non-dividend stocks, American call = European call.
    """
    dt = params.T / n_steps
    u = np.exp(params.sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(params.r * dt) - d) / (u - d)
    
    # Terminal values
    S = S0 * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)
    V = np.maximum(S - params.K, 0)
    
    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        S = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)
        V_cont = np.exp(-params.r * dt) * (p * V[1:] + (1 - p) * V[:-1])
        V_exercise = np.maximum(S - params.K, 0)
        V = np.maximum(V_cont, V_exercise)
    
    return V[0]


class AmericanPINN(nn.Module):
    """
    PINN for American options using penalty method.
    
    The loss function includes:
    1. PDE residual (should be ≤ 0 in continuation region)
    2. Terminal condition
    3. Obstacle penalty: penalize V < payoff
    4. Complementarity: (V - payoff) * max(LV, 0) should be small
    """
    
    def __init__(
        self,
        params: AmericanParams,
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
        S_max: float = 300.0,
    ):
        super().__init__()
        self.params = params
        self.S_max = S_max
        
        layers = []
        in_dim = 2
        
        act_fn = nn.Tanh() if activation == "tanh" else nn.GELU()
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
        self.output_scale = nn.Parameter(torch.tensor(50.0))
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with obstacle constraint built-in."""
        S_norm = S / self.S_max
        t_norm = t / self.params.T
        
        x = torch.stack([S_norm, t_norm], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        raw = self.network(x).squeeze(-1)
        
        # Time value (non-negative via softplus)
        time_value = torch.nn.functional.softplus(raw) * self.output_scale
        
        # American option value = payoff + time value
        # This guarantees V >= payoff
        payoff = american_payoff(S, self.params)
        V = payoff + time_value
        
        return V
    
    def terminal_condition(self, S: torch.Tensor) -> torch.Tensor:
        """Terminal payoff."""
        return american_payoff(S, self.params)


class AmericanPINNTrainer:
    """
    Trainer for American PINN with penalty method.
    
    Uses soft penalty for:
    - PDE violation in continuation region
    - Obstacle violation (V < payoff)
    - Complementarity violation
    """
    
    def __init__(
        self,
        model: AmericanPINN,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_obstacle: float = 100.0,
        lambda_comp: float = 1.0,
    ):
        self.model = model
        self.params = model.params
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_obstacle = lambda_obstacle
        self.lambda_comp = lambda_comp
        self.history = []
    
    def compute_loss(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
    ) -> dict:
        """Compute American option loss."""
        
        S_int = S_int.requires_grad_(True)
        t_int = t_int.requires_grad_(True)
        
        # Forward pass
        V = self.model(S_int, t_int)
        payoff = american_payoff(S_int, self.params)
        
        # PDE operator
        LV = bs_pde_operator(V, S_int, t_int, self.params)
        
        # PDE loss: LV should be ≤ 0 (penalize positive LV)
        # In continuation region: LV = 0
        # In exercise region: LV < 0
        pde_loss = (torch.relu(LV) ** 2).mean()
        
        # Obstacle: V ≥ payoff (already enforced in forward, but add soft penalty)
        obstacle_violation = torch.relu(payoff - V)
        obstacle_loss = (obstacle_violation ** 2).mean()
        
        # Complementarity: (V - payoff) * max(LV, 0) ≈ 0
        # If V > payoff (continuation), then LV = 0
        # If V = payoff (exercise), then LV < 0 (and product is 0)
        excess = V - payoff
        comp_loss = (excess * torch.relu(LV)).mean()
        
        # Terminal condition
        V_term = self.model(S_term, torch.full_like(S_term, self.params.T))
        payoff_term = self.model.terminal_condition(S_term)
        ic_loss = ((V_term - payoff_term) ** 2).mean()
        
        total_loss = (
            self.lambda_pde * pde_loss
            + self.lambda_ic * ic_loss
            + self.lambda_obstacle * obstacle_loss
            + self.lambda_comp * comp_loss
        )
        
        return {
            "total": total_loss,
            "pde": pde_loss,
            "ic": ic_loss,
            "obstacle": obstacle_loss,
            "comp": comp_loss,
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
            S_int = torch.rand(n_interior, device=device) * S_max
            t_int = torch.rand(n_interior, device=device) * self.params.T
            S_term = torch.rand(n_terminal, device=device) * S_max
            
            losses = self.train_step(S_int, t_int, S_term)
            self.history.append(losses)
            
            if epoch % log_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: total={losses['total']:.4f}, "
                      f"pde={losses['pde']:.4f}, ic={losses['ic']:.4f}, "
                      f"obstacle={losses['obstacle']:.6f}, time={elapsed:.1f}s")
        
        return self.history


def find_early_exercise_boundary(
    model: AmericanPINN,
    t_values: np.ndarray,
    S_range: Tuple[float, float] = (50, 150),
    n_search: int = 100,
) -> np.ndarray:
    """
    Find the early exercise boundary S*(t) for American put.
    
    The boundary is where V(S, t) = payoff(S) transitions.
    
    For American put: S*(t) is the threshold below which exercise is optimal.
    """
    model.eval()
    device = next(model.parameters()).device
    
    boundaries = []
    
    with torch.no_grad():
        for t in t_values:
            S_vals = torch.linspace(S_range[0], S_range[1], n_search, device=device)
            t_vals = torch.full_like(S_vals, t)
            
            V = model(S_vals, t_vals)
            payoff = american_payoff(S_vals, model.params)
            
            # Find where V ≈ payoff (exercise boundary)
            diff = (V - payoff).abs().cpu().numpy()
            S_np = S_vals.cpu().numpy()
            
            # For put: find largest S where V ≈ payoff
            if model.params.is_put:
                exercise_mask = diff < 0.5  # Tolerance
                if exercise_mask.any():
                    boundary = S_np[exercise_mask].max()
                else:
                    boundary = S_range[0]
            else:
                # For call: find smallest S where V ≈ payoff
                exercise_mask = diff < 0.5
                if exercise_mask.any():
                    boundary = S_np[exercise_mask].min()
                else:
                    boundary = S_range[1]
            
            boundaries.append(boundary)
    
    return np.array(boundaries)
