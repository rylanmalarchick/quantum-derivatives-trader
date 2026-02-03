"""
Heston Stochastic Volatility Model.

The model:
    dS = rS dt + √v S dW₁
    dv = κ(θ - v) dt + ξ√v dW₂
    ⟨dW₁, dW₂⟩ = ρ dt

The PDE (2D):
    ∂V/∂t + (1/2)vS²(∂²V/∂S²) + ρξvS(∂²V/∂S∂v) + (1/2)ξ²v(∂²V/∂v²)
    + rS(∂V/∂S) + κ(θ-v)(∂V/∂v) - rV = 0

This is more challenging for PINNs due to:
1. Higher dimensionality
2. Mixed derivatives
3. Non-constant volatility
"""

import torch
from dataclasses import dataclass


@dataclass
class HestonParams:
    """Heston model parameters."""
    r: float      # Risk-free rate
    kappa: float  # Mean reversion speed
    theta: float  # Long-run variance
    xi: float     # Vol of vol
    rho: float    # Correlation
    K: float      # Strike
    T: float      # Time to maturity
    v0: float     # Initial variance


def heston_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    params: HestonParams,
) -> torch.Tensor:
    """
    Compute the Heston PDE residual.

    This requires computing mixed partial derivatives, which is more complex
    than the 1D Black-Scholes case.

    Args:
        V: Option value
        S: Spot price
        v: Variance
        t: Time
        params: Heston parameters

    Returns:
        PDE residual
    """
    # Enable gradients
    S.requires_grad_(True)
    v.requires_grad_(True)
    t.requires_grad_(True)

    # First derivatives
    dV_dt = torch.autograd.grad(V.sum(), t, create_graph=True)[0]
    dV_dS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
    dV_dv = torch.autograd.grad(V.sum(), v, create_graph=True)[0]

    # Second derivatives
    d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S, create_graph=True)[0]
    d2V_dv2 = torch.autograd.grad(dV_dv.sum(), v, create_graph=True)[0]

    # Mixed derivative ∂²V/∂S∂v
    d2V_dSdv = torch.autograd.grad(dV_dS.sum(), v, create_graph=True)[0]

    # PDE residual
    residual = (
        dV_dt
        + 0.5 * v * S**2 * d2V_dS2
        + params.rho * params.xi * v * S * d2V_dSdv
        + 0.5 * params.xi**2 * v * d2V_dv2
        + params.r * S * dV_dS
        + params.kappa * (params.theta - v) * dV_dv
        - params.r * V
    )

    return residual


def heston_terminal_condition(
    S: torch.Tensor,
    params: HestonParams,
    option_type: str = "call"
) -> torch.Tensor:
    """Terminal payoff condition."""
    if option_type == "call":
        return torch.relu(S - params.K)
    else:
        return torch.relu(params.K - S)


def heston_boundary_conditions(
    S: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    params: HestonParams,
) -> dict[str, torch.Tensor]:
    """
    Boundary conditions for Heston model.

    The Heston PDE has boundaries in both S and v directions.
    """
    tau = params.T - t

    conditions = {}

    # S = 0: Call value is 0
    conditions["S_zero"] = torch.zeros_like(S)

    # S → ∞: V ≈ S - K*exp(-r*tau)
    conditions["S_large"] = S - params.K * torch.exp(-params.r * tau)

    # v = 0: Reduces to deterministic case
    # v → ∞: More complex, typically use asymptotic behavior

    return conditions
