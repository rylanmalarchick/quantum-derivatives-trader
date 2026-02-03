"""
Black-Scholes PDE for European options.

The PDE:
    ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

Boundary conditions (European call):
    V(0, t) = 0
    V(S, t) → S - Ke^{-r(T-t)} as S → ∞
    V(S, T) = max(S - K, 0)  [terminal condition]
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class BSParams:
    """Black-Scholes parameters."""
    r: float      # Risk-free rate
    sigma: float  # Volatility
    K: float      # Strike
    T: float      # Time to maturity


def bs_pde_residual(
    V: torch.Tensor,
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
    grad_fn: Callable,
) -> torch.Tensor:
    """
    Compute the PDE residual. Should be zero if V satisfies Black-Scholes.

    This is the physics constraint for the PINN.

    Args:
        V: Option value tensor
        S: Spot price tensor
        t: Time tensor
        params: Black-Scholes parameters
        grad_fn: Gradient function (autodiff)

    Returns:
        PDE residual tensor
    """
    # First derivatives
    dV_dt = grad_fn(V, t)
    dV_dS = grad_fn(V, S)

    # Second derivative
    d2V_dS2 = grad_fn(dV_dS, S)

    # PDE residual
    residual = (
        dV_dt
        + 0.5 * params.sigma**2 * S**2 * d2V_dS2
        + params.r * S * dV_dS
        - params.r * V
    )

    return residual


def bs_analytical(
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
    option_type: str = "call"
) -> torch.Tensor:
    """
    Closed-form Black-Scholes for European options.

    Args:
        S: Spot price tensor
        t: Current time tensor
        params: Black-Scholes parameters
        option_type: "call" or "put"

    Returns:
        Option value tensor
    """
    from scipy.stats import norm

    S_np = S.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    tau = params.T - t_np  # Time to maturity

    # Handle tau = 0 case
    tau = np.maximum(tau, 1e-10)

    d1 = (np.log(S_np / params.K) + (params.r + 0.5 * params.sigma**2) * tau) / (
        params.sigma * np.sqrt(tau)
    )
    d2 = d1 - params.sigma * np.sqrt(tau)

    if option_type == "call":
        price = S_np * norm.cdf(d1) - params.K * np.exp(-params.r * tau) * norm.cdf(d2)
    else:  # put
        price = params.K * np.exp(-params.r * tau) * norm.cdf(-d2) - S_np * norm.cdf(-d1)

    return torch.tensor(price, dtype=S.dtype, device=S.device)


def bs_delta(S: torch.Tensor, t: torch.Tensor, params: BSParams) -> torch.Tensor:
    """Analytical delta for European call."""
    from scipy.stats import norm

    S_np = S.detach().cpu().numpy()
    tau = params.T - t.detach().cpu().numpy()
    tau = np.maximum(tau, 1e-10)

    d1 = (np.log(S_np / params.K) + (params.r + 0.5 * params.sigma**2) * tau) / (
        params.sigma * np.sqrt(tau)
    )

    return torch.tensor(norm.cdf(d1), dtype=S.dtype, device=S.device)


def bs_gamma(S: torch.Tensor, t: torch.Tensor, params: BSParams) -> torch.Tensor:
    """Analytical gamma for European call."""
    from scipy.stats import norm

    S_np = S.detach().cpu().numpy()
    tau = params.T - t.detach().cpu().numpy()
    tau = np.maximum(tau, 1e-10)

    d1 = (np.log(S_np / params.K) + (params.r + 0.5 * params.sigma**2) * tau) / (
        params.sigma * np.sqrt(tau)
    )

    gamma = norm.pdf(d1) / (S_np * params.sigma * np.sqrt(tau))

    return torch.tensor(gamma, dtype=S.dtype, device=S.device)
