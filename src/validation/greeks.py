"""
Greeks validation module for comparing PINN-computed sensitivities
against analytical Black-Scholes formulas.

This module is crucial for hedging applications where accurate Greeks
are required for risk management.
"""

import torch
import numpy as np
from scipy.stats import norm
from typing import Callable, Optional
from dataclasses import dataclass

from ..pde.black_scholes import BSParams


@dataclass
class GreeksValidationResult:
    """Results from Greeks validation."""
    delta_mae: float
    delta_max_error: float
    gamma_mae: float
    gamma_max_error: float
    theta_mae: float
    theta_max_error: float
    vega_mae: Optional[float] = None
    vega_max_error: Optional[float] = None
    
    def __repr__(self) -> str:
        lines = [
            "Greeks Validation Results:",
            f"  Delta: MAE={self.delta_mae:.6f}, Max={self.delta_max_error:.6f}",
            f"  Gamma: MAE={self.gamma_mae:.6f}, Max={self.gamma_max_error:.6f}",
            f"  Theta: MAE={self.theta_mae:.6f}, Max={self.theta_max_error:.6f}",
        ]
        if self.vega_mae is not None:
            lines.append(
                f"  Vega:  MAE={self.vega_mae:.6f}, Max={self.vega_max_error:.6f}"
            )
        return "\n".join(lines)


# =============================================================================
# PINN Greeks computation using automatic differentiation
# =============================================================================

def compute_pinn_delta(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
) -> torch.Tensor:
    """
    Compute Delta (dV/dS) from a trained PINN using automatic differentiation.
    
    Args:
        model: Trained PINN model
        S: Spot price tensor
        t: Time tensor
        params: Black-Scholes parameters (unused but kept for API consistency)
    
    Returns:
        Delta tensor with same shape as input
    """
    S = S.clone().requires_grad_(True)
    t = t.clone().detach()
    
    V = model(S, t)
    
    delta = torch.autograd.grad(
        V.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return delta.detach()


def compute_pinn_gamma(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
) -> torch.Tensor:
    """
    Compute Gamma (d2V/dS2) from a trained PINN using automatic differentiation.
    
    Args:
        model: Trained PINN model
        S: Spot price tensor
        t: Time tensor
        params: Black-Scholes parameters (unused but kept for API consistency)
    
    Returns:
        Gamma tensor with same shape as input
    """
    S = S.clone().requires_grad_(True)
    t = t.clone().detach()
    
    V = model(S, t)
    
    # First derivative (delta)
    delta = torch.autograd.grad(
        V.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Second derivative (gamma)
    gamma = torch.autograd.grad(
        delta.sum(),
        S,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return gamma.detach()


def compute_pinn_theta(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
) -> torch.Tensor:
    """
    Compute Theta (dV/dt) from a trained PINN using automatic differentiation.
    
    Note: This is the sensitivity with respect to current time t, not time to
    maturity. For Black-Scholes convention (negative theta for long calls),
    the sign may need adjustment depending on the time convention used.
    
    Args:
        model: Trained PINN model
        S: Spot price tensor
        t: Time tensor
        params: Black-Scholes parameters (unused but kept for API consistency)
    
    Returns:
        Theta tensor with same shape as input
    """
    S = S.clone().detach()
    t = t.clone().requires_grad_(True)
    
    V = model(S, t)
    
    theta = torch.autograd.grad(
        V.sum(),
        t,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return theta.detach()


def compute_pinn_vega(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
    forward_with_sigma: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Compute Vega (dV/dsigma) from a PINN that takes sigma as input.
    
    Unlike the standard PINN that learns for fixed sigma, this requires a model
    that takes sigma as an additional input parameter, OR a wrapper function
    that recreates the forward pass with sigma as a differentiable parameter.
    
    Args:
        model: PINN model (may not be used if forward_with_sigma handles everything)
        S: Spot price tensor
        t: Time tensor
        sigma: Volatility tensor (requires gradients)
        forward_with_sigma: Function (S, t, sigma) -> V that allows differentiating through sigma
    
    Returns:
        Vega tensor with same shape as input
    """
    S = S.clone().detach()
    t = t.clone().detach()
    sigma = sigma.clone().requires_grad_(True)
    
    V = forward_with_sigma(S, t, sigma)
    
    vega = torch.autograd.grad(
        V.sum(),
        sigma,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return vega.detach()


def compute_all_greeks(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    params: BSParams,
    forward_with_sigma: Optional[Callable] = None,
    sigma: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    Compute all Greeks from a trained PINN.
    
    Args:
        model: Trained PINN model
        S: Spot price tensor
        t: Time tensor
        params: Black-Scholes parameters
        forward_with_sigma: Optional function for computing vega
        sigma: Optional volatility tensor for vega computation
    
    Returns:
        Dictionary with keys: "delta", "gamma", "theta", and optionally "vega"
    """
    # Compute delta and gamma together efficiently
    S_grad = S.clone().requires_grad_(True)
    t_grad = t.clone().requires_grad_(True)
    
    V = model(S_grad, t_grad)
    
    # Delta
    delta = torch.autograd.grad(
        V.sum(),
        S_grad,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gamma
    gamma = torch.autograd.grad(
        delta.sum(),
        S_grad,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Theta
    theta = torch.autograd.grad(
        V.sum(),
        t_grad,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    result = {
        "delta": delta.detach(),
        "gamma": gamma.detach(),
        "theta": theta.detach(),
    }
    
    # Compute vega if sigma function is provided
    if forward_with_sigma is not None and sigma is not None:
        result["vega"] = compute_pinn_vega(
            model, S, t, sigma, forward_with_sigma
        )
    
    return result


# =============================================================================
# Analytical Greeks formulas (Black-Scholes)
# =============================================================================

def _compute_d1_d2(
    S: np.ndarray,
    K: float,
    tau: np.ndarray,
    r: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute d1 and d2 for Black-Scholes formula.
    
    Args:
        S: Spot prices
        K: Strike price
        tau: Time to maturity (T - t)
        r: Risk-free rate
        sigma: Volatility
    
    Returns:
        Tuple (d1, d2)
    """
    tau = np.maximum(tau, 1e-10)  # Avoid division by zero
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    return d1, d2


def analytical_delta(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    tau: Optional[np.ndarray] = None,
    option_type: str = "call",
) -> np.ndarray:
    """
    Analytical Delta (dV/dS) for Black-Scholes European options.
    
    Args:
        S: Spot prices
        K: Strike price
        T: Total time to maturity (used if tau not provided)
        r: Risk-free rate
        sigma: Volatility
        tau: Time to maturity array (T - t). If None, uses T.
        option_type: "call" or "put"
    
    Returns:
        Delta values
    """
    S = np.atleast_1d(S)
    if tau is None:
        tau = np.full_like(S, T)
    else:
        tau = np.atleast_1d(tau)
    
    d1, _ = _compute_d1_d2(S, K, tau, r, sigma)
    
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def analytical_gamma(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    tau: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Analytical Gamma (d2V/dS2) for Black-Scholes European options.
    
    Gamma is the same for calls and puts.
    
    Args:
        S: Spot prices
        K: Strike price
        T: Total time to maturity (used if tau not provided)
        r: Risk-free rate
        sigma: Volatility
        tau: Time to maturity array (T - t). If None, uses T.
    
    Returns:
        Gamma values
    """
    S = np.atleast_1d(S)
    if tau is None:
        tau = np.full_like(S, T)
    else:
        tau = np.atleast_1d(tau)
    
    tau = np.maximum(tau, 1e-10)
    d1, _ = _compute_d1_d2(S, K, tau, r, sigma)
    
    return norm.pdf(d1) / (S * sigma * np.sqrt(tau))


def analytical_theta(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    tau: Optional[np.ndarray] = None,
    option_type: str = "call",
) -> np.ndarray:
    """
    Analytical Theta (dV/dtau) for Black-Scholes European options.
    
    Note: This returns dV/dtau (derivative with respect to time-to-maturity),
    which is typically negative for long options (time decay).
    
    Args:
        S: Spot prices
        K: Strike price
        T: Total time to maturity (used if tau not provided)
        r: Risk-free rate
        sigma: Volatility
        tau: Time to maturity array (T - t). If None, uses T.
        option_type: "call" or "put"
    
    Returns:
        Theta values (typically negative for long options)
    """
    S = np.atleast_1d(S)
    if tau is None:
        tau = np.full_like(S, T)
    else:
        tau = np.atleast_1d(tau)
    
    tau = np.maximum(tau, 1e-10)
    d1, d2 = _compute_d1_d2(S, K, tau, r, sigma)
    
    # First term: time decay of option premium
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(tau))
    
    if option_type == "call":
        term2 = -r * K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * tau) * norm.cdf(-d2)
    
    return term1 + term2


def analytical_vega(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    tau: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Analytical Vega (dV/dsigma) for Black-Scholes European options.
    
    Vega is the same for calls and puts.
    
    Args:
        S: Spot prices
        K: Strike price
        T: Total time to maturity (used if tau not provided)
        r: Risk-free rate
        sigma: Volatility
        tau: Time to maturity array (T - t). If None, uses T.
    
    Returns:
        Vega values
    """
    S = np.atleast_1d(S)
    if tau is None:
        tau = np.full_like(S, T)
    else:
        tau = np.atleast_1d(tau)
    
    tau = np.maximum(tau, 1e-10)
    d1, _ = _compute_d1_d2(S, K, tau, r, sigma)
    
    return S * norm.pdf(d1) * np.sqrt(tau)


# =============================================================================
# Validation function
# =============================================================================

def validate_greeks(
    model: torch.nn.Module,
    params: BSParams,
    S_range: tuple[float, float] = (50.0, 150.0),
    t_values: Optional[np.ndarray] = None,
    n_spots: int = 50,
    option_type: str = "call",
    device: Optional[torch.device] = None,
) -> GreeksValidationResult:
    """
    Validate PINN-computed Greeks against analytical Black-Scholes formulas.
    
    Computes both PINN and analytical Greeks over a grid of (S, t) values
    and returns comparison metrics (MAE, max error) for each Greek.
    
    Args:
        model: Trained PINN model
        params: Black-Scholes parameters
        S_range: (min, max) range for spot prices
        t_values: Array of time values (default: [0.0, 0.25, 0.5, 0.75])
        n_spots: Number of spot price points
        option_type: "call" or "put"
        device: Torch device (default: inferred from model)
    
    Returns:
        GreeksValidationResult with MAE and max error for each Greek
    """
    if device is None:
        device = next(model.parameters()).device
    
    if t_values is None:
        t_values = np.array([0.0, 0.25, 0.5, 0.75])
    
    # Generate grid
    S_vals = np.linspace(S_range[0], S_range[1], n_spots)
    
    # Collect errors
    delta_errors = []
    gamma_errors = []
    theta_errors = []
    
    model.eval()
    
    for t_val in t_values:
        tau = params.T - t_val  # Time to maturity
        if tau <= 0:
            continue
        
        # Create tensors
        S_tensor = torch.tensor(S_vals, dtype=torch.float32, device=device)
        t_tensor = torch.full_like(S_tensor, t_val)
        
        # PINN Greeks
        pinn_greeks = compute_all_greeks(model, S_tensor, t_tensor, params)
        
        pinn_delta = pinn_greeks["delta"].cpu().numpy()
        pinn_gamma = pinn_greeks["gamma"].cpu().numpy()
        pinn_theta = pinn_greeks["theta"].cpu().numpy()
        
        # Analytical Greeks
        tau_arr = np.full_like(S_vals, tau)
        
        ana_delta = analytical_delta(
            S_vals, params.K, params.T, params.r, params.sigma, tau_arr, option_type
        )
        ana_gamma = analytical_gamma(
            S_vals, params.K, params.T, params.r, params.sigma, tau_arr
        )
        ana_theta = analytical_theta(
            S_vals, params.K, params.T, params.r, params.sigma, tau_arr, option_type
        )
        
        # PINN theta is dV/dt, analytical is dV/dtau = -dV/dt
        # So we negate PINN theta for comparison
        pinn_theta_adjusted = -pinn_theta
        
        delta_errors.extend(np.abs(pinn_delta - ana_delta))
        gamma_errors.extend(np.abs(pinn_gamma - ana_gamma))
        theta_errors.extend(np.abs(pinn_theta_adjusted - ana_theta))
    
    delta_errors = np.array(delta_errors)
    gamma_errors = np.array(gamma_errors)
    theta_errors = np.array(theta_errors)
    
    return GreeksValidationResult(
        delta_mae=float(np.mean(delta_errors)),
        delta_max_error=float(np.max(delta_errors)),
        gamma_mae=float(np.mean(gamma_errors)),
        gamma_max_error=float(np.max(gamma_errors)),
        theta_mae=float(np.mean(theta_errors)),
        theta_max_error=float(np.max(theta_errors)),
    )
