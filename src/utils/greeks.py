"""
Option Greeks computation module.

Provides both analytical (Black-Scholes) and autodiff-based methods
for computing option Greeks: delta, gamma, theta, vega, and rho.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import norm


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Container for option Greeks values.
    
    Attributes:
        delta: First derivative with respect to spot price (dV/dS).
        gamma: Second derivative with respect to spot price (d²V/dS²).
        theta: Time decay - derivative with respect to time (dV/dt).
        vega: Volatility sensitivity (dV/dσ).
        rho: Interest rate sensitivity (dV/dr).
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert Greeks to dictionary representation."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
        }


class GreeksCalculator:
    """Analytical Greeks calculator using Black-Scholes formulas.
    
    Computes option Greeks for European vanilla options using closed-form
    Black-Scholes solutions.
    
    Attributes:
        S: Spot price of the underlying asset.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility (annualized).
        q: Dividend yield (annualized, default 0).
        option_type: Call or put option.
    
    Example:
        >>> calc = GreeksCalculator(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> greeks = calc.compute_all()
        >>> print(f"Delta: {greeks.delta:.4f}")
        Delta: 0.6368
    """
    
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: Union[OptionType, str] = OptionType.CALL,
    ) -> None:
        """Initialize the Greeks calculator.
        
        Args:
            S: Spot price of the underlying asset.
            K: Strike price.
            T: Time to expiration in years.
            r: Risk-free interest rate (annualized).
            sigma: Volatility (annualized).
            q: Dividend yield (annualized, default 0).
            option_type: Type of option ('call' or 'put').
        
        Raises:
            ValueError: If any parameter is invalid (negative prices, etc.).
        """
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T <= 0:
            raise ValueError(f"Time to expiration must be positive, got {T}")
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {sigma}")
        
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        
        if isinstance(option_type, str):
            option_type = OptionType(option_type.lower())
        self.option_type = option_type
        
        # Precompute d1 and d2
        self._d1, self._d2 = self._compute_d1_d2()
    
    def _compute_d1_d2(self) -> Tuple[float, float]:
        """Compute d1 and d2 parameters for Black-Scholes formulas.
        
        Returns:
            Tuple of (d1, d2) values.
        """
        sqrt_T = math.sqrt(self.T)
        d1 = (
            math.log(self.S / self.K)
            + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt_T)
        d2 = d1 - self.sigma * sqrt_T
        return d1, d2
    
    def delta(self) -> float:
        """Compute delta (dV/dS).
        
        Delta measures the sensitivity of option price to changes in
        the underlying asset price.
        
        Returns:
            Delta value. For calls: [0, 1], for puts: [-1, 0].
        """
        discount_q = math.exp(-self.q * self.T)
        if self.option_type == OptionType.CALL:
            return discount_q * norm.cdf(self._d1)
        else:
            return discount_q * (norm.cdf(self._d1) - 1)
    
    def gamma(self) -> float:
        """Compute gamma (d²V/dS²).
        
        Gamma measures the rate of change of delta with respect to
        the underlying asset price. Same for calls and puts.
        
        Returns:
            Gamma value (always positive).
        """
        discount_q = math.exp(-self.q * self.T)
        return (
            discount_q * norm.pdf(self._d1)
            / (self.S * self.sigma * math.sqrt(self.T))
        )
    
    def theta(self) -> float:
        """Compute theta (dV/dt).
        
        Theta measures the time decay of the option. Note: this returns
        the per-day theta (divided by 365).
        
        Returns:
            Theta value (typically negative for long options).
        """
        sqrt_T = math.sqrt(self.T)
        discount_r = math.exp(-self.r * self.T)
        discount_q = math.exp(-self.q * self.T)
        
        term1 = -(
            self.S * discount_q * norm.pdf(self._d1) * self.sigma
            / (2 * sqrt_T)
        )
        
        if self.option_type == OptionType.CALL:
            term2 = -self.r * self.K * discount_r * norm.cdf(self._d2)
            term3 = self.q * self.S * discount_q * norm.cdf(self._d1)
        else:
            term2 = self.r * self.K * discount_r * norm.cdf(-self._d2)
            term3 = -self.q * self.S * discount_q * norm.cdf(-self._d1)
        
        # Return annualized theta; divide by 365 for daily
        return (term1 + term2 + term3) / 365.0
    
    def vega(self) -> float:
        """Compute vega (dV/dσ).
        
        Vega measures the sensitivity of option price to changes in
        volatility. Same for calls and puts.
        
        Returns:
            Vega value (per 1% change in volatility, divide by 100).
        """
        discount_q = math.exp(-self.q * self.T)
        return (
            self.S * discount_q * norm.pdf(self._d1) * math.sqrt(self.T)
        ) / 100.0  # Per 1% move in volatility
    
    def rho(self) -> float:
        """Compute rho (dV/dr).
        
        Rho measures the sensitivity of option price to changes in
        the risk-free interest rate.
        
        Returns:
            Rho value (per 1% change in interest rate, divide by 100).
        """
        discount_r = math.exp(-self.r * self.T)
        if self.option_type == OptionType.CALL:
            return (
                self.K * self.T * discount_r * norm.cdf(self._d2)
            ) / 100.0
        else:
            return (
                -self.K * self.T * discount_r * norm.cdf(-self._d2)
            ) / 100.0
    
    def compute_all(self) -> Greeks:
        """Compute all Greeks at once.
        
        Returns:
            Greeks dataclass containing all computed values.
        """
        return Greeks(
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho(),
        )
    
    def price(self) -> float:
        """Compute the Black-Scholes option price.
        
        Returns:
            Option price.
        """
        discount_r = math.exp(-self.r * self.T)
        discount_q = math.exp(-self.q * self.T)
        
        if self.option_type == OptionType.CALL:
            return (
                self.S * discount_q * norm.cdf(self._d1)
                - self.K * discount_r * norm.cdf(self._d2)
            )
        else:
            return (
                self.K * discount_r * norm.cdf(-self._d2)
                - self.S * discount_q * norm.cdf(-self._d1)
            )


def compute_greeks_autodiff(
    model: torch.nn.Module,
    S: torch.Tensor,
    t: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute option Greeks using PyTorch autodiff.
    
    Uses automatic differentiation to compute Greeks from a neural network
    model (e.g., a Physics-Informed Neural Network). This allows computing
    Greeks for any learned pricing function.
    
    Args:
        model: PyTorch model that takes (S, t) or (S, t, sigma, r) as input
            and returns option prices. The model should accept a tensor of
            shape (batch, num_features).
        S: Spot prices tensor of shape (batch,) or (batch, 1).
        t: Time to expiration tensor of shape (batch,) or (batch, 1).
        sigma: Optional volatility tensor for vega computation.
        r: Optional interest rate tensor for rho computation.
        create_graph: Whether to create computation graph for higher-order
            derivatives (default False for efficiency).
    
    Returns:
        Dictionary containing:
            - 'delta': dV/dS
            - 'gamma': d²V/dS²
            - 'theta': dV/dt
            - 'vega': dV/dσ (if sigma provided)
            - 'rho': dV/dr (if r provided)
            - 'price': Model output V
    
    Example:
        >>> model = MyPricingPINN()
        >>> S = torch.tensor([100.0], requires_grad=True)
        >>> t = torch.tensor([1.0], requires_grad=True)
        >>> greeks = compute_greeks_autodiff(model, S, t)
        >>> print(f"Delta: {greeks['delta'].item():.4f}")
    
    Note:
        Input tensors must have requires_grad=True for the derivatives
        you want to compute. The function will enable gradients as needed.
    """
    # Ensure inputs are properly shaped and have gradients enabled
    S = S.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    if S.dim() == 1:
        S = S.unsqueeze(-1)
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    
    # Build input tensor based on what's provided
    inputs = [S, t]
    input_names = ['S', 't']
    
    if sigma is not None:
        sigma = sigma.clone().detach().requires_grad_(True)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        inputs.append(sigma)
        input_names.append('sigma')
    
    if r is not None:
        r = r.clone().detach().requires_grad_(True)
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        inputs.append(r)
        input_names.append('r')
    
    x = torch.cat(inputs, dim=-1)
    
    # Forward pass
    model.eval()
    V = model(x)
    
    if V.dim() > 1:
        V = V.squeeze(-1)
    
    results: Dict[str, torch.Tensor] = {'price': V.detach()}
    
    # Compute delta: dV/dS
    dV_dS = torch.autograd.grad(
        outputs=V,
        inputs=S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,  # Need graph for gamma
        retain_graph=True,
    )[0]
    results['delta'] = dV_dS.squeeze(-1).detach()
    
    # Compute gamma: d²V/dS²
    d2V_dS2 = torch.autograd.grad(
        outputs=dV_dS,
        inputs=S,
        grad_outputs=torch.ones_like(dV_dS),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    results['gamma'] = d2V_dS2.squeeze(-1).detach()
    
    # Compute theta: dV/dt
    dV_dt = torch.autograd.grad(
        outputs=V,
        inputs=t,
        grad_outputs=torch.ones_like(V),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    results['theta'] = dV_dt.squeeze(-1).detach()
    
    # Compute vega if sigma provided: dV/dσ
    if sigma is not None:
        dV_dsigma = torch.autograd.grad(
            outputs=V,
            inputs=sigma,
            grad_outputs=torch.ones_like(V),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        results['vega'] = dV_dsigma.squeeze(-1).detach()
    
    # Compute rho if r provided: dV/dr
    if r is not None:
        dV_dr = torch.autograd.grad(
            outputs=V,
            inputs=r,
            grad_outputs=torch.ones_like(V),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        results['rho'] = dV_dr.squeeze(-1).detach()
    
    return results


def compute_greeks_batch(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> Dict[str, np.ndarray]:
    """Compute Greeks for a batch of spot prices.
    
    Vectorized computation of Black-Scholes Greeks for multiple spot prices.
    
    Args:
        S: Array of spot prices.
        K: Strike price.
        T: Time to expiration.
        r: Risk-free rate.
        sigma: Volatility.
        q: Dividend yield.
        option_type: 'call' or 'put'.
    
    Returns:
        Dictionary mapping Greek names to numpy arrays.
    """
    S = np.atleast_1d(S).astype(float)
    sqrt_T = np.sqrt(T)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    discount_r = np.exp(-r * T)
    discount_q = np.exp(-q * T)
    
    is_call = option_type.lower() == "call"
    
    # Delta
    if is_call:
        delta = discount_q * norm.cdf(d1)
    else:
        delta = discount_q * (norm.cdf(d1) - 1)
    
    # Gamma (same for call and put)
    gamma = discount_q * norm.pdf(d1) / (S * sigma * sqrt_T)
    
    # Theta
    term1 = -(S * discount_q * norm.pdf(d1) * sigma) / (2 * sqrt_T)
    if is_call:
        term2 = -r * K * discount_r * norm.cdf(d2)
        term3 = q * S * discount_q * norm.cdf(d1)
    else:
        term2 = r * K * discount_r * norm.cdf(-d2)
        term3 = -q * S * discount_q * norm.cdf(-d1)
    theta = (term1 + term2 + term3) / 365.0
    
    # Vega
    vega = (S * discount_q * norm.pdf(d1) * sqrt_T) / 100.0
    
    # Rho
    if is_call:
        rho = (K * T * discount_r * norm.cdf(d2)) / 100.0
    else:
        rho = (-K * T * discount_r * norm.cdf(-d2)) / 100.0
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
    }
