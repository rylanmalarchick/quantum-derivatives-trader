"""
Analytical (closed-form) pricing solutions.

Used as ground truth for validating numerical methods.
"""

import numpy as np
import torch
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional

from ..pde.black_scholes import BSParams


@dataclass
class OptionContract:
    """Option contract specification."""
    K: float              # Strike price
    T: float              # Time to maturity
    option_type: str      # "call" or "put"
    exercise: str = "european"  # "european" or "american"


class AnalyticalPricer:
    """
    Closed-form option pricing formulas.

    Provides exact solutions for benchmarking numerical methods.
    """

    def __init__(self, r: float, sigma: float):
        """
        Args:
            r: Risk-free rate
            sigma: Volatility
        """
        self.r = r
        self.sigma = sigma

    def black_scholes(
        self,
        S: np.ndarray,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> np.ndarray:
        """
        Black-Scholes formula for European options.

        Args:
            S: Spot prices
            K: Strike
            T: Time to maturity
            option_type: "call" or "put"

        Returns:
            Option prices
        """
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)  # Avoid division by zero

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def delta(
        self,
        S: np.ndarray,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> np.ndarray:
        """Delta: ∂V/∂S"""
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )

        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(
        self,
        S: np.ndarray,
        K: float,
        T: float,
    ) -> np.ndarray:
        """Gamma: ∂²V/∂S² (same for call and put)"""
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )

        return norm.pdf(d1) / (S * self.sigma * np.sqrt(tau))

    def vega(
        self,
        S: np.ndarray,
        K: float,
        T: float,
    ) -> np.ndarray:
        """Vega: ∂V/∂σ (same for call and put)"""
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )

        return S * norm.pdf(d1) * np.sqrt(tau)

    def theta(
        self,
        S: np.ndarray,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> np.ndarray:
        """Theta: ∂V/∂t (negative of time decay)"""
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)

        term1 = -S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(tau))

        if option_type == "call":
            term2 = -self.r * K * np.exp(-self.r * tau) * norm.cdf(d2)
        else:
            term2 = self.r * K * np.exp(-self.r * tau) * norm.cdf(-d2)

        return term1 + term2

    def rho(
        self,
        S: np.ndarray,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> np.ndarray:
        """Rho: ∂V/∂r"""
        S = np.atleast_1d(S)
        tau = np.maximum(T, 1e-10)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * np.sqrt(tau)
        )
        d2 = d1 - self.sigma * np.sqrt(tau)

        if option_type == "call":
            return K * tau * np.exp(-self.r * tau) * norm.cdf(d2)
        else:
            return -K * tau * np.exp(-self.r * tau) * norm.cdf(-d2)

    def implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """
        Compute implied volatility using Newton-Raphson.

        Args:
            S: Spot price
            K: Strike
            T: Time to maturity
            market_price: Observed market price
            option_type: "call" or "put"
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Implied volatility
        """
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * market_price / S

        for _ in range(max_iter):
            # Store original sigma temporarily
            old_sigma = self.sigma
            self.sigma = sigma

            price = self.black_scholes(S, K, T, option_type)[0]
            vega = self.vega(S, K, T)[0]

            self.sigma = old_sigma  # Restore

            if abs(vega) < 1e-10:
                break

            diff = market_price - price
            if abs(diff) < tol:
                return sigma

            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 5.0))  # Bounds

        return sigma


class BarrierPricer:
    """
    Analytical pricing for barrier options.

    Barrier options have path-dependent payoffs that depend on
    whether the underlying crosses a barrier level.
    """

    def __init__(self, r: float, sigma: float):
        self.r = r
        self.sigma = sigma

    def down_and_out_call(
        self,
        S: float,
        K: float,
        H: float,  # Barrier level (H < S)
        T: float,
    ) -> float:
        """
        Down-and-out call option.

        Worthless if S ever touches H from above.
        """
        if S <= H:
            return 0.0

        lambda_ = (self.r + 0.5 * self.sigma**2) / self.sigma**2
        y = np.log(H**2 / (S * K)) / (self.sigma * np.sqrt(T)) + lambda_ * self.sigma * np.sqrt(T)

        x1 = np.log(S / H) / (self.sigma * np.sqrt(T)) + lambda_ * self.sigma * np.sqrt(T)
        y1 = np.log(H / S) / (self.sigma * np.sqrt(T)) + lambda_ * self.sigma * np.sqrt(T)

        pricer = AnalyticalPricer(self.r, self.sigma)
        vanilla = pricer.black_scholes(S, K, T, "call")[0]

        # Reflection principle adjustment
        adjustment = (H / S) ** (2 * lambda_ - 2) * pricer.black_scholes(
            H**2 / S, K, T, "call"
        )[0]

        return vanilla - adjustment

    def up_and_out_call(
        self,
        S: float,
        K: float,
        H: float,  # Barrier level (H > S, H > K)
        T: float,
    ) -> float:
        """
        Up-and-out call option.

        Worthless if S ever touches H from below.
        """
        if S >= H:
            return 0.0

        lambda_ = (self.r + 0.5 * self.sigma**2) / self.sigma**2

        pricer = AnalyticalPricer(self.r, self.sigma)

        # This is a simplified formula for H > K
        if H <= K:
            return 0.0

        vanilla = pricer.black_scholes(S, K, T, "call")[0]
        capped = pricer.black_scholes(S, H, T, "call")[0]

        adjustment = (H / S) ** (2 * lambda_ - 2) * (
            pricer.black_scholes(H**2 / S, K, T, "call")[0]
            - pricer.black_scholes(H**2 / S, H, T, "call")[0]
        )

        return vanilla - capped - adjustment
