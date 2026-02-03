"""
Monte Carlo pricing engine.

Classical Monte Carlo baseline for benchmarking quantum methods.
Includes variance reduction techniques.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
import torch


@dataclass
class MCResult:
    """Monte Carlo simulation result."""
    price: float
    std_error: float
    n_paths: int
    confidence_interval: tuple[float, float]


class MonteCarloEngine:
    """
    Monte Carlo engine for option pricing.

    Simulates paths under risk-neutral measure and computes
    discounted expected payoff.
    """

    def __init__(
        self,
        r: float,
        sigma: float,
        seed: Optional[int] = None,
    ):
        self.r = r
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def simulate_gbm(
        self,
        S0: float,
        T: float,
        n_paths: int,
        n_steps: int = 1,
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.

        Args:
            S0: Initial spot price
            T: Time horizon
            n_paths: Number of paths
            n_steps: Number of time steps

        Returns:
            (n_paths, n_steps + 1) array of prices
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        drift = (self.r - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for t in range(1, n_steps + 1):
            Z = self.rng.standard_normal(n_paths)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * Z)

        return paths

    def price_european(
        self,
        payoff_fn: Callable,
        S0: float,
        T: float,
        n_paths: int = 100000,
    ) -> MCResult:
        """
        Price European option via Monte Carlo.

        Args:
            payoff_fn: Payoff function f(S_T) -> payoff
            S0: Initial spot
            T: Time to maturity
            n_paths: Number of simulation paths

        Returns:
            MCResult with price and statistics
        """
        # Simulate terminal prices
        Z = self.rng.standard_normal(n_paths)
        S_T = S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )

        # Compute payoffs
        payoffs = np.array([payoff_fn(s) for s in S_T])

        # Discounted expectation
        discounted = np.exp(-self.r * T) * payoffs

        price = np.mean(discounted)
        std = np.std(discounted) / np.sqrt(n_paths)

        return MCResult(
            price=price,
            std_error=std,
            n_paths=n_paths,
            confidence_interval=(price - 1.96 * std, price + 1.96 * std),
        )

    def price_with_antithetic(
        self,
        payoff_fn: Callable,
        S0: float,
        T: float,
        n_paths: int = 50000,
    ) -> MCResult:
        """
        Monte Carlo with antithetic variates for variance reduction.

        Uses both Z and -Z to reduce variance.
        """
        Z = self.rng.standard_normal(n_paths)

        # Regular paths
        S_T_pos = S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )

        # Antithetic paths
        S_T_neg = S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T - self.sigma * np.sqrt(T) * Z
        )

        # Average payoffs
        payoffs_pos = np.array([payoff_fn(s) for s in S_T_pos])
        payoffs_neg = np.array([payoff_fn(s) for s in S_T_neg])
        payoffs = (payoffs_pos + payoffs_neg) / 2

        discounted = np.exp(-self.r * T) * payoffs

        price = np.mean(discounted)
        std = np.std(discounted) / np.sqrt(n_paths)

        return MCResult(
            price=price,
            std_error=std,
            n_paths=2 * n_paths,  # Effectively 2x paths
            confidence_interval=(price - 1.96 * std, price + 1.96 * std),
        )

    def price_with_control_variate(
        self,
        payoff_fn: Callable,
        S0: float,
        K: float,
        T: float,
        n_paths: int = 100000,
    ) -> MCResult:
        """
        Monte Carlo with control variate for variance reduction.

        Uses the stock price as control variate (known expectation).
        """
        Z = self.rng.standard_normal(n_paths)
        S_T = S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )

        # Payoffs
        payoffs = np.array([payoff_fn(s) for s in S_T])
        discounted = np.exp(-self.r * T) * payoffs

        # Control variate: discounted S_T
        control = np.exp(-self.r * T) * S_T
        control_mean = S0  # E[e^{-rT} S_T] = S_0

        # Optimal coefficient
        cov = np.cov(discounted, control)[0, 1]
        var_control = np.var(control)
        c = -cov / var_control if var_control > 0 else 0

        # Adjusted payoffs
        adjusted = discounted + c * (control - control_mean)

        price = np.mean(adjusted)
        std = np.std(adjusted) / np.sqrt(n_paths)

        return MCResult(
            price=price,
            std_error=std,
            n_paths=n_paths,
            confidence_interval=(price - 1.96 * std, price + 1.96 * std),
        )

    def price_asian(
        self,
        S0: float,
        K: float,
        T: float,
        n_paths: int = 100000,
        n_steps: int = 252,  # Daily for 1 year
        average_type: str = "arithmetic",
    ) -> MCResult:
        """
        Price Asian option (average price option).

        Args:
            S0: Initial spot
            K: Strike
            T: Time to maturity
            n_paths: Simulation paths
            n_steps: Averaging points
            average_type: "arithmetic" or "geometric"

        Returns:
            MCResult for Asian call
        """
        paths = self.simulate_gbm(S0, T, n_paths, n_steps)

        if average_type == "arithmetic":
            avg = np.mean(paths[:, 1:], axis=1)  # Exclude t=0
        else:  # geometric
            avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        # Asian call payoff
        payoffs = np.maximum(avg - K, 0)
        discounted = np.exp(-self.r * T) * payoffs

        price = np.mean(discounted)
        std = np.std(discounted) / np.sqrt(n_paths)

        return MCResult(
            price=price,
            std_error=std,
            n_paths=n_paths,
            confidence_interval=(price - 1.96 * std, price + 1.96 * std),
        )

    def price_basket(
        self,
        S0: np.ndarray,
        weights: np.ndarray,
        K: float,
        T: float,
        sigma: np.ndarray,
        correlation: np.ndarray,
        n_paths: int = 100000,
    ) -> MCResult:
        """
        Price basket option on multiple assets.

        Args:
            S0: (n_assets,) initial prices
            weights: (n_assets,) basket weights
            K: Strike
            T: Time to maturity
            sigma: (n_assets,) volatilities
            correlation: (n_assets, n_assets) correlation matrix
            n_paths: Simulation paths

        Returns:
            MCResult for basket call
        """
        n_assets = len(S0)

        # Cholesky decomposition for correlated normals
        L = np.linalg.cholesky(correlation)

        # Generate correlated terminal prices
        Z = self.rng.standard_normal((n_paths, n_assets))
        Z_corr = Z @ L.T

        S_T = np.zeros((n_paths, n_assets))
        for i in range(n_assets):
            S_T[:, i] = S0[i] * np.exp(
                (self.r - 0.5 * sigma[i]**2) * T
                + sigma[i] * np.sqrt(T) * Z_corr[:, i]
            )

        # Basket value
        basket = S_T @ weights

        # Payoff
        payoffs = np.maximum(basket - K, 0)
        discounted = np.exp(-self.r * T) * payoffs

        price = np.mean(discounted)
        std = np.std(discounted) / np.sqrt(n_paths)

        return MCResult(
            price=price,
            std_error=std,
            n_paths=n_paths,
            confidence_interval=(price - 1.96 * std, price + 1.96 * std),
        )
