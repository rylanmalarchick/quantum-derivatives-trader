"""
Synthetic option data generation for supervised learning.

Generates Black-Scholes option prices for training neural networks
in a supervised manner. This provides ground truth labels that can
be used alongside or instead of pure physics-informed training.

Use cases:
1. Pre-training PINNs before physics-informed fine-tuning
2. Hybrid loss combining PDE residual and supervised loss
3. Benchmarking PINN accuracy against known solutions
"""

import torch
import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

from ..pde.black_scholes import BSParams, bs_analytical


@dataclass
class OptionDataBatch:
    """
    A batch of option data with inputs and targets.

    Attributes:
        S: Spot prices, shape (batch_size,).
        t: Current times, shape (batch_size,).
        V: Option values (prices), shape (batch_size,).
        delta: Option deltas (dV/dS), shape (batch_size,) or None.
        gamma: Option gammas (d2V/dS2), shape (batch_size,) or None.
        theta: Option thetas (dV/dt), shape (batch_size,) or None.
    """

    S: torch.Tensor
    t: torch.Tensor
    V: torch.Tensor
    delta: Optional[torch.Tensor] = None
    gamma: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None


class SyntheticOptionData:
    """
    Generator for synthetic Black-Scholes option data.

    Generates training data with analytical Black-Scholes prices as labels.
    Useful for supervised pre-training or hybrid training approaches.

    Attributes:
        params: Black-Scholes parameters (r, sigma, K, T).
        S_max: Maximum spot price in generated data.
        option_type: Type of option ("call" or "put").
        device: PyTorch device for generated tensors.

    Example:
        >>> from src.pde.black_scholes import BSParams
        >>> params = BSParams(r=0.05, sigma=0.2, K=100.0, T=1.0)
        >>> generator = SyntheticOptionData(params, S_max=200.0)
        >>> batch = generator.generate_batch(batch_size=256)
        >>> print(batch.S.shape, batch.V.shape)
        torch.Size([256]) torch.Size([256])
    """

    def __init__(
        self,
        params: BSParams,
        S_max: float = 200.0,
        option_type: Literal["call", "put"] = "call",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            params: Black-Scholes parameters.
            S_max: Maximum spot price for data generation.
            option_type: "call" or "put".
            device: PyTorch device. Defaults to CPU.
        """
        self.params = params
        self.S_max = S_max
        self.option_type = option_type
        self.device = device or torch.device("cpu")

    def generate_batch(
        self,
        batch_size: int,
        include_greeks: bool = False,
        S_min: float = 1.0,
    ) -> OptionDataBatch:
        """
        Generate a batch of synthetic option data.

        Args:
            batch_size: Number of samples to generate.
            include_greeks: Whether to compute and include Greeks.
            S_min: Minimum spot price (avoid S=0 numerical issues).

        Returns:
            OptionDataBatch with spot prices, times, and option values.
        """
        # Sample random (S, t) points
        S = torch.rand(batch_size, device=self.device) * (self.S_max - S_min) + S_min
        t = torch.rand(batch_size, device=self.device) * self.params.T

        # Compute analytical Black-Scholes prices
        V = bs_analytical(S, t, self.params, self.option_type)

        batch = OptionDataBatch(S=S, t=t, V=V)

        if include_greeks:
            batch.delta = self._compute_delta(S, t)
            batch.gamma = self._compute_gamma(S, t)
            batch.theta = self._compute_theta(S, t)

        return batch

    def _compute_delta(
        self, S: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute analytical delta."""
        from scipy.stats import norm

        S_np = S.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
        tau = self.params.T - t_np
        tau = np.maximum(tau, 1e-10)

        d1 = (
            np.log(S_np / self.params.K)
            + (self.params.r + 0.5 * self.params.sigma**2) * tau
        ) / (self.params.sigma * np.sqrt(tau))

        if self.option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0

        return torch.tensor(delta, dtype=torch.float32, device=self.device)

    def _compute_gamma(
        self, S: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute analytical gamma."""
        from scipy.stats import norm

        S_np = S.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
        tau = self.params.T - t_np
        tau = np.maximum(tau, 1e-10)

        d1 = (
            np.log(S_np / self.params.K)
            + (self.params.r + 0.5 * self.params.sigma**2) * tau
        ) / (self.params.sigma * np.sqrt(tau))

        gamma = norm.pdf(d1) / (S_np * self.params.sigma * np.sqrt(tau))

        return torch.tensor(gamma, dtype=torch.float32, device=self.device)

    def _compute_theta(
        self, S: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute analytical theta."""
        from scipy.stats import norm

        S_np = S.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()
        tau = self.params.T - t_np
        tau = np.maximum(tau, 1e-10)

        d1 = (
            np.log(S_np / self.params.K)
            + (self.params.r + 0.5 * self.params.sigma**2) * tau
        ) / (self.params.sigma * np.sqrt(tau))
        d2 = d1 - self.params.sigma * np.sqrt(tau)

        # Theta for call option
        term1 = -S_np * norm.pdf(d1) * self.params.sigma / (2 * np.sqrt(tau))
        term2 = -self.params.r * self.params.K * np.exp(-self.params.r * tau) * norm.cdf(d2)

        if self.option_type == "call":
            theta = term1 + term2
        else:
            theta = term1 + self.params.r * self.params.K * np.exp(-self.params.r * tau) * norm.cdf(-d2)

        return torch.tensor(theta, dtype=torch.float32, device=self.device)

    def generate_grid_data(
        self,
        n_S: int = 50,
        n_t: int = 20,
        S_min: float = 1.0,
    ) -> OptionDataBatch:
        """
        Generate option data on a regular grid.

        Useful for evaluation and visualization.

        Args:
            n_S: Number of spot price points.
            n_t: Number of time points.
            S_min: Minimum spot price.

        Returns:
            OptionDataBatch with grid data.
        """
        S_1d = torch.linspace(S_min, self.S_max, n_S, device=self.device)
        t_1d = torch.linspace(0, self.params.T - 1e-6, n_t, device=self.device)

        t_mesh, S_mesh = torch.meshgrid(t_1d, S_1d, indexing="ij")
        S = S_mesh.flatten()
        t = t_mesh.flatten()

        V = bs_analytical(S, t, self.params, self.option_type)

        return OptionDataBatch(S=S, t=t, V=V)

    def create_dataloader(
        self,
        n_samples: int,
        batch_size: int,
        include_greeks: bool = False,
    ) -> "SyntheticDataLoader":
        """
        Create an iterable data loader for training.

        Args:
            n_samples: Total number of samples to generate.
            batch_size: Batch size for iteration.
            include_greeks: Whether to include Greeks in batches.

        Returns:
            SyntheticDataLoader that yields OptionDataBatch objects.
        """
        return SyntheticDataLoader(
            generator=self,
            n_samples=n_samples,
            batch_size=batch_size,
            include_greeks=include_greeks,
        )


class SyntheticDataLoader:
    """
    Iterable data loader for synthetic option data.

    Generates fresh random data each epoch (important for generalization).
    """

    def __init__(
        self,
        generator: SyntheticOptionData,
        n_samples: int,
        batch_size: int,
        include_greeks: bool = False,
    ):
        """
        Initialize the data loader.

        Args:
            generator: SyntheticOptionData instance.
            n_samples: Total samples per epoch.
            batch_size: Batch size.
            include_greeks: Include Greeks in batches.
        """
        self.generator = generator
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.include_greeks = include_greeks

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches."""
        n_batches = len(self)
        for i in range(n_batches):
            # Last batch may be smaller
            remaining = self.n_samples - i * self.batch_size
            current_batch_size = min(self.batch_size, remaining)

            yield self.generator.generate_batch(
                batch_size=current_batch_size,
                include_greeks=self.include_greeks,
            )
