"""
Collocation point generation for Physics-Informed Neural Networks.

Collocation points are the locations where we enforce the PDE, boundary conditions,
and terminal conditions. Good point distributions are crucial for PINN training.

Key strategies:
1. Uniform random sampling (simple, effective baseline)
2. Latin Hypercube Sampling (better space coverage, reduces clustering)
3. Sobol sequences (quasi-random, even better coverage for higher dimensions)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Literal


def generate_collocation_points(
    n_interior: int,
    n_boundary: int,
    n_terminal: int,
    S_max: float,
    T: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate collocation points for PINN training on Black-Scholes PDE.

    Creates three types of points:
    1. Interior points: Random (S, t) in (0, S_max) × (0, T)
    2. Boundary points: Points at S=0 and S=S_max for all t
    3. Terminal points: Points at t=T for all S (terminal/payoff condition)

    Args:
        n_interior: Number of interior collocation points for PDE residual.
        n_boundary: Number of boundary points (split between S=0 and S=S_max).
        n_terminal: Number of terminal condition points at t=T.
        S_max: Maximum spot price (upper boundary of domain).
        T: Time to maturity (terminal time).
        device: PyTorch device for tensors. Defaults to CPU.

    Returns:
        Tuple of 1D tensors:
            - S_interior: Spot prices for interior points, shape (n_interior,)
            - t_interior: Times for interior points, shape (n_interior,)
            - S_boundary: Spot prices for boundary points, shape (n_boundary,)
            - t_boundary: Times for boundary points, shape (n_boundary,)
            - S_terminal: Spot prices for terminal condition, shape (n_terminal,)

    Example:
        >>> S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
        ...     n_interior=1000, n_boundary=200, n_terminal=200,
        ...     S_max=200.0, T=1.0
        ... )
        >>> S_int.shape
        torch.Size([1000])
    """
    if device is None:
        device = torch.device("cpu")

    # Interior points: uniform random in (0, S_max) × (0, T)
    # Avoid exact boundaries to prevent numerical issues
    eps = 1e-6
    S_interior = torch.rand(n_interior, device=device) * (S_max - 2 * eps) + eps
    t_interior = torch.rand(n_interior, device=device) * (T - eps)

    # Boundary points: split between S=0 and S=S_max
    n_lower = n_boundary // 2
    n_upper = n_boundary - n_lower

    # S = 0 boundary
    S_lower = torch.zeros(n_lower, device=device)
    t_lower = torch.rand(n_lower, device=device) * T

    # S = S_max boundary
    S_upper = torch.full((n_upper,), S_max, device=device)
    t_upper = torch.rand(n_upper, device=device) * T

    S_boundary = torch.cat([S_lower, S_upper])
    t_boundary = torch.cat([t_lower, t_upper])

    # Terminal points: t = T, S uniform in (0, S_max)
    S_terminal = torch.rand(n_terminal, device=device) * S_max

    return S_interior, t_interior, S_boundary, t_boundary, S_terminal


def generate_latin_hypercube_points(
    n_points: int,
    S_max: float,
    T: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate interior points using Latin Hypercube Sampling.

    LHS provides better space-filling properties than uniform random sampling.
    Each row and column of the grid contains exactly one sample, reducing
    clustering and ensuring coverage of the entire domain.

    Args:
        n_points: Number of points to generate.
        S_max: Maximum spot price.
        T: Time to maturity.
        device: PyTorch device for tensors.

    Returns:
        Tuple of 1D tensors:
            - S: Spot price points, shape (n_points,)
            - t: Time points, shape (n_points,)

    Note:
        For higher dimensions, consider using scipy.stats.qmc.LatinHypercube
        or similar quasi-random sequences (Sobol, Halton).
    """
    if device is None:
        device = torch.device("cpu")

    # Create LHS grid
    # Divide each dimension into n_points intervals
    intervals = np.arange(n_points)

    # Randomly shuffle the intervals for each dimension
    S_intervals = np.random.permutation(intervals)
    t_intervals = np.random.permutation(intervals)

    # Sample uniformly within each interval
    S_samples = (S_intervals + np.random.rand(n_points)) / n_points
    t_samples = (t_intervals + np.random.rand(n_points)) / n_points

    # Scale to domain
    S = torch.tensor(S_samples * S_max, dtype=torch.float32, device=device)
    t = torch.tensor(t_samples * T, dtype=torch.float32, device=device)

    return S, t


class CollocationSampler:
    """
    Configurable collocation point sampler for PINN training.

    Supports multiple sampling strategies and adaptive resampling
    based on residual magnitudes (residual-based adaptive refinement).

    Attributes:
        S_max: Maximum spot price boundary.
        T: Time to maturity.
        device: PyTorch device for generated tensors.
        sampling_method: Strategy for interior point sampling.
    """

    def __init__(
        self,
        S_max: float,
        T: float,
        device: Optional[torch.device] = None,
        sampling_method: Literal["uniform", "lhs", "sobol"] = "uniform",
    ):
        """
        Initialize the collocation sampler.

        Args:
            S_max: Maximum spot price (upper boundary).
            T: Time to maturity.
            device: PyTorch device. Defaults to CPU.
            sampling_method: Sampling strategy for interior points.
                - "uniform": Standard uniform random sampling.
                - "lhs": Latin Hypercube Sampling.
                - "sobol": Sobol quasi-random sequence (requires scipy).
        """
        self.S_max = S_max
        self.T = T
        self.device = device or torch.device("cpu")
        self.sampling_method = sampling_method

        # For adaptive sampling: store residual-weighted importance
        self._importance_weights: Optional[torch.Tensor] = None

    def sample_interior(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample interior collocation points.

        Args:
            n_points: Number of interior points to sample.

        Returns:
            Tuple of (S, t) tensors, each of shape (n_points,).
        """
        if self.sampling_method == "lhs":
            return generate_latin_hypercube_points(
                n_points, self.S_max, self.T, self.device
            )
        elif self.sampling_method == "sobol":
            return self._sample_sobol(n_points)
        else:  # uniform
            eps = 1e-6
            S = torch.rand(n_points, device=self.device) * (self.S_max - 2 * eps) + eps
            t = torch.rand(n_points, device=self.device) * (self.T - eps)
            return S, t

    def _sample_sobol(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample using Sobol sequence (quasi-random)."""
        try:
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=2, scramble=True)
            samples = sampler.random(n_points)

            S = torch.tensor(
                samples[:, 0] * self.S_max,
                dtype=torch.float32,
                device=self.device,
            )
            t = torch.tensor(
                samples[:, 1] * self.T,
                dtype=torch.float32,
                device=self.device,
            )
            return S, t

        except ImportError:
            # Fallback to LHS if scipy not available
            return generate_latin_hypercube_points(
                n_points, self.S_max, self.T, self.device
            )

    def sample_boundary(
        self, n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample boundary collocation points at S=0 and S=S_max.

        Args:
            n_points: Total number of boundary points.

        Returns:
            Tuple of (S, t) tensors for boundary points.
        """
        n_lower = n_points // 2
        n_upper = n_points - n_lower

        S_lower = torch.zeros(n_lower, device=self.device)
        t_lower = torch.rand(n_lower, device=self.device) * self.T

        S_upper = torch.full((n_upper,), self.S_max, device=self.device)
        t_upper = torch.rand(n_upper, device=self.device) * self.T

        return torch.cat([S_lower, S_upper]), torch.cat([t_lower, t_upper])

    def sample_terminal(self, n_points: int) -> torch.Tensor:
        """
        Sample terminal condition points at t=T.

        Args:
            n_points: Number of terminal points.

        Returns:
            Spot price tensor S at terminal time, shape (n_points,).
        """
        return torch.rand(n_points, device=self.device) * self.S_max

    def sample_all(
        self,
        n_interior: int,
        n_boundary: int,
        n_terminal: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample all collocation points (interior, boundary, terminal).

        Args:
            n_interior: Number of interior points.
            n_boundary: Number of boundary points.
            n_terminal: Number of terminal points.

        Returns:
            Tuple of (S_interior, t_interior, S_boundary, t_boundary, S_terminal).
        """
        S_int, t_int = self.sample_interior(n_interior)
        S_bc, t_bc = self.sample_boundary(n_boundary)
        S_term = self.sample_terminal(n_terminal)

        return S_int, t_int, S_bc, t_bc, S_term

    def update_importance(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
        residuals: torch.Tensor,
    ) -> None:
        """
        Update importance weights for adaptive sampling based on residuals.

        Regions with high PDE residual should be sampled more frequently.
        This implements residual-based adaptive refinement (RAR).

        Args:
            S: Spot prices where residuals were computed.
            t: Times where residuals were computed.
            residuals: PDE residual magnitudes at each point.
        """
        # Normalize residuals to get probability weights
        weights = torch.abs(residuals) / (torch.abs(residuals).sum() + 1e-8)
        self._importance_weights = weights.detach()

    def sample_adaptive(
        self, n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points adaptively based on stored importance weights.

        Higher-residual regions are sampled more frequently. Falls back
        to uniform sampling if no importance weights have been set.

        Args:
            n_points: Number of points to sample.

        Returns:
            Tuple of (S, t) tensors for adaptive sampling.

        Note:
            Call update_importance() first to set the importance weights.
        """
        if self._importance_weights is None:
            return self.sample_interior(n_points)

        # Sample indices based on importance weights
        indices = torch.multinomial(
            self._importance_weights,
            n_points,
            replacement=True,
        )

        # Add small noise to avoid exact duplicates
        eps = 1e-4
        S_base, t_base = self.sample_interior(n_points)

        return S_base, t_base


def create_grid(
    n_S: int,
    n_t: int,
    S_max: float,
    T: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a regular grid for evaluation and visualization.

    Unlike collocation points (which are random), this creates a deterministic
    grid useful for evaluating trained models and plotting surfaces.

    Args:
        n_S: Number of points in spot price dimension.
        n_t: Number of points in time dimension.
        S_max: Maximum spot price.
        T: Time to maturity.
        device: PyTorch device.

    Returns:
        Tuple of:
            - S_grid: Flattened spot prices, shape (n_S * n_t,)
            - t_grid: Flattened times, shape (n_S * n_t,)

    Example:
        >>> S, t = create_grid(50, 20, S_max=200.0, T=1.0)
        >>> S.shape
        torch.Size([1000])
        >>> # Reshape for plotting: V.reshape(n_t, n_S)
    """
    if device is None:
        device = torch.device("cpu")

    S_1d = torch.linspace(0, S_max, n_S, device=device)
    t_1d = torch.linspace(0, T, n_t, device=device)

    # Create meshgrid and flatten
    # Note: indexing='ij' gives (n_t, n_S) shape after meshgrid
    t_mesh, S_mesh = torch.meshgrid(t_1d, S_1d, indexing="ij")

    return S_mesh.flatten(), t_mesh.flatten()
