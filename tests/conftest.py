"""
Shared pytest fixtures for quantum-derivatives-trader tests.
"""

import pytest
import torch
import numpy as np

from src.pde.black_scholes import BSParams


@pytest.fixture
def bs_params() -> BSParams:
    """Common Black-Scholes parameters for testing."""
    return BSParams(
        r=0.05,      # 5% risk-free rate
        sigma=0.2,   # 20% volatility
        K=100.0,     # Strike price
        T=1.0,       # 1 year to maturity
    )


@pytest.fixture
def device() -> torch.device:
    """Default device for testing (CPU)."""
    return torch.device("cpu")


@pytest.fixture
def sample_spots(device: torch.device) -> torch.Tensor:
    """Sample spot prices for testing."""
    return torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0], device=device)


@pytest.fixture
def sample_times(device: torch.device) -> torch.Tensor:
    """Sample times for testing (all at t=0)."""
    return torch.zeros(5, device=device)


@pytest.fixture
def seed() -> int:
    """Random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(seed: int):
    """Set random seeds for reproducibility across all tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)
