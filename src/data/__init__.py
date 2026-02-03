"""Data generation for PINN training."""

from .collocation import (
    generate_collocation_points,
    generate_latin_hypercube_points,
    CollocationSampler,
    create_grid,
)
from .synthetic import SyntheticOptionData

__all__ = [
    "generate_collocation_points",
    "generate_latin_hypercube_points",
    "CollocationSampler",
    "create_grid",
    "SyntheticOptionData",
]
