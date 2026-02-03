"""Classical PINN implementation."""

from .pinn import PINN
from .losses import PINNLoss
from .networks import MLP, ResidualMLP

__all__ = ["PINN", "PINNLoss", "MLP", "ResidualMLP"]
