"""
Neural network architectures for PINNs.

Key design considerations:
1. Smooth activations (Tanh, Swish) for accurate gradient computation
2. Residual connections for deeper networks
3. Fourier features for high-frequency components
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for PINNs.

    Uses Tanh activation for smoothness (needed for PDE residual computation).
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dims: list[int] = [64, 64, 64, 64],
        activation: str = "tanh",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Build layers
        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "swish":
                layers.append(nn.SiLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier for tanh)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim),
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class ResidualMLP(nn.Module):
    """
    MLP with residual connections.

    Better gradient flow for deeper networks.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)


class FourierFeaturesMLP(nn.Module):
    """
    MLP with Fourier feature encoding.

    Helps capture high-frequency components in the solution.
    Reference: Tancik et al., "Fourier Features Let Networks Learn
               High Frequency Functions in Low Dimensional Domains"
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dims: list[int] = [128, 128, 128, 128],
        n_frequencies: int = 32,
        sigma: float = 1.0,
    ):
        super().__init__()

        self.n_frequencies = n_frequencies

        # Random Fourier features matrix (fixed, not trained)
        B = torch.randn(in_dim, n_frequencies) * sigma
        self.register_buffer("B", B)

        # MLP on Fourier features
        # Input dim is 2 * n_frequencies (sin and cos)
        self.mlp = MLP(
            in_dim=2 * n_frequencies,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier feature encoding
        x_proj = 2 * np.pi * x @ self.B
        features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return self.mlp(features)


class AdaptiveActivation(nn.Module):
    """
    Learnable activation function.

    a * tanh(x) + b * x, where a and b are learnable.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * torch.tanh(x) + self.b * x
