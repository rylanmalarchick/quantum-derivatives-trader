"""
Hybrid Quantum-Classical PINN.

Architecture:
    Classical preprocessing → Quantum circuit → Classical postprocessing

The quantum circuit handles the nonlinear function approximation,
while classical layers handle input normalization and output scaling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .variational import QuantumLayer, MultiQubitMeasurement


class HybridPINN(nn.Module):
    """
    Hybrid quantum-classical PINN for option pricing.

    Input: (S, t)
    Output: V (option value)

    Architecture:
        1. Classical preprocessing (normalization, feature expansion)
        2. Quantum circuit (variational quantum layer)
        3. Classical postprocessing (output scaling)
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        classical_hidden: int = 32,
        S_max: float = 200.0,
        T_max: float = 1.0,
        circuit_type: str = "hardware_efficient",
    ):
        super().__init__()

        self.S_max = S_max
        self.T_max = T_max

        # Classical preprocessing: normalize and expand features
        self.pre_net = nn.Sequential(
            nn.Linear(2, classical_hidden),
            nn.Tanh(),
            nn.Linear(classical_hidden, 2),  # Back to 2D for quantum encoding
            nn.Tanh(),
        )

        # Quantum layer
        self.quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
        )

        # Classical postprocessing: scale output to option value range
        self.post_net = nn.Sequential(
            nn.Linear(1, classical_hidden),
            nn.ReLU(),
            nn.Linear(classical_hidden, 1),
            nn.Softplus(),  # Option values are positive
        )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.

        Args:
            S: Spot price tensor
            t: Time tensor

        Returns:
            Option value tensor
        """
        # Ensure 1D
        S = S.view(-1)
        t = t.view(-1)

        # Normalize inputs
        S_norm = S / self.S_max  # [0, 1]
        t_norm = t / self.T_max  # [0, 1]

        x = torch.stack([S_norm, t_norm], dim=-1)

        # Classical preprocessing
        x = self.pre_net(x)

        # Scale to [0, 2π] for quantum encoding
        x = x * np.pi

        # Quantum circuit
        q_out = self.quantum(x)  # (batch,)

        # Classical postprocessing
        V = self.post_net(q_out.unsqueeze(-1)).squeeze(-1)

        # Scale by S for correct magnitude
        V = V * S

        return V


class DeepHybridPINN(nn.Module):
    """
    Deeper hybrid architecture with multiple quantum layers.

    Uses quantum circuits at multiple stages of processing,
    interleaved with classical layers.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_quantum_layers: int = 2,
        layers_per_quantum: int = 2,
        classical_hidden: int = 16,
        S_max: float = 200.0,
        T_max: float = 1.0,
    ):
        super().__init__()

        self.S_max = S_max
        self.T_max = T_max

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(2, classical_hidden),
            nn.Tanh(),
        )

        # Interleaved quantum-classical layers
        self.layers = nn.ModuleList()
        for i in range(n_quantum_layers):
            # Classical -> Quantum -> Classical
            self.layers.append(nn.Linear(classical_hidden, 2))
            self.layers.append(QuantumLayer(n_qubits, layers_per_quantum))
            self.layers.append(nn.Linear(1, classical_hidden))
            self.layers.append(nn.Tanh())

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(classical_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        S = S.view(-1)
        t = t.view(-1)

        # Normalize
        x = torch.stack([S / self.S_max, t / self.T_max], dim=-1)

        # Input embedding
        x = self.input_embed(x)

        # Interleaved layers
        i = 0
        while i < len(self.layers):
            # Classical to 2D
            x = self.layers[i](x)
            x = x * np.pi  # Scale for quantum

            # Quantum
            x = self.layers[i + 1](x)

            # Back to classical hidden
            x = self.layers[i + 2](x.unsqueeze(-1))
            x = self.layers[i + 3](x)

            i += 4

        # Output
        V = self.output(x).squeeze(-1)
        V = V * S

        return V


class QuantumResidualPINN(nn.Module):
    """
    Quantum circuit as a residual correction to classical PINN.

    V(S, t) = V_classical(S, t) + α * V_quantum(S, t)

    This allows the quantum circuit to learn corrections to
    the classical solution, which may be easier to train.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        classical_hidden: list[int] = [32, 32],
        S_max: float = 200.0,
        T_max: float = 1.0,
        alpha: float = 0.1,
    ):
        super().__init__()

        self.S_max = S_max
        self.T_max = T_max
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Classical branch
        layers = []
        in_dim = 2
        for h in classical_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classical = nn.Sequential(*layers)

        # Quantum branch
        self.quantum_pre = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
        )
        self.quantum = QuantumLayer(n_qubits, n_layers)
        self.quantum_post = nn.Sequential(
            nn.Linear(1, 1),
        )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        S = S.view(-1)
        t = t.view(-1)

        # Normalize
        x = torch.stack([S / self.S_max, t / self.T_max], dim=-1)

        # Classical prediction
        V_classical = self.classical(x).squeeze(-1)

        # Quantum correction
        x_q = self.quantum_pre(x) * np.pi
        q_out = self.quantum(x_q)
        V_quantum = self.quantum_post(q_out.unsqueeze(-1)).squeeze(-1)

        # Combined output
        V = (V_classical + self.alpha * V_quantum) * S

        return torch.relu(V)  # Option values are non-negative
