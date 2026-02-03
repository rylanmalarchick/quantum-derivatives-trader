"""
Variational Quantum Circuits for function approximation.

Replace the classical neural network in PINN with a parameterized quantum circuit.
This explores whether quantum expressivity provides advantages for PDE solutions.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def create_vqc(n_qubits: int, n_layers: int, dev: qml.device):
    """
    Create a variational quantum circuit for function approximation.

    Architecture: Hardware-efficient ansatz with trainable rotations and entanglement.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        dev: PennyLane device

    Returns:
        QNode circuit function
    """

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """
        Variational quantum circuit.

        Args:
            inputs: (2,) tensor [S, t] - normalized to [0, 2π]
            weights: (n_layers, n_qubits, 3) rotation parameters

        Returns:
            Expectation value ∈ [-1, 1]
        """
        # Encode inputs via angle embedding
        for i in range(n_qubits):
            qml.RY(inputs[i % 2], wires=i)  # Alternate S and t encoding

        # Variational layers
        for layer in range(n_layers):
            # Single-qubit rotations
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)

            # Entanglement (ring topology)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])

        # Measurement
        return qml.expval(qml.PauliZ(0))

    return circuit


def create_data_reuploading_circuit(n_qubits: int, n_layers: int, dev: qml.device):
    """
    Data re-uploading circuit.

    Interleaves data encoding with variational layers for increased expressivity.
    Reference: Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier"
    """

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            inputs: (2,) tensor [S, t]
            weights: (n_layers, n_qubits, 3) rotation parameters
        """
        for layer in range(n_layers):
            # Data encoding (re-upload each layer)
            for i in range(n_qubits):
                angle = inputs[i % 2] + weights[layer, i, 0]
                qml.RY(angle, wires=i)

            # Variational rotations
            for i in range(n_qubits):
                qml.RZ(weights[layer, i, 1], wires=i)
                qml.RX(weights[layer, i, 2], wires=i)

            # Entanglement
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    return circuit


class QuantumLayer(nn.Module):
    """
    Quantum layer that can be used in a hybrid model.

    Wraps a variational quantum circuit as a PyTorch module.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        circuit_type: str = "hardware_efficient",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Quantum device (use lightning for speed with adjoint diff)
        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        # Choose circuit type
        if circuit_type == "hardware_efficient":
            self.circuit = create_vqc(n_qubits, n_layers, self.dev)
        elif circuit_type == "data_reuploading":
            self.circuit = create_data_reuploading_circuit(n_qubits, n_layers, self.dev)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        # Trainable weights
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.

        Args:
            x: (batch, 2) normalized inputs [S, t]

        Returns:
            (batch,) circuit outputs in [-1, 1]
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            out = self.circuit(x[i], self.weights)
            outputs.append(out)

        return torch.stack(outputs)

    def get_circuit_info(self) -> dict:
        """Get information about the quantum circuit."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_parameters": self.weights.numel(),
            "device": str(self.dev),
        }


class MultiQubitMeasurement(nn.Module):
    """
    Quantum layer with multiple measurement outputs.

    Returns expectation values for multiple observables, allowing
    the quantum circuit to output more information per forward pass.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        n_outputs: int = 4,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_outputs = min(n_outputs, n_qubits)

        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            # Encoding
            for i in range(n_qubits):
                qml.RY(inputs[i % 2], wires=i)

            # Variational layers
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)

                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

            # Multiple measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_outputs)]

        self.circuit = circuit

        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) inputs

        Returns:
            (batch, n_outputs) measurement expectations
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            out = self.circuit(x[i], self.weights)
            outputs.append(torch.stack(out))

        return torch.stack(outputs)
