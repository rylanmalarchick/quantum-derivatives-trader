"""
Tests for quantum computing components.
"""

import pytest
import torch
import numpy as np

# Skip all tests if pennylane is not available
pennylane = pytest.importorskip("pennylane")

from src.quantum.variational import QuantumLayer, create_vqc
from src.quantum.hybrid_pinn import HybridPINN
from src.quantum.amplitude_estimation import QuantumMonteCarloEstimator, QAEResult


class TestQuantumLayer:
    """Tests for QuantumLayer module."""

    def test_quantum_layer_forward_shape(self):
        """QuantumLayer should output tensor of correct shape."""
        layer = QuantumLayer(n_qubits=4, n_layers=2)

        batch_size = 3
        x = torch.rand(batch_size, 2)  # (S, t) normalized

        output = layer(x)

        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()

    def test_quantum_layer_output_range(self):
        """QuantumLayer output should be in [-1, 1] (expectation value)."""
        layer = QuantumLayer(n_qubits=4, n_layers=2)

        x = torch.rand(5, 2)
        output = layer(x)

        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    @pytest.mark.parametrize("n_qubits", [2, 4, 6])
    def test_quantum_layer_different_qubit_counts(self, n_qubits):
        """QuantumLayer should work with different qubit counts."""
        layer = QuantumLayer(n_qubits=n_qubits, n_layers=2)

        x = torch.rand(2, 2)
        output = layer(x)

        assert output.shape == (2,)

    @pytest.mark.parametrize("n_layers", [1, 2, 4])
    def test_quantum_layer_different_layer_counts(self, n_layers):
        """QuantumLayer should work with different layer counts."""
        layer = QuantumLayer(n_qubits=4, n_layers=n_layers)

        x = torch.rand(2, 2)
        output = layer(x)

        assert output.shape == (2,)

    @pytest.mark.parametrize("circuit_type", ["hardware_efficient", "data_reuploading"])
    def test_quantum_layer_circuit_types(self, circuit_type):
        """QuantumLayer should support different circuit types."""
        layer = QuantumLayer(n_qubits=4, n_layers=2, circuit_type=circuit_type)

        x = torch.rand(2, 2)
        output = layer(x)

        assert output.shape == (2,)

    def test_quantum_layer_get_circuit_info(self):
        """get_circuit_info should return expected information."""
        n_qubits = 4
        n_layers = 3
        layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)

        info = layer.get_circuit_info()

        assert info["n_qubits"] == n_qubits
        assert info["n_layers"] == n_layers
        assert info["n_parameters"] == n_layers * n_qubits * 3


class TestCreateVQC:
    """Tests for create_vqc function."""

    def test_create_vqc_returns_callable(self):
        """create_vqc should return a callable QNode."""
        dev = pennylane.device("default.qubit", wires=4)
        circuit = create_vqc(n_qubits=4, n_layers=2, dev=dev)

        assert callable(circuit)

    def test_vqc_returns_scalar(self):
        """VQC should return a scalar expectation value."""
        dev = pennylane.device("default.qubit", wires=4)
        circuit = create_vqc(n_qubits=4, n_layers=2, dev=dev)

        inputs = torch.tensor([0.5, 0.3])
        weights = torch.randn(2, 4, 3)

        result = circuit(inputs, weights)

        # Should be a scalar (or 0-d tensor)
        assert result.numel() == 1

    def test_vqc_expectation_range(self):
        """VQC expectation should be in [-1, 1]."""
        dev = pennylane.device("default.qubit", wires=4)
        circuit = create_vqc(n_qubits=4, n_layers=2, dev=dev)

        inputs = torch.tensor([1.0, 2.0])
        weights = torch.randn(2, 4, 3)

        result = circuit(inputs, weights)

        assert -1.0 <= result.item() <= 1.0


class TestHybridPINN:
    """Tests for HybridPINN model."""

    def test_hybrid_pinn_forward_shape(self):
        """HybridPINN should output tensor of correct shape."""
        model = HybridPINN(
            n_qubits=4,
            n_layers=2,
            classical_hidden=16,
            S_max=200.0,
            T_max=1.0,
        )

        batch_size = 3
        S = torch.rand(batch_size) * 200
        t = torch.rand(batch_size)

        V = model(S, t)

        assert V.shape == (batch_size,)
        assert not torch.isnan(V).any()

    def test_hybrid_pinn_positive_output(self):
        """HybridPINN output should be non-negative (option values)."""
        model = HybridPINN(
            n_qubits=4,
            n_layers=2,
            classical_hidden=16,
            S_max=200.0,
            T_max=1.0,
        )

        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.tensor([0.0, 0.5, 0.9])

        V = model(S, t)

        assert torch.all(V >= 0)

    def test_hybrid_pinn_handles_edge_cases(self):
        """HybridPINN should handle S=0 and t=T."""
        model = HybridPINN(
            n_qubits=4,
            n_layers=2,
            classical_hidden=16,
            S_max=200.0,
            T_max=1.0,
        )

        # S = 0
        S = torch.tensor([0.0, 0.0])
        t = torch.tensor([0.0, 0.5])

        V = model(S, t)

        assert V.shape == (2,)
        assert not torch.isnan(V).any()

    def test_hybrid_pinn_gradient_flow(self):
        """Gradients should flow through hybrid model."""
        model = HybridPINN(
            n_qubits=4,
            n_layers=2,
            classical_hidden=16,
            S_max=200.0,
            T_max=1.0,
        )

        S = torch.tensor([100.0], requires_grad=True)
        t = torch.tensor([0.5], requires_grad=True)

        V = model(S, t)
        V.sum().backward()

        # Check quantum layer has gradients
        assert model.quantum.weights.grad is not None


class TestQuantumMonteCarloEstimator:
    """Tests for QuantumMonteCarloEstimator."""

    def test_qmc_output_structure(self):
        """QMC estimate should return QAEResult with correct fields."""
        qmc = QuantumMonteCarloEstimator(n_qubits=6)

        # European call payoff
        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        result = qmc.estimate_expectation(
            payoff_fn=payoff_fn,
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_iterations=10,
        )

        assert isinstance(result, QAEResult)
        assert hasattr(result, "estimate")
        assert hasattr(result, "confidence_interval")
        assert hasattr(result, "n_queries")
        assert hasattr(result, "classical_equivalent_samples")

    def test_qmc_estimate_positive_for_atm_call(self):
        """ATM call should have positive estimated value."""
        qmc = QuantumMonteCarloEstimator(n_qubits=6)

        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        result = qmc.estimate_expectation(
            payoff_fn=payoff_fn,
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_iterations=10,
        )

        assert result.estimate > 0

    def test_qmc_confidence_interval_contains_estimate(self):
        """Confidence interval should contain the estimate."""
        qmc = QuantumMonteCarloEstimator(n_qubits=6)

        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        result = qmc.estimate_expectation(
            payoff_fn=payoff_fn,
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_iterations=10,
        )

        assert result.confidence_interval[0] <= result.estimate <= result.confidence_interval[1]

    def test_qmc_quadratic_speedup(self):
        """Classical equivalent samples should be n_iterations^2."""
        qmc = QuantumMonteCarloEstimator(n_qubits=6)

        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)
        n_iterations = 20

        result = qmc.estimate_expectation(
            payoff_fn=payoff_fn,
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            n_iterations=n_iterations,
        )

        assert result.classical_equivalent_samples == n_iterations ** 2

    def test_encode_lognormal_returns_valid_distribution(self):
        """encode_lognormal should return normalized amplitudes."""
        qmc = QuantumMonteCarloEstimator(n_qubits=6)

        prices, amplitudes = qmc.encode_lognormal(mu=4.6, sigma=0.2)

        # Amplitudes squared should sum to 1 (normalized probability)
        prob_sum = np.sum(amplitudes ** 2)
        np.testing.assert_allclose(prob_sum, 1.0, rtol=1e-6)

        # All amplitudes should be real and non-negative
        assert np.all(amplitudes >= 0)


class TestQuantumGradients:
    """Tests for gradient computation through quantum circuits."""

    def test_quantum_layer_trainable(self):
        """QuantumLayer weights should be trainable."""
        layer = QuantumLayer(n_qubits=4, n_layers=2)

        x = torch.rand(2, 2)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert layer.weights.grad is not None
        assert not torch.all(layer.weights.grad == 0)

    def test_hybrid_pinn_trainable(self):
        """All HybridPINN parameters should be trainable."""
        model = HybridPINN(
            n_qubits=4,
            n_layers=2,
            classical_hidden=16,
            S_max=200.0,
            T_max=1.0,
        )

        S = torch.rand(3) * 200
        t = torch.rand(3)

        V = model(S, t)
        loss = V.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
