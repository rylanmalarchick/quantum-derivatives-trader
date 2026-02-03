"""
Quantum Amplitude Estimation for Monte Carlo pricing.

This provides quadratic speedup over classical MC: O(1/ε) vs O(1/ε²)

For now, we simulate this classically to study the algorithm.
Real quantum advantage requires fault-tolerant hardware.

Reference: Stamatopoulos et al., "Option Pricing using Quantum Computers"
"""

import pennylane as qml
import numpy as np
import torch
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class QAEResult:
    """Result of quantum amplitude estimation."""
    estimate: float
    confidence_interval: tuple[float, float]
    n_queries: int
    classical_equivalent_samples: int  # How many classical samples for same error


class QuantumMonteCarloEstimator:
    """
    Quantum Amplitude Estimation for option pricing.

    The algorithm:
    1. Encode the probability distribution of stock prices in quantum state
    2. Encode the payoff function as a rotation
    3. Use amplitude estimation to extract E[payoff]

    Key insight: QAE achieves O(1/N) error vs classical O(1/√N),
    meaning quadratically fewer queries for the same precision.
    """

    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.n_price_qubits = n_qubits - 1  # Reserve 1 for payoff encoding
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def encode_lognormal(
        self,
        mu: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode log-normal distribution (stock price at maturity) into amplitudes.

        Uses discretization: divide price range into 2^n bins and compute
        probability amplitudes for each bin.

        Args:
            mu: Mean of log(S_T)
            sigma: Std of log(S_T)

        Returns:
            (prices, amplitudes) arrays
        """
        n_bins = 2 ** self.n_price_qubits

        # Price range (truncated log-normal)
        S_min, S_max = 0.01, 500.0
        prices = np.linspace(S_min, S_max, n_bins)

        # Log-normal PDF
        from scipy.stats import lognorm
        pdf = lognorm.pdf(prices, s=sigma, scale=np.exp(mu))
        pdf = pdf / pdf.sum()  # Normalize to probability

        # Amplitudes are sqrt of probabilities
        amplitudes = np.sqrt(pdf)

        return prices, amplitudes

    def _create_state_prep_circuit(
        self,
        amplitudes: np.ndarray,
    ) -> Callable:
        """
        Create circuit to prepare quantum state with given amplitudes.

        Uses amplitude encoding to prepare |ψ⟩ = Σ α_i |i⟩
        where α_i are the probability amplitudes.
        """
        # Normalize amplitudes
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        @qml.qnode(self.dev, interface="torch")
        def state_prep():
            qml.AmplitudeEmbedding(
                amplitudes,
                wires=range(self.n_price_qubits),
                normalize=True,
            )
            return qml.state()

        return state_prep

    def _payoff_rotation(
        self,
        payoff_values: np.ndarray,
        max_payoff: float,
    ) -> np.ndarray:
        """
        Encode payoff as rotation angles.

        The amplitude of |1⟩ on ancilla qubit encodes E[payoff].
        """
        # Normalize payoffs to [0, 1]
        normalized = payoff_values / max_payoff

        # Convert to rotation angles
        # sin²(θ/2) = payoff, so θ = 2*arcsin(√payoff)
        angles = 2 * np.arcsin(np.sqrt(np.clip(normalized, 0, 1)))

        return angles

    def estimate_expectation(
        self,
        payoff_fn: Callable[[float], float],
        S0: float,
        r: float,
        sigma: float,
        T: float,
        n_iterations: int = 10,
    ) -> QAEResult:
        """
        Estimate E[e^{-rT} * payoff(S_T)] using amplitude estimation.

        This is a classical simulation of the quantum algorithm.
        On real quantum hardware, this would use phase estimation.

        Args:
            payoff_fn: Payoff function f(S) -> payoff
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_iterations: Number of Grover iterations (determines precision)

        Returns:
            QAEResult with estimate and confidence interval
        """
        # Risk-neutral distribution parameters
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        vol = sigma * np.sqrt(T)

        # Discretize distribution
        prices, amplitudes = self.encode_lognormal(mu, vol)

        # Compute payoffs
        payoffs = np.array([payoff_fn(p) for p in prices])
        discounted = np.exp(-r * T) * payoffs

        # Probabilities from amplitudes
        probs = amplitudes ** 2

        # True expectation (what QAE estimates)
        estimate = np.sum(probs * discounted)

        # Standard deviation for confidence interval
        variance = np.sum(probs * (discounted - estimate) ** 2)
        std = np.sqrt(variance)

        # QAE error scaling: O(1/M) where M = n_iterations
        # vs classical MC error: O(1/√N) where N = number of samples
        qae_error = std / n_iterations

        # Equivalent classical samples for same error
        # std/√N = std/M => N = M²
        classical_equivalent = n_iterations ** 2

        return QAEResult(
            estimate=estimate,
            confidence_interval=(estimate - 2 * qae_error, estimate + 2 * qae_error),
            n_queries=n_iterations,
            classical_equivalent_samples=classical_equivalent,
        )


class IterativeQAE:
    """
    Iterative Quantum Amplitude Estimation.

    More practical variant that doesn't require quantum phase estimation,
    using instead a sequence of Grover iterations with maximum likelihood.

    Reference: Grinko et al., "Iterative Quantum Amplitude Estimation"
    """

    def __init__(self, n_price_qubits: int = 6):
        self.n_price_qubits = n_price_qubits
        self.n_bins = 2 ** n_price_qubits

    def estimate(
        self,
        payoff_fn: Callable,
        distribution_params: dict,
        n_shots: int = 1000,
        n_rounds: int = 5,
    ) -> QAEResult:
        """
        Iterative amplitude estimation.

        Uses multiple rounds with increasing Grover depth,
        then combines results via maximum likelihood.
        """
        # This is a simplified simulation
        # Real implementation would use actual quantum circuits

        mu = distribution_params["mu"]
        sigma = distribution_params["sigma"]
        r = distribution_params["r"]
        T = distribution_params["T"]

        # Setup distribution
        from scipy.stats import lognorm
        prices = np.linspace(0.01, 500, self.n_bins)
        pdf = lognorm.pdf(prices, s=sigma, scale=np.exp(mu))
        pdf = pdf / pdf.sum()

        # Compute target amplitude
        payoffs = np.array([payoff_fn(p) for p in prices])
        discounted = np.exp(-r * T) * payoffs
        true_value = np.sum(pdf * discounted)

        # Simulate estimation with quantum-like precision
        # Each round provides more precision
        estimates = []
        for k in range(1, n_rounds + 1):
            # Grover depth increases each round
            grover_depth = 2**k - 1

            # Simulate measurement with appropriate variance
            # QAE variance scales as O(1/grover_depth²)
            variance = np.sum(pdf * (discounted - true_value)**2) / (grover_depth**2)
            estimate = true_value + np.random.normal(0, np.sqrt(variance / n_shots))
            estimates.append(estimate)

        # Combine estimates (simplified MLE)
        final_estimate = np.mean(estimates)
        error = np.std(estimates)

        total_queries = sum(2**k - 1 for k in range(1, n_rounds + 1)) * n_shots

        return QAEResult(
            estimate=final_estimate,
            confidence_interval=(final_estimate - 2*error, final_estimate + 2*error),
            n_queries=total_queries,
            classical_equivalent_samples=int(total_queries**2),
        )


def compare_quantum_classical(
    payoff_fn: Callable,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    K: float,
    n_classical_samples: int = 10000,
    n_quantum_iterations: int = 100,
) -> dict:
    """
    Compare quantum and classical Monte Carlo for option pricing.

    Returns comparison metrics showing the quantum advantage.
    """
    from scipy.stats import norm

    # Classical Monte Carlo
    Z = np.random.randn(n_classical_samples)
    S_T = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.array([payoff_fn(s) for s in S_T])
    classical_estimate = np.exp(-r*T) * np.mean(payoffs)
    classical_std = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_classical_samples)

    # Quantum amplitude estimation (simulated)
    qae = QuantumMonteCarloEstimator(n_qubits=8)
    qae_result = qae.estimate_expectation(
        payoff_fn, S0, r, sigma, T, n_iterations=n_quantum_iterations
    )

    return {
        "classical": {
            "estimate": classical_estimate,
            "std_error": classical_std,
            "n_samples": n_classical_samples,
        },
        "quantum": {
            "estimate": qae_result.estimate,
            "confidence_interval": qae_result.confidence_interval,
            "n_queries": qae_result.n_queries,
            "equivalent_classical_samples": qae_result.classical_equivalent_samples,
        },
        "speedup": qae_result.classical_equivalent_samples / n_classical_samples,
    }
