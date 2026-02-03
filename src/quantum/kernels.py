"""
Quantum Kernel Methods for Option Pricing.

This module implements quantum kernel methods for option pricing using
the quantum kernel trick: k(x, x') = |⟨φ(x)|φ(x')⟩|², where |φ(x)⟩ is a
quantum feature map implemented via parameterized quantum circuits.

Quantum kernels can access feature spaces that are exponentially large in
the number of qubits, potentially providing advantages for learning complex
functions like option pricing surfaces.

References:
    - Schuld, M., & Killoran, N. (2019). "Quantum Machine Learning in Feature 
      Hilbert Spaces." Physical Review Letters, 122(4), 040504.
      arXiv:1803.07128
    
    - Havlíček, V., et al. (2019). "Supervised learning with quantum-enhanced 
      feature spaces." Nature, 567(7747), 209-212.
      arXiv:1804.11326
    
    - Huang, H.-Y., et al. (2021). "Power of data in quantum machine learning."
      Nature Communications, 12, 2631.
      arXiv:2011.01938

Author: Quantum Derivatives Trader Team
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Callable, Literal
from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class FeatureMapType(str, Enum):
    """Supported quantum feature map types."""
    
    ZZ = "zz"
    IQP = "iqp"
    PAULI = "pauli"
    AMPLITUDE = "amplitude"


@dataclass
class KernelConfig:
    """Configuration for quantum kernel computation.
    
    Attributes:
        n_qubits: Number of qubits in the quantum circuit.
        n_layers: Number of repetitions of the feature map.
        feature_map: Type of quantum feature map to use.
        use_adjoint: If True, use adjoint method for overlap computation.
    """
    
    n_qubits: int = 4
    n_layers: int = 2
    feature_map: FeatureMapType = FeatureMapType.ZZ
    use_adjoint: bool = True


class QuantumKernel:
    """
    Quantum kernel for machine learning with quantum feature maps.
    
    Implements the quantum kernel trick where the kernel value between two
    data points is computed as the squared overlap of their quantum feature
    map states:
    
        k(x, x') = |⟨φ(x)|φ(x')⟩|²
    
    where |φ(x)⟩ = U(x)|0⟩ is the quantum state prepared by applying the
    feature map circuit U(x) to the ground state.
    
    The kernel is computed via the overlap circuit:
        1. Prepare |φ(x)⟩ by applying U(x)
        2. Apply U†(x') (adjoint of feature map)
        3. Measure probability of |0...0⟩ state
    
    This gives k(x, x') = |⟨0|U†(x')U(x)|0⟩|² = |⟨φ(x')|φ(x)⟩|²
    
    Attributes:
        n_qubits: Number of qubits in the feature map circuit.
        n_layers: Number of feature map layers (for increased expressivity).
        feature_map_type: Type of feature map used.
        device: PennyLane quantum device.
    
    Example:
        >>> kernel = QuantumKernel(n_qubits=4, feature_map="zz")
        >>> x1 = np.array([0.5, 0.3])
        >>> x2 = np.array([0.6, 0.35])
        >>> k_val = kernel.kernel(x1, x2)
        >>> print(f"Kernel value: {k_val:.4f}")
        
        >>> # Compute full kernel matrix
        >>> X = np.random.randn(10, 2)
        >>> K = kernel.kernel_matrix(X)
        >>> assert K.shape == (10, 10)
        >>> assert np.allclose(K, K.T)  # Symmetric
        >>> assert np.all(np.diag(K) == 1.0)  # Normalized
    
    References:
        Schuld, M., & Killoran, N. (2019). Physical Review Letters, 122, 040504.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: Literal["zz", "iqp", "pauli", "amplitude"] = "zz",
        device_name: str = "default.qubit",
    ) -> None:
        """
        Initialize the quantum kernel.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit. More qubits
                allow encoding more features and accessing larger Hilbert spaces.
            n_layers: Number of repetitions of the feature map. More layers
                can increase expressivity but also circuit depth.
            feature_map: Type of quantum feature map:
                - "zz": ZZ feature map with pairwise ZZ interactions.
                        Good for capturing correlations between features.
                - "iqp": IQP (Instantaneous Quantum Polynomial) feature map.
                        Creates highly entangled states, classically hard to simulate.
                - "pauli": General Pauli feature map with all single-qubit Paulis.
                - "amplitude": Amplitude encoding (requires 2^n_qubits features).
            device_name: PennyLane device to use. Options include:
                - "default.qubit": Default CPU simulator
                - "lightning.qubit": Fast C++ simulator
                - "lightning.gpu": GPU-accelerated simulator
        
        Raises:
            ValueError: If feature_map is not a recognized type.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = FeatureMapType(feature_map)
        self.device_name = device_name
        
        # Create quantum device
        self.device = qml.device(device_name, wires=n_qubits)
        
        # Select feature map function
        self._feature_map_fn = self._get_feature_map_fn()
        
        # Create the kernel circuit
        self._kernel_circuit = self._create_kernel_circuit()
    
    def _get_feature_map_fn(self) -> Callable[[np.ndarray], None]:
        """Get the feature map function based on type."""
        feature_maps = {
            FeatureMapType.ZZ: self._zz_feature_map,
            FeatureMapType.IQP: self._iqp_feature_map,
            FeatureMapType.PAULI: self._pauli_feature_map,
            FeatureMapType.AMPLITUDE: self._amplitude_feature_map,
        }
        return feature_maps[self.feature_map_type]
    
    def _zz_feature_map(self, x: np.ndarray) -> None:
        """
        ZZ feature map with pairwise ZZ interactions.
        
        This feature map creates entanglement through ZZ(θ) gates where
        θ depends on products of input features. The circuit structure is:
        
        Layer l:
            1. Hadamard on all qubits
            2. RZ(2*x_i) on qubit i for each feature
            3. ZZ interaction: CNOT-RZ-CNOT with angle (π - x_i)(π - x_j)
        
        The ZZ interactions encode correlations between features into the
        quantum state, making this feature map particularly suitable for
        learning functions that depend on feature interactions.
        
        Args:
            x: Input feature vector. Features are mapped cyclically to qubits
               if len(x) != n_qubits.
        
        References:
            Havlíček et al. (2019). Nature, 567, 209-212.
        """
        n_features = min(len(x), self.n_qubits)
        
        for layer in range(self.n_layers):
            # Hadamard layer creates superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Single-qubit Z rotations encode individual features
            for i in range(self.n_qubits):
                feature_idx = i % n_features
                qml.RZ(2.0 * x[feature_idx], wires=i)
            
            # ZZ interactions encode feature correlations
            # ZZ(θ) = exp(-i θ Z⊗Z / 2)
            for i in range(self.n_qubits - 1):
                j = i + 1
                # Use cyclic feature indices
                fi, fj = i % n_features, j % n_features
                
                # Angle based on (π - x_i)(π - x_j) for better expressivity
                theta = 2.0 * (np.pi - x[fi]) * (np.pi - x[fj])
                
                # Implement ZZ rotation via CNOT-RZ-CNOT
                qml.CNOT(wires=[i, j])
                qml.RZ(theta, wires=j)
                qml.CNOT(wires=[i, j])
    
    def _iqp_feature_map(self, x: np.ndarray) -> None:
        """
        IQP (Instantaneous Quantum Polynomial) feature map.
        
        IQP circuits consist of diagonal gates (in the computational basis)
        sandwiched between Hadamard layers. They are believed to be hard to
        classically simulate, suggesting potential quantum advantage.
        
        Structure:
            H^⊗n → Diagonal(x) → H^⊗n → Diagonal(x) → ...
        
        where Diagonal(x) includes both single-qubit Z rotations and
        two-qubit ZZ interactions parameterized by the data.
        
        Args:
            x: Input feature vector.
        
        References:
            Bremner et al. (2016). "Average-case complexity versus approximate
            simulation of commuting quantum computations." PRL 117, 080501.
        """
        n_features = min(len(x), self.n_qubits)
        
        for layer in range(self.n_layers):
            # Hadamard layer
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Diagonal gates: Z rotations
            for i in range(self.n_qubits):
                qml.RZ(x[i % n_features], wires=i)
            
            # Diagonal gates: ZZ interactions
            for i in range(self.n_qubits - 1):
                fi, fj = i % n_features, (i + 1) % n_features
                theta = x[fi] * x[fj]
                
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(theta, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
    
    def _pauli_feature_map(self, x: np.ndarray) -> None:
        """
        General Pauli feature map with X, Y, Z rotations.
        
        This feature map uses all three Pauli rotations for richer encoding.
        Each layer applies RX, RY, RZ rotations with different feature
        combinations, followed by an entangling layer.
        
        Args:
            x: Input feature vector.
        """
        n_features = min(len(x), self.n_qubits)
        
        for layer in range(self.n_layers):
            # Rotation layer with all Paulis
            for i in range(self.n_qubits):
                fi = i % n_features
                qml.RX(x[fi] * np.pi, wires=i)
                qml.RY(x[fi] * np.pi / 2, wires=i)
                qml.RZ(x[fi] * np.pi, wires=i)
            
            # Entangling layer with CZ gates
            for i in range(self.n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            
            # Ring connection
            if self.n_qubits > 2:
                qml.CZ(wires=[self.n_qubits - 1, 0])
    
    def _amplitude_feature_map(self, x: np.ndarray) -> None:
        """
        Amplitude encoding feature map.
        
        Encodes the (normalized) input vector directly into the amplitudes
        of the quantum state. This requires len(x) = 2^n_qubits.
        
        For smaller inputs, pads with zeros. For larger inputs, truncates.
        
        Warning: This encoding requires O(2^n) classical preprocessing and
        a complex circuit decomposition. Use for small problems only.
        
        Args:
            x: Input feature vector.
        """
        dim = 2 ** self.n_qubits
        
        # Pad or truncate to correct dimension
        if len(x) < dim:
            x_padded = np.zeros(dim)
            x_padded[:len(x)] = x
        else:
            x_padded = x[:dim]
        
        # Normalize
        norm = np.linalg.norm(x_padded)
        if norm > 1e-10:
            x_normalized = x_padded / norm
        else:
            x_normalized = x_padded
            x_normalized[0] = 1.0  # Avoid zero state
        
        # Use PennyLane's amplitude encoding
        qml.AmplitudeEmbedding(x_normalized, wires=range(self.n_qubits), 
                               normalize=True)
    
    def _create_kernel_circuit(self) -> qml.QNode:
        """
        Create the quantum circuit for kernel computation.
        
        The kernel k(x, x') = |⟨φ(x)|φ(x')⟩|² is computed by:
            1. Apply U(x) to prepare |φ(x)⟩
            2. Apply U†(x') to compute overlap with |φ(x')⟩
            3. Measure probability of |0...0⟩
        
        Returns:
            QNode computing the kernel value.
        """
        @qml.qnode(self.device)
        def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> float:
            # Prepare |φ(x1)⟩
            self._feature_map_fn(x1)
            
            # Apply U†(x2) using adjoint
            qml.adjoint(lambda: self._feature_map_fn(x2))()
            
            # Return probabilities of all computational basis states
            return qml.probs(wires=range(self.n_qubits))
        
        return kernel_circuit
    
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the quantum kernel between two data points.
        
        The kernel value is k(x1, x2) = |⟨φ(x1)|φ(x2)⟩|², representing
        the fidelity between the two quantum states prepared by the
        feature map.
        
        Properties:
            - k(x, x) = 1 (normalized)
            - k(x, x') = k(x', x) (symmetric)
            - 0 ≤ k(x, x') ≤ 1 (bounded)
            - K matrix is positive semi-definite (valid kernel)
        
        Args:
            x1: First input vector of shape (n_features,).
            x2: Second input vector of shape (n_features,).
        
        Returns:
            Kernel value in [0, 1].
        
        Example:
            >>> kernel = QuantumKernel(n_qubits=4)
            >>> k = kernel.kernel(np.array([0.1, 0.2]), np.array([0.1, 0.2]))
            >>> assert np.isclose(k, 1.0)  # Same point
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        
        probs = self._kernel_circuit(x1, x2)
        
        # Kernel value is probability of measuring |0...0⟩
        return float(probs[0])
    
    def kernel_matrix(
        self, 
        X1: np.ndarray, 
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the kernel matrix between two sets of data points.
        
        If X2 is None, computes the Gram matrix K[i,j] = k(X1[i], X1[j]).
        Otherwise, computes K[i,j] = k(X1[i], X2[j]).
        
        The Gram matrix is guaranteed to be symmetric positive semi-definite,
        making it suitable for use with kernel methods like SVM and KRR.
        
        Args:
            X1: First set of data points, shape (n_samples_1, n_features).
            X2: Optional second set of data points, shape (n_samples_2, n_features).
                If None, computes the Gram matrix of X1 with itself.
        
        Returns:
            Kernel matrix of shape (n_samples_1, n_samples_2) or 
            (n_samples_1, n_samples_1) if X2 is None.
        
        Example:
            >>> kernel = QuantumKernel(n_qubits=4)
            >>> X = np.random.randn(20, 2)
            >>> K = kernel.kernel_matrix(X)
            >>> # Verify positive semi-definite
            >>> eigvals = np.linalg.eigvalsh(K)
            >>> assert np.all(eigvals >= -1e-10)
        """
        X1 = np.asarray(X1, dtype=np.float64)
        
        if X2 is None:
            # Gram matrix (symmetric)
            n = len(X1)
            K = np.zeros((n, n))
            
            for i in range(n):
                K[i, i] = 1.0  # Diagonal is always 1
                for j in range(i + 1, n):
                    k_ij = self.kernel(X1[i], X1[j])
                    K[i, j] = k_ij
                    K[j, i] = k_ij
            
            return K
        else:
            # Cross kernel matrix
            X2 = np.asarray(X2, dtype=np.float64)
            n1, n2 = len(X1), len(X2)
            K = np.zeros((n1, n2))
            
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self.kernel(X1[i], X2[j])
            
            return K
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of the quantum feature space.
        
        For n qubits, the feature space is the 2^n dimensional Hilbert space.
        This exponential scaling is what provides potential quantum advantage.
        
        Returns:
            Dimension of the feature Hilbert space (2^n_qubits).
        """
        return 2 ** self.n_qubits
    
    def __repr__(self) -> str:
        return (
            f"QuantumKernel(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"feature_map='{self.feature_map_type.value}', "
            f"feature_dim={self.get_feature_dimension()})"
        )


class QuantumKernelRegression:
    """
    Kernel Ridge Regression using Quantum Kernels.
    
    Implements kernel ridge regression (KRR) with a quantum kernel:
        
        f(x) = Σᵢ αᵢ k(xᵢ, x)
    
    where the coefficients α are found by solving:
        
        α = (K + λI)⁻¹ y
    
    Here K is the quantum kernel matrix and λ is the regularization parameter.
    
    This approach combines:
    - Quantum kernel: Potential access to exponentially large feature spaces
    - Classical optimization: Closed-form solution, no barren plateaus
    
    The quantum advantage, if any, comes from the kernel evaluation, not the
    training procedure which remains classical.
    
    Attributes:
        kernel: The QuantumKernel used for computing similarities.
        alpha: Regularization parameter (ridge penalty).
        coefficients: Learned regression coefficients after fitting.
        X_train: Training data stored for prediction.
    
    Example:
        >>> # Generate synthetic option pricing data
        >>> X_train = np.random.rand(100, 2)  # (S, t) normalized
        >>> y_train = X_train[:, 0] - X_train[:, 1]  # Simplified pricing
        >>> 
        >>> # Fit quantum kernel regression
        >>> qkr = QuantumKernelRegression(n_qubits=4, alpha=0.01)
        >>> qkr.fit(X_train, y_train)
        >>> 
        >>> # Predict on test data
        >>> X_test = np.random.rand(20, 2)
        >>> y_pred = qkr.predict(X_test)
    
    References:
        Schuld, M. (2021). "Supervised quantum machine learning models are 
        kernel methods." arXiv:2101.11020.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        feature_map: Literal["zz", "iqp", "pauli", "amplitude"] = "zz",
        alpha: float = 1e-3,
        kernel: Optional[QuantumKernel] = None,
    ) -> None:
        """
        Initialize quantum kernel regression.
        
        Args:
            n_qubits: Number of qubits for the quantum kernel.
            n_layers: Number of feature map layers.
            feature_map: Type of quantum feature map.
            alpha: Regularization parameter (ridge penalty). Larger values
                give more regularization, preventing overfitting but potentially
                underfitting. Typical range: 1e-4 to 1e-1.
            kernel: Optional pre-configured QuantumKernel. If provided,
                n_qubits, n_layers, and feature_map are ignored.
        """
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = QuantumKernel(
                n_qubits=n_qubits,
                n_layers=n_layers,
                feature_map=feature_map,
            )
        
        self.alpha = alpha
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.coefficients: Optional[np.ndarray] = None
        self._K_train: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelRegression":
        """
        Fit the kernel ridge regression model.
        
        Solves the ridge regression problem:
            α = (K + λI)⁻¹ y
        
        where K is the quantum kernel Gram matrix.
        
        Args:
            X: Training inputs of shape (n_samples, n_features).
               For option pricing, typically n_features=2 for (S, t).
            y: Training targets of shape (n_samples,).
               For option pricing, these are option values.
        
        Returns:
            self, for method chaining.
        
        Raises:
            ValueError: If X and y have incompatible shapes.
        
        Note:
            The kernel matrix computation has O(n²) complexity in the number
            of training samples, and each kernel evaluation requires running
            a quantum circuit. For large datasets, consider using subset
            methods or Nyström approximation.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples, "
                f"got {len(X)} and {len(y)}"
            )
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute kernel Gram matrix
        self._K_train = self.kernel.kernel_matrix(X)
        
        # Solve ridge regression: (K + αI)β = y
        n = len(X)
        regularized_K = self._K_train + self.alpha * np.eye(n)
        
        # Use Cholesky for numerical stability (K is PSD)
        try:
            L = np.linalg.cholesky(regularized_K)
            self.coefficients = np.linalg.solve(
                L.T, np.linalg.solve(L, y)
            )
        except np.linalg.LinAlgError:
            # Fall back to standard solve if Cholesky fails
            self.coefficients = np.linalg.solve(regularized_K, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new inputs.
        
        Computes:
            f(x) = Σᵢ αᵢ k(xᵢ, x)
        
        where xᵢ are training points and αᵢ are learned coefficients.
        
        Args:
            X: Test inputs of shape (n_samples, n_features) or (n_features,).
        
        Returns:
            Predicted values of shape (n_samples,).
        
        Raises:
            ValueError: If the model has not been fitted.
        
        Note:
            Prediction requires O(n_train) kernel evaluations per test point,
            each requiring a quantum circuit execution.
        """
        if self.X_train is None or self.coefficients is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Compute kernel between test and training points
        K_test = self.kernel.kernel_matrix(X, self.X_train)
        
        # Predict: y = K_test @ coefficients
        return K_test @ self.coefficients
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score on test data.
        
        R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
        
        Args:
            X: Test inputs of shape (n_samples, n_features).
            y: True target values of shape (n_samples,).
        
        Returns:
            R² score in (-∞, 1]. Score of 1 is perfect prediction.
        """
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot < 1e-10:
            return 1.0 if ss_res < 1e-10 else 0.0
        
        return 1.0 - ss_res / ss_tot
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """Get the learned regression coefficients."""
        return self.coefficients.copy() if self.coefficients is not None else None
    
    def __repr__(self) -> str:
        fitted = self.coefficients is not None
        n_train = len(self.X_train) if self.X_train is not None else 0
        return (
            f"QuantumKernelRegression(kernel={self.kernel}, alpha={self.alpha}, "
            f"fitted={fitted}, n_train={n_train})"
        )


class QuantumKernelPricer:
    """
    Option Pricing using Quantum Kernel Regression.
    
    This class wraps QuantumKernelRegression for the specific task of option
    pricing, providing a convenient interface for financial applications.
    
    The model learns the option pricing function V(S, t) from training data,
    where:
        - S: Spot price of the underlying asset
        - t: Time to maturity
        - V: Option value
    
    Input normalization is handled automatically to ensure features are in
    an appropriate range for the quantum encoding.
    
    Workflow:
        1. Initialize with market parameters (S_max, T_max for normalization)
        2. Fit on training data: (S, t) -> V pairs
        3. Predict option values for new (S, t) pairs
    
    Attributes:
        regressor: The underlying QuantumKernelRegression model.
        S_max: Maximum spot price for normalization.
        T_max: Maximum time to maturity for normalization.
        is_fitted: Whether the model has been trained.
    
    Example:
        >>> # Create pricer
        >>> pricer = QuantumKernelPricer(n_qubits=6, S_max=200.0, T_max=1.0)
        >>> 
        >>> # Training data (e.g., from Monte Carlo simulation)
        >>> S_train = np.random.uniform(50, 150, 100)
        >>> t_train = np.random.uniform(0.1, 1.0, 100)
        >>> V_train = compute_option_values(S_train, t_train)  # Your pricing model
        >>> 
        >>> # Fit the quantum kernel pricer
        >>> pricer.fit(S_train, t_train, V_train)
        >>> 
        >>> # Price new options
        >>> S_test = np.array([100.0, 110.0, 90.0])
        >>> t_test = np.array([0.5, 0.5, 0.5])
        >>> prices = pricer.predict(S_test, t_test)
    
    Note:
        This is primarily intended for interpolation within the training data
        range. Extrapolation to very different (S, t) values may be unreliable.
    """
    
    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        feature_map: Literal["zz", "iqp", "pauli", "amplitude"] = "zz",
        alpha: float = 1e-3,
        S_max: float = 200.0,
        T_max: float = 1.0,
        include_moneyness: bool = True,
    ) -> None:
        """
        Initialize the quantum kernel option pricer.
        
        Args:
            n_qubits: Number of qubits for the quantum kernel. More qubits
                can capture more complex pricing surfaces but increase
                computation time.
            n_layers: Number of feature map layers for expressivity.
            feature_map: Type of quantum feature map. "zz" is recommended
                for capturing price-time correlations.
            alpha: Regularization parameter. Increase if overfitting, decrease
                if underfitting.
            S_max: Maximum expected spot price, used for normalization.
                Should be set to cover the range of prices you'll encounter.
            T_max: Maximum time to maturity in years, used for normalization.
            include_moneyness: If True, also include log-moneyness as a feature,
                which can improve learning for options far from ATM.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map = feature_map
        self.alpha = alpha
        self.S_max = S_max
        self.T_max = T_max
        self.include_moneyness = include_moneyness
        
        # Strike price (set during fit if moneyness is used)
        self.K: Optional[float] = None
        
        # Create the regressor
        self.regressor = QuantumKernelRegression(
            n_qubits=n_qubits,
            n_layers=n_layers,
            feature_map=feature_map,
            alpha=alpha,
        )
        
        self.is_fitted = False
        
        # Store normalization statistics
        self._S_mean: Optional[float] = None
        self._S_std: Optional[float] = None
    
    def _prepare_features(
        self, 
        S: np.ndarray, 
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Prepare normalized features for the quantum kernel.
        
        Normalizes S and t to [0, π] range, which is natural for
        angle encoding in quantum circuits.
        
        Args:
            S: Spot prices of shape (n_samples,) or scalar.
            t: Times to maturity of shape (n_samples,) or scalar.
        
        Returns:
            Feature matrix of shape (n_samples, n_features).
        """
        S = np.atleast_1d(np.asarray(S, dtype=np.float64))
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        
        # Normalize to [0, 1]
        S_norm = S / self.S_max
        t_norm = t / self.T_max
        
        # Scale to [0, π] for natural angle encoding
        S_scaled = S_norm * np.pi
        t_scaled = t_norm * np.pi
        
        if self.include_moneyness and self.K is not None:
            # Add log-moneyness: log(S/K)
            moneyness = np.log(S / self.K)
            # Normalize moneyness (typical range is about [-0.5, 0.5])
            moneyness_scaled = (moneyness + 0.5) * np.pi
            moneyness_scaled = np.clip(moneyness_scaled, 0, np.pi)
            
            return np.column_stack([S_scaled, t_scaled, moneyness_scaled])
        else:
            return np.column_stack([S_scaled, t_scaled])
    
    def fit(
        self,
        S: np.ndarray,
        t: np.ndarray,
        V: np.ndarray,
        K: Optional[float] = None,
    ) -> "QuantumKernelPricer":
        """
        Fit the quantum kernel pricer on training data.
        
        Args:
            S: Spot prices of shape (n_samples,).
            t: Times to maturity of shape (n_samples,).
            V: Option values of shape (n_samples,).
            K: Strike price (optional). If provided and include_moneyness=True,
               log-moneyness will be used as an additional feature.
        
        Returns:
            self, for method chaining.
        
        Raises:
            ValueError: If input arrays have incompatible shapes.
        """
        S = np.asarray(S, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        V = np.asarray(V, dtype=np.float64).ravel()
        
        if not (len(S) == len(t) == len(V)):
            raise ValueError(
                f"S, t, and V must have same length, got {len(S)}, {len(t)}, {len(V)}"
            )
        
        # Store strike price for moneyness computation
        self.K = K
        
        # Prepare features
        X = self._prepare_features(S, t)
        
        # Fit the regressor
        self.regressor.fit(X, V)
        self.is_fitted = True
        
        return self
    
    def predict(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict option values for given spot prices and times.
        
        Args:
            S: Spot prices of shape (n_samples,) or scalar.
            t: Times to maturity of shape (n_samples,) or scalar.
        
        Returns:
            Predicted option values of shape (n_samples,).
        
        Raises:
            ValueError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Pricer has not been fitted. Call fit() first.")
        
        S = np.atleast_1d(np.asarray(S, dtype=np.float64))
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        
        X = self._prepare_features(S, t)
        V_pred = self.regressor.predict(X)
        
        # Ensure non-negative option values
        return np.maximum(V_pred, 0.0)
    
    def price(self, S: float, t: float) -> float:
        """
        Price a single option.
        
        Convenience method for pricing a single option contract.
        
        Args:
            S: Spot price of the underlying.
            t: Time to maturity in years.
        
        Returns:
            Predicted option value.
        """
        return float(self.predict(np.array([S]), np.array([t]))[0])
    
    def compute_greeks(
        self,
        S: float,
        t: float,
        dS: float = 0.01,
        dt: float = 0.001,
    ) -> dict[str, float]:
        """
        Compute option Greeks using finite differences.
        
        This method approximates Delta, Gamma, and Theta by finite differences
        on the learned pricing function.
        
        Args:
            S: Spot price.
            t: Time to maturity.
            dS: Step size for spot price differencing (as fraction of S).
            dt: Step size for time differencing.
        
        Returns:
            Dictionary with keys 'delta', 'gamma', 'theta'.
        
        Note:
            These are numerical approximations from the learned model.
            Accuracy depends on model quality and step sizes.
        """
        if not self.is_fitted:
            raise ValueError("Pricer has not been fitted. Call fit() first.")
        
        # Current price
        V = self.price(S, t)
        
        # Delta = ∂V/∂S
        S_up = S * (1 + dS)
        S_down = S * (1 - dS)
        V_up = self.price(S_up, t)
        V_down = self.price(S_down, t)
        delta = (V_up - V_down) / (S_up - S_down)
        
        # Gamma = ∂²V/∂S²
        gamma = (V_up - 2 * V + V_down) / ((S * dS) ** 2)
        
        # Theta = ∂V/∂t (note: t decreases as time passes)
        if t > dt:
            V_earlier = self.price(S, t - dt)
            theta = (V_earlier - V) / dt  # Negative since t decreases
        else:
            theta = 0.0
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'value': V,
        }
    
    def __repr__(self) -> str:
        return (
            f"QuantumKernelPricer(n_qubits={self.n_qubits}, "
            f"feature_map='{self.feature_map}', "
            f"S_max={self.S_max}, T_max={self.T_max}, "
            f"fitted={self.is_fitted})"
        )


class ProjectedQuantumKernel:
    """
    Projected Quantum Kernel for improved trainability.
    
    Standard quantum kernels can suffer from exponential concentration
    (the kernel values concentrate around a fixed value as the number of
    qubits increases). The projected quantum kernel addresses this by
    measuring in random bases and using the measurement statistics as
    classical features.
    
    Instead of computing |⟨φ(x)|φ(x')⟩|² directly, we:
        1. Compute classical feature vectors f(x) from quantum measurements
        2. Use classical kernel: k(x, x') = f(x) · f(x')
    
    This preserves some quantum advantage while avoiding concentration issues.
    
    References:
        Huang, H.-Y., et al. (2021). "Power of data in quantum machine learning."
        Nature Communications, 12, 2631.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_projections: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize projected quantum kernel.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit.
            n_projections: Number of random projections (measurement bases).
                More projections give better approximation but increase cost.
            seed: Random seed for reproducible projection bases.
        """
        self.n_qubits = n_qubits
        self.n_projections = n_projections
        
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Generate random unitary bases for projection
        rng = np.random.default_rng(seed)
        self._random_unitaries = []
        for _ in range(n_projections):
            # Generate random unitary via QR decomposition
            random_matrix = rng.standard_normal((2**n_qubits, 2**n_qubits)) + \
                           1j * rng.standard_normal((2**n_qubits, 2**n_qubits))
            Q, _ = np.linalg.qr(random_matrix)
            self._random_unitaries.append(Q)
        
        # Create circuit for state preparation
        self._state_circuit = self._create_state_circuit()
    
    def _create_state_circuit(self) -> qml.QNode:
        """Create circuit that returns the quantum state."""
        @qml.qnode(self.device)
        def circuit(x: np.ndarray) -> np.ndarray:
            n = min(len(x), self.n_qubits)
            
            # Simple but effective encoding
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i % n] * np.pi, wires=i)
                qml.RZ(x[i % n] * np.pi / 2, wires=i)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            return qml.state()
        
        return circuit
    
    def projected_features(self, x: np.ndarray) -> np.ndarray:
        """
        Compute projected quantum features.
        
        Prepares the quantum state |φ(x)⟩ and measures it in multiple
        random bases, returning the measurement probabilities as features.
        
        Args:
            x: Input vector.
        
        Returns:
            Classical feature vector of shape (n_projections,).
        """
        x = np.asarray(x, dtype=np.float64)
        
        # Get quantum state
        state = self._state_circuit(x)
        
        # Project onto random bases
        features = np.zeros(self.n_projections)
        for i, U in enumerate(self._random_unitaries):
            # Apply random unitary and compute |⟨0|U|φ(x)⟩|²
            projected = U @ state
            features[i] = np.abs(projected[0]) ** 2
        
        return features
    
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute projected kernel via classical inner product.
        
        k(x1, x2) = f(x1) · f(x2)
        
        where f(x) are the projected features.
        
        Args:
            x1: First input vector.
            x2: Second input vector.
        
        Returns:
            Kernel value.
        """
        f1 = self.projected_features(x1)
        f2 = self.projected_features(x2)
        return float(np.dot(f1, f2))
    
    def kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute kernel matrix using projected features.
        
        This is more efficient than the base kernel matrix computation
        because features can be computed once and reused.
        
        Args:
            X1: First set of points, shape (n_samples_1, n_features).
            X2: Optional second set, shape (n_samples_2, n_features).
        
        Returns:
            Kernel matrix.
        """
        # Compute features for X1
        F1 = np.array([self.projected_features(x) for x in X1])
        
        if X2 is None:
            # Gram matrix: K = F1 @ F1.T
            return F1 @ F1.T
        else:
            # Cross kernel
            F2 = np.array([self.projected_features(x) for x in X2])
            return F1 @ F2.T
    
    def __repr__(self) -> str:
        return (
            f"ProjectedQuantumKernel(n_qubits={self.n_qubits}, "
            f"n_projections={self.n_projections})"
        )


# Convenience functions for quick experiments

def create_quantum_pricer(
    n_qubits: int = 6,
    feature_map: str = "zz",
    alpha: float = 1e-3,
    S_max: float = 200.0,
    T_max: float = 1.0,
) -> QuantumKernelPricer:
    """
    Create a quantum kernel pricer with default settings.
    
    This is a convenience function for quickly creating a pricer
    for experimentation.
    
    Args:
        n_qubits: Number of qubits.
        feature_map: Type of feature map ("zz", "iqp", "pauli").
        alpha: Regularization parameter.
        S_max: Maximum spot price for normalization.
        T_max: Maximum time for normalization.
    
    Returns:
        Configured QuantumKernelPricer instance.
    """
    return QuantumKernelPricer(
        n_qubits=n_qubits,
        n_layers=2,
        feature_map=feature_map,
        alpha=alpha,
        S_max=S_max,
        T_max=T_max,
    )
