# System Architecture

This document describes the software architecture of the quantum-classical hybrid PINN derivatives pricing system.

---

## 1. Module Structure

```
quantum-derivatives-trader/
│
├── src/
│   ├── __init__.py
│   │
│   ├── pde/                          # PDE Definitions
│   │   ├── __init__.py
│   │   ├── black_scholes.py          # BS PDE, residual, analytical
│   │   ├── heston.py                 # Stochastic volatility PDE
│   │   └── jump_diffusion.py         # Merton jump-diffusion
│   │
│   ├── classical/                    # Classical PINN Implementation
│   │   ├── __init__.py
│   │   ├── networks.py               # MLP, ResidualMLP architectures
│   │   ├── pinn.py                   # PINN model and trainer
│   │   └── losses.py                 # Loss function components
│   │
│   ├── quantum/                      # Quantum Components
│   │   ├── __init__.py
│   │   ├── variational.py            # VQC layers, data encoding
│   │   ├── hybrid_pinn.py            # Hybrid quantum-classical PINN
│   │   ├── amplitude_estimation.py   # QAE for Monte Carlo
│   │   ├── kernels.py                # Quantum kernels
│   │   └── tensor_network.py         # MPS, TTN implementations
│   │
│   ├── pricing/                      # Pricing Engines
│   │   ├── __init__.py
│   │   ├── analytical.py             # Closed-form solutions
│   │   ├── monte_carlo.py            # Classical MC pricing
│   │   ├── finite_difference.py      # FD grid methods
│   │   └── pinn_pricer.py            # PINN-based pricer interface
│   │
│   ├── data/                         # Data Generation
│   │   ├── __init__.py
│   │   ├── collocation.py            # Collocation point sampling
│   │   ├── synthetic.py              # Synthetic training data
│   │   └── market_data.py            # Market data interfaces
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── greeks.py                 # Greeks computation
│       └── visualization.py          # Plotting utilities
│
├── scripts/                          # Executable Scripts
│   ├── train_classical.py            # Train classical PINN
│   ├── train_hybrid.py               # Train hybrid quantum PINN
│   └── benchmark.py                  # Run benchmarks
│
├── tests/                            # Test Suite
│   ├── test_pde/
│   ├── test_classical/
│   ├── test_quantum/
│   └── test_pricing/
│
├── notebooks/                        # Jupyter Notebooks
│   ├── 01_black_scholes_pinn.ipynb
│   ├── 02_hybrid_exploration.ipynb
│   └── 03_qae_analysis.ipynb
│
├── docs/                             # Documentation
│   ├── PHASES.md
│   ├── theory.md
│   └── architecture.md
│
└── ocaml/                            # Optional OCaml Core
    └── numerical/                    # High-performance numerics
```

---

## 2. Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            PRICING INTERFACE                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Analytical  │  │ Monte Carlo │  │   Finite    │  │  PINN Pricer    │ │
│  │   Pricer    │  │   Pricer    │  │ Difference  │  │ (Classical/Hyb) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
└─────────┼────────────────┼────────────────┼──────────────────┼──────────┘
          │                │                │                  │
          ▼                ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           PDE DEFINITIONS                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Black-Scholes  │  │     Heston      │  │    Jump Diffusion       │  │
│  │  - residual()   │  │  - residual()   │  │    - residual()         │  │
│  │  - analytical() │  │  - boundaries() │  │    - boundaries()       │  │
│  │  - greeks()     │  │                 │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL ARCHITECTURES                             │
│                                                                         │
│  ┌───────────────────────────────┐  ┌─────────────────────────────────┐ │
│  │      CLASSICAL PINN           │  │       HYBRID PINN               │ │
│  │  ┌─────────────────────────┐  │  │  ┌───────────────────────────┐  │ │
│  │  │         MLP             │  │  │  │    Classical Pre-Net      │  │ │
│  │  │  (S,t) → Hidden → V     │  │  │  │    (normalization)        │  │ │
│  │  └─────────────────────────┘  │  │  └─────────────┬─────────────┘  │ │
│  │  ┌─────────────────────────┐  │  │                ▼                │ │
│  │  │     ResidualMLP         │  │  │  ┌───────────────────────────┐  │ │
│  │  │  Skip connections       │  │  │  │    Quantum Layer (VQC)    │  │ │
│  │  └─────────────────────────┘  │  │  │  - Angle encoding         │  │ │
│  └───────────────────────────────┘  │  │  - Variational rotations  │  │ │
│                                     │  │  - Entanglement           │  │ │
│                                     │  │  - Measurement            │  │ │
│                                     │  └─────────────┬─────────────┘  │ │
│                                     │                ▼                │ │
│                                     │  ┌───────────────────────────┐  │ │
│                                     │  │   Classical Post-Net      │  │ │
│                                     │  │   (output scaling)        │  │ │
│                                     │  └───────────────────────────┘  │ │
│                                     └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING SYSTEM                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        LOSS FUNCTION                              │  │
│  │   L = λ₁·L_PDE + λ₂·L_BC + λ₃·L_IC                               │  │
│  │       │           │          │                                    │  │
│  │       ▼           ▼          ▼                                    │  │
│  │   Autodiff    Boundary   Terminal                                 │  │
│  │   PDE         Values     Payoff                                   │  │
│  │   Residual                                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    COLLOCATION SAMPLING                           │  │
│  │  - Interior points: random in (0, S_max) × (0, T)                 │  │
│  │  - Boundary points: S=0 and S=S_max edges                         │  │
│  │  - Terminal points: t=T with sampled S values                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Classical PINN Training

```
                    ┌─────────────────────────────────────────┐
                    │           TRAINING LOOP                 │
                    └─────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. COLLOCATION SAMPLING                                                 │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  generate_collocation_points(n_interior, n_boundary, n_term)   │   │
│     └────────────────────────────────────────────────────────────────┘   │
│                          │                                               │
│           ┌──────────────┼──────────────┬───────────────┐               │
│           ▼              ▼              ▼               ▼               │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐       │
│     │ S_int,   │   │ S_bc,    │   │ S_term   │   │  (batched    │       │
│     │ t_int    │   │ t_bc     │   │          │   │   tensors)   │       │
│     └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────────────┘       │
└──────────┼──────────────┼──────────────┼────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. FORWARD PASS                                                         │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  V = model.forward(S, t)                                       │   │
│     │  - Input normalization: S/S_max, t/T                          │   │
│     │  - MLP forward pass                                            │   │
│     │  - Output: option value predictions                            │   │
│     └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. LOSS COMPUTATION                                                     │
│                                                                          │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  L_PDE: Autodiff to compute dV/dt, dV/dS, d²V/dS²             │   │
│     │         Substitute into BS equation, compute MSE of residual   │   │
│     └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  L_BC:  Compare V(0,t) to 0, V(S_max,t) to asymptotic         │   │
│     └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  L_IC:  Compare V(S,T) to payoff max(S-K, 0)                  │   │
│     └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│     L_total = λ₁·L_PDE + λ₂·L_BC + λ₃·L_IC                              │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. BACKWARD PASS & UPDATE                                               │
│     ┌────────────────────────────────────────────────────────────────┐   │
│     │  loss.backward()                                               │   │
│     │  optimizer.step()                                              │   │
│     │  scheduler.step(loss)                                          │   │
│     └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              [Repeat for n_epochs]
```

### 3.2 Hybrid PINN Training

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HYBRID FORWARD PASS                                                     │
│                                                                          │
│  Input: (S, t)                                                           │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  1. Classical Preprocessing                                      │    │
│  │     x = stack([S/S_max, t/T])                                   │    │
│  │     x = pre_net(x)  # Linear → Tanh → Linear → Tanh             │    │
│  │     x = x * π       # Scale to [0, 2π] for quantum              │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  2. Quantum Circuit Execution                                    │    │
│  │     for each sample in batch:                                    │    │
│  │       ├─ Angle encoding: RY(x[0]), RY(x[1])                     │    │
│  │       ├─ Variational layers:                                     │    │
│  │       │   └─ for layer in range(n_layers):                      │    │
│  │       │       ├─ RX(θ), RY(θ), RZ(θ) on each qubit              │    │
│  │       │       └─ CNOT ring entanglement                          │    │
│  │       └─ Measurement: ⟨Z₀⟩ → output ∈ [-1, 1]                   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  3. Classical Postprocessing                                     │    │
│  │     V = post_net(q_out)  # Linear → ReLU → Linear → Softplus    │    │
│  │     V = V * S            # Scale by spot for correct magnitude   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  Output: V (option value)                                                │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  GRADIENT COMPUTATION                                                    │
│                                                                          │
│  PennyLane provides differentiable quantum circuits:                     │
│                                                                          │
│  ∂L/∂θ_quantum = ∂L/∂V × ∂V/∂q_out × ∂q_out/∂θ                          │
│                                     └────────────┘                       │
│                                     Parameter-shift                      │
│                                     or backprop                          │
│                                                                          │
│  All gradients flow through unified autodiff (PyTorch + PennyLane)       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. API Overview

### 4.1 PDE Interface

```python
from src.pde.black_scholes import BSParams, bs_pde_residual, bs_analytical

# Define model parameters
params = BSParams(r=0.05, sigma=0.2, K=100.0, T=1.0)

# Compute PDE residual at points (for PINN loss)
residual = bs_pde_residual(V, S, t, params, grad_fn=torch.autograd.grad)

# Get analytical solution for validation
V_exact = bs_analytical(S, t, params, option_type="call")
```

### 4.2 Classical PINN

```python
from src.classical.pinn import PINN, PINNTrainer
from src.pde.black_scholes import BSParams

# Create model
model = PINN(
    hidden_dims=[64, 64, 64, 64],
    S_max=200.0,
    T_max=1.0,
    use_residual=False,
)

# Create trainer
params = BSParams(r=0.05, sigma=0.2, K=100.0, T=1.0)
trainer = PINNTrainer(
    model=model,
    params=params,
    lr=1e-3,
    lambda_pde=1.0,
    lambda_bc=10.0,
    lambda_ic=10.0,
)

# Train
history = trainer.train(
    n_epochs=5000,
    n_interior=1000,
    n_boundary=200,
    n_terminal=200,
    print_every=500,
)

# Predict with Greeks
results = model.predict_with_greeks(S_test, t_test)
# Returns: {"V": ..., "delta": ..., "gamma": ..., "theta": ...}
```

### 4.3 Hybrid PINN

```python
from src.quantum.hybrid_pinn import HybridPINN

# Create hybrid model
model = HybridPINN(
    n_qubits=6,
    n_layers=4,
    classical_hidden=32,
    S_max=200.0,
    T_max=1.0,
    circuit_type="hardware_efficient",  # or "data_reuploading"
)

# Training uses same interface as classical PINN
# (same loss function, same trainer structure)
```

### 4.4 Quantum Amplitude Estimation

```python
from src.quantum.amplitude_estimation import QuantumMonteCarloEstimator

# Create QAE estimator
qae = QuantumMonteCarloEstimator(n_qubits=8)

# Define payoff
def call_payoff(S):
    return max(S - 100, 0)

# Estimate option price
result = qae.estimate_expectation(
    payoff_fn=call_payoff,
    S0=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_iterations=10,
)

print(f"Price: {result.estimate:.4f}")
print(f"95% CI: {result.confidence_interval}")
print(f"Queries: {result.n_queries}")
print(f"Classical equivalent: {result.classical_equivalent_samples} samples")
```

### 4.5 Pricing Engines

```python
from src.pricing import AnalyticalPricer, MonteCarloPricer, PINNPricer

# Common interface across all pricers
pricer = AnalyticalPricer(r=0.05, sigma=0.2)
price = pricer.price(S=100, K=100, T=1.0, option_type="call")

# Monte Carlo
mc_pricer = MonteCarloPricer(r=0.05, sigma=0.2, n_paths=100000)
price, std_error = mc_pricer.price(S=100, K=100, T=1.0, option_type="call")

# PINN-based (after training)
pinn_pricer = PINNPricer(model=trained_model, params=params)
price = pinn_pricer.price(S=100, t=0.0)
greeks = pinn_pricer.greeks(S=100, t=0.0)
```

---

## 5. Extension Points

### 5.1 Adding a New PDE

To add support for a new PDE (e.g., SABR model):

```python
# src/pde/sabr.py

from dataclasses import dataclass
import torch

@dataclass
class SABRParams:
    """SABR model parameters."""
    alpha: float   # Initial volatility
    beta: float    # CEV exponent
    rho: float     # Correlation
    nu: float      # Vol of vol
    r: float       # Risk-free rate
    K: float       # Strike
    T: float       # Maturity

def sabr_pde_residual(
    V: torch.Tensor,
    F: torch.Tensor,    # Forward price
    sigma: torch.Tensor,
    t: torch.Tensor,
    params: SABRParams,
) -> torch.Tensor:
    """
    SABR PDE residual.
    
    Implement the 2D SABR PDE similar to Heston.
    """
    # Compute derivatives via autodiff
    # Return residual that should equal zero
    pass

def sabr_terminal_condition(F: torch.Tensor, params: SABRParams) -> torch.Tensor:
    """Terminal payoff."""
    return torch.relu(F - params.K)
```

### 5.2 Adding a New Quantum Circuit

```python
# src/quantum/circuits/amplitude_encoding.py

import pennylane as qml

def create_amplitude_encoding_circuit(n_qubits: int, dev: qml.Device):
    """
    Create a circuit that uses amplitude encoding for inputs.
    
    More efficient for high-dimensional inputs but requires
    more complex state preparation.
    """
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        # Amplitude encoding
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        
        # Variational layers
        for layer in range(n_layers):
            # ... trainable rotations and entanglement
            pass
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit
```

### 5.3 Adding a New Pricing Model

```python
# src/pricing/local_volatility.py

from abc import ABC, abstractmethod

class BasePricer(ABC):
    """Abstract base class for all pricers."""
    
    @abstractmethod
    def price(self, S: float, K: float, T: float, **kwargs) -> float:
        """Compute option price."""
        pass
    
    @abstractmethod
    def greeks(self, S: float, K: float, T: float, **kwargs) -> dict:
        """Compute option Greeks."""
        pass

class LocalVolatilityPricer(BasePricer):
    """
    Local volatility model pricer.
    
    Uses Dupire's formula to extract local vol from market prices,
    then prices via finite difference or Monte Carlo.
    """
    
    def __init__(self, vol_surface: callable, r: float):
        self.local_vol = vol_surface
        self.r = r
    
    def price(self, S: float, K: float, T: float, **kwargs) -> float:
        # Implement local vol pricing
        pass
```

### 5.4 Custom Loss Functions

```python
# src/classical/losses.py

class WeightedPINNLoss:
    """
    PINN loss with spatially-varying weights.
    
    Useful for emphasizing accuracy in specific regions
    (e.g., near the strike for at-the-money options).
    """
    
    def __init__(self, params, K, weight_fn=None):
        self.params = params
        self.K = K
        self.weight_fn = weight_fn or (lambda S: 1.0)
    
    def __call__(self, model, S_int, t_int, ...):
        # Compute losses with position-dependent weights
        weights = self.weight_fn(S_int)
        L_pde = (weights * residual**2).mean()
        # ...
```

---

## 6. Testing Strategy

```
tests/
├── test_pde/
│   ├── test_black_scholes.py      # BS PDE residual, analytical
│   ├── test_heston.py             # Heston PDE, mixed derivatives
│   └── test_boundary.py           # Boundary condition tests
│
├── test_classical/
│   ├── test_networks.py           # MLP, ResidualMLP forward/backward
│   ├── test_pinn.py               # PINN training convergence
│   └── test_losses.py             # Loss computation accuracy
│
├── test_quantum/
│   ├── test_variational.py        # VQC forward pass, gradients
│   ├── test_hybrid.py             # Hybrid PINN integration
│   └── test_qae.py                # QAE estimates, error bounds
│
└── test_pricing/
    ├── test_analytical.py         # BS formula accuracy
    ├── test_monte_carlo.py        # MC convergence
    └── test_comparison.py         # Cross-method validation
```

**Testing Principles:**

1. **Unit Tests**: Each function tested in isolation
2. **Integration Tests**: End-to-end training and pricing
3. **Numerical Accuracy**: Compare against analytical solutions
4. **Gradient Verification**: Finite difference vs autodiff
5. **Convergence Tests**: Training stability and loss decay

---

## 7. Performance Considerations

### 7.1 Classical PINN

- **GPU Acceleration**: All PyTorch operations GPU-compatible
- **Batch Processing**: Collocation points processed in batches
- **JIT Compilation**: `torch.jit.script` for production deployment

### 7.2 Hybrid PINN

- **PennyLane Lightning**: Fast C++ backend for simulation
- **Batching Strategy**: Sequential quantum circuit calls (current limitation)
- **Future**: Parameter-shift rule parallelization

### 7.3 Memory Management

- **Gradient Checkpointing**: For deep networks
- **Mixed Precision**: FP16 for training speedup
- **Collocation Caching**: Reuse points across related problems
