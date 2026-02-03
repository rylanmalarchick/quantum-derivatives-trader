# System Architecture

> **Executive Summary**: This document describes the software architecture of the quantum-classical hybrid PINN system for derivatives pricing. The codebase follows a modular design with clear separation between PDE definitions, classical neural networks, quantum circuits, and pricing engines. All components are fully tested (200+ tests) and designed for extensibility.

**Key Design Principles**:
- PDEs are pure functions, independent of neural network implementation
- Unified pricing interface across analytical, MC, FD, and PINN methods  
- Quantum layers are drop-in replacements for classical layers
- Greeks computed via automatic differentiation, not finite differences

---


## 1. Module Structure Overview

```
quantum-derivatives-trader/
├── src/
│   ├── __init__.py
│   ├── pde/                    # PDE definitions
│   │   ├── __init__.py
│   │   ├── black_scholes.py    # Black-Scholes PDE and analytical solutions
│   │   ├── heston.py           # Heston stochastic volatility model
│   │   └── jump_diffusion.py   # Merton jump-diffusion model
│   │
│   ├── classical/              # Classical PINN implementation
│   │   ├── __init__.py
│   │   ├── networks.py         # MLP, ResidualMLP architectures
│   │   ├── pinn.py             # PINN model and trainer
│   │   └── losses.py           # Loss functions with adaptive weighting
│   │
│   ├── quantum/                # Quantum and hybrid implementations
│   │   ├── __init__.py
│   │   ├── variational.py      # Variational quantum circuits
│   │   ├── hybrid_pinn.py      # Hybrid quantum-classical PINN
│   │   ├── amplitude_estimation.py  # QAE for Monte Carlo
│   │   ├── tensor_network.py   # MPS, TTN implementations
│   │   └── kernels.py          # Quantum kernel methods
│   │
│   ├── pricing/                # Pricing engines
│   │   ├── __init__.py
│   │   ├── analytical.py       # Closed-form solutions
│   │   ├── monte_carlo.py      # Classical Monte Carlo
│   │   ├── finite_difference.py # Finite difference methods
│   │   └── pinn_pricer.py      # PINN-based pricer
│   │
│   ├── data/                   # Data generation
│   │   ├── __init__.py
│   │   ├── collocation.py      # Collocation point sampling
│   │   ├── synthetic.py        # Synthetic data generation
│   │   └── market_data.py      # Market data interfaces
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── greeks.py           # Greeks computation
│       └── visualization.py    # Plotting utilities
│
├── tests/                      # Test suite
│   ├── conftest.py
│   ├── test_pde.py
│   ├── test_classical_pinn.py
│   ├── test_quantum.py
│   └── test_pricing.py
│
├── scripts/                    # Runnable scripts
│   ├── train_classical.py
│   ├── train_hybrid.py
│   └── benchmark.py
│
├── notebooks/                  # Jupyter notebooks
│   └── ...
│
└── docs/                       # Documentation
    ├── PHASES.md
    ├── theory.md
    └── architecture.md
```

---

## 2. Core Components

### 2.1 PDE Module (`src/pde/`)

Defines the partial differential equations and their boundary conditions.

```python
# src/pde/black_scholes.py

@dataclass
class BSParams:
    """Black-Scholes parameters."""
    r: float      # Risk-free rate
    sigma: float  # Volatility
    K: float      # Strike
    T: float      # Time to maturity

def bs_pde_residual(V, S, t, params, grad_fn) -> torch.Tensor:
    """Compute PDE residual: should be zero if V satisfies BS."""
    dV_dt = grad_fn(V, t)
    dV_dS = grad_fn(V, S)
    d2V_dS2 = grad_fn(dV_dS, S)
    
    return dV_dt + 0.5*σ²*S²*d2V_dS2 + r*S*dV_dS - r*V

def bs_analytical(S, t, params) -> torch.Tensor:
    """Closed-form Black-Scholes solution."""
    ...
```

**Design Principle:** PDEs are pure functions operating on tensors, independent of the neural network implementation.

### 2.2 Classical Module (`src/classical/`)

Implements the classical PINN architecture.

```python
# src/classical/pinn.py

class PINN(nn.Module):
    """Physics-Informed Neural Network."""
    
    def __init__(self, hidden_dims, S_max, T_max, use_residual=False):
        self.network = MLP(...) or ResidualMLP(...)
        self.S_max, self.T_max = S_max, T_max
    
    def forward(self, S, t) -> torch.Tensor:
        """Predict option value V(S, t)."""
        x = normalize(S, t)
        return self.network(x)
    
    def predict_with_greeks(self, S, t) -> dict:
        """Predict V and compute Greeks via autodiff."""
        ...

class PINNTrainer:
    """Trainer with loss computation and optimization."""
    
    def compute_loss(self, S_int, t_int, S_bc, t_bc, S_term) -> dict:
        """Compute PDE, BC, and IC losses."""
        ...
    
    def train(self, n_epochs, ...) -> dict:
        """Full training loop with fresh collocation points."""
        ...
```

### 2.3 Quantum Module (`src/quantum/`)

Implements variational quantum circuits and hybrid architectures.

```python
# src/quantum/variational.py

class QuantumLayer(nn.Module):
    """Variational quantum circuit as a PyTorch layer."""
    
    def __init__(self, n_qubits, n_layers, circuit_type):
        self.circuit = create_vqc(n_qubits, n_layers, device)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
    
    def forward(self, x) -> torch.Tensor:
        """Execute quantum circuit for each input."""
        outputs = [self.circuit(x[i], self.weights) for i in range(batch)]
        return torch.stack(outputs)
```

```python
# src/quantum/hybrid_pinn.py

class HybridPINN(nn.Module):
    """Hybrid quantum-classical architecture."""
    
    def __init__(self, n_qubits, n_layers, classical_hidden, ...):
        self.pre_net = nn.Sequential(...)   # Classical preprocessing
        self.quantum = QuantumLayer(...)     # Quantum circuit
        self.post_net = nn.Sequential(...)   # Classical postprocessing
    
    def forward(self, S, t) -> torch.Tensor:
        x = self.pre_net(normalize(S, t))
        x = self.quantum(x * π)
        return self.post_net(x) * S
```

### 2.4 Pricing Module (`src/pricing/`)

Provides a unified interface for all pricing methods.

```python
# src/pricing/pinn_pricer.py

class PINNPricer:
    """Option pricer using trained PINN model."""
    
    def __init__(self, model, params, device):
        self.model = model.eval()
        self.params = params
    
    @classmethod
    def load(cls, model_path, params, model_type) -> "PINNPricer":
        """Load trained model from disk."""
        ...
    
    def price(self, S, t) -> np.ndarray:
        """Price option at given spot and time."""
        ...
    
    def delta(self, S, t) -> np.ndarray:
        """Compute delta via autodiff."""
        ...
    
    def greeks(self, S, t) -> dict:
        """Compute all Greeks at once."""
        ...
    
    def pde_residual(self, S, t) -> np.ndarray:
        """Compute PDE residual for validation."""
        ...
```

---

## 3. Data Flow

### 3.1 Classical PINN Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Classical PINN Training                             │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │  Collocation Points  │
                        │   generate_points()  │
                        └──────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │   Interior    │     │   Boundary    │     │   Terminal    │
    │   (S, t)      │     │   (S_bc, t)   │     │   (S_T)       │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
    ┌───────────────────────────────────────────────────────────┐
    │                        PINN Model                          │
    │   Input → Normalize → MLP → V(S,t)                        │
    └───────────────────────────────────────────────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │  L_PDE        │     │  L_BC         │     │  L_IC         │
    │  (residual²)  │     │  (BC error²)  │     │  (payoff err) │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │   L_total =          │
                        │   λ₁L_PDE + λ₂L_BC + │
                        │   λ₃L_IC             │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Backpropagation    │
                        │   Adam Optimizer     │
                        └──────────────────────┘
```

### 3.2 Hybrid Quantum-Classical PINN Training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Hybrid PINN Training                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────────┐
                        │   Input (S, t)       │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Normalization      │
                        │   S/S_max, t/T_max   │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Classical PreNet   │
                        │   Linear → Tanh →    │
                        │   Linear → Tanh      │
                        └──────────────────────┘
                                   │
                                   ▼ × π (scale to [0, 2π])
                        ┌──────────────────────┐
                        │   Quantum Layer      │
                        │                      │
                        │   ┌────────────────┐ │
                        │   │ Data Encoding  │ │
                        │   │ RY(x₀), RY(x₁) │ │
                        │   └────────────────┘ │
                        │          ↓           │
                        │   ┌────────────────┐ │
                        │   │ Variational    │ │
                        │   │ RX,RY,RZ + CNOT│ │
                        │   └────────────────┘ │
                        │          ↓ × L       │
                        │   ┌────────────────┐ │
                        │   │ Measurement    │ │
                        │   │ ⟨σ_z⟩ ∈ [-1,1] │ │
                        │   └────────────────┘ │
                        └──────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │   Classical PostNet  │
                        │   Linear → ReLU →    │
                        │   Linear → Softplus  │
                        └──────────────────────┘
                                   │
                                   ▼ × S (scale to option value)
                        ┌──────────────────────┐
                        │   Output: V(S, t)    │
                        └──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────┐
            │   PINN Loss   │             │   Parameter   │
            │   (same as    │             │   gradient    │
            │    classical) │             │   via backprop│
            └───────────────┘             └───────────────┘
                                                  │
                    ┌─────────────────────────────┤
                    ▼                             ▼
            ┌───────────────┐             ┌───────────────┐
            │   Classical   │             │   Quantum     │
            │   parameters  │             │   weights     │
            │   update      │             │   update      │
            └───────────────┘             └───────────────┘
```

### 3.3 Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Inference Pipeline                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   User Request:      │
    │   price(S=100, t=0)  │
    └──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │     PINNPricer       │
    │   (trained model)    │
    └──────────────────────┘
               │
               ├─────────────────────────────────────────────┐
               │                                             │
               ▼                                             ▼
    ┌──────────────────────┐                      ┌──────────────────────┐
    │   Price Only         │                      │   Price + Greeks     │
    │   (no grad)          │                      │   (with grad)        │
    └──────────────────────┘                      └──────────────────────┘
               │                                             │
               ▼                                             ▼
    ┌──────────────────────┐                      ┌──────────────────────┐
    │   model(S, t)        │                      │   V = model(S, t)    │
    │   return V           │                      │   Δ = ∂V/∂S         │
    └──────────────────────┘                      │   Γ = ∂²V/∂S²       │
                                                  │   Θ = ∂V/∂t         │
                                                  └──────────────────────┘
```

---

## 4. Pricing Engine API

### 4.1 Abstract Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class PricingResult:
    """Standard result from pricing engine."""
    price: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    error_estimate: float | None = None
    computation_time: float | None = None

class BasePricer(ABC):
    """Abstract base class for all pricing engines."""
    
    @abstractmethod
    def price(self, S: float, t: float) -> float:
        """Price option at given spot and time."""
        pass
    
    @abstractmethod
    def greeks(self, S: float, t: float) -> dict[str, float]:
        """Compute all Greeks."""
        pass
```

### 4.2 Implemented Pricers

| Pricer | Method | Strengths | Limitations |
|--------|--------|-----------|-------------|
| `AnalyticalPricer` | Black-Scholes formula | Exact, fast | Only vanilla European |
| `MonteCarloPricer` | Simulation | Flexible, handles exotics | Slow for high precision |
| `FiniteDifferencePricer` | PDE discretization | Handles early exercise | Curse of dimensionality |
| `PINNPricer` | Trained neural network | Fast inference, smooth Greeks | Requires training |

### 4.3 Usage Example

```python
from src.pde.black_scholes import BSParams
from src.pricing.pinn_pricer import PINNPricer
from src.pricing.analytical import AnalyticalPricer

# Parameters
params = BSParams(r=0.05, sigma=0.2, K=100, T=1.0)

# Load trained PINN
pinn_pricer = PINNPricer.load(
    "models/classical_pinn.pt",
    params,
    model_type="classical"
)

# Compare with analytical
analytical_pricer = AnalyticalPricer(params)

# Price at S=100, t=0
S, t = 100.0, 0.0

pinn_price = pinn_pricer.price(S, t)
analytical_price = analytical_pricer.price(S, t)

print(f"PINN: {pinn_price:.4f}")
print(f"Analytical: {analytical_price:.4f}")
print(f"Error: {abs(pinn_price - analytical_price):.6f}")

# Get all Greeks
greeks = pinn_pricer.greeks(S, t)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
```

---

## 5. Extension Points

### 5.1 Adding a New PDE

To add a new PDE (e.g., local volatility):

```python
# src/pde/local_volatility.py

@dataclass
class LocalVolParams:
    """Local volatility model parameters."""
    r: float
    sigma_func: Callable[[float, float], float]  # σ(S, t)
    K: float
    T: float

def local_vol_pde_residual(V, S, t, params, grad_fn) -> torch.Tensor:
    """Local volatility PDE residual."""
    sigma = params.sigma_func(S, t)
    
    dV_dt = grad_fn(V, t)
    dV_dS = grad_fn(V, S)
    d2V_dS2 = grad_fn(dV_dS, S)
    
    return dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + params.r * S * dV_dS - params.r * V
```

Then update the loss function to use the new residual.

### 5.2 Adding a New Quantum Circuit

To add a new VQC architecture:

```python
# src/quantum/variational.py

def create_strongly_entangling_circuit(n_qubits, n_layers, dev):
    """Strongly entangling layers (Schuld et al.)."""
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        
        # Strongly entangling layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit
```

Then add to `QuantumLayer`:

```python
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, circuit_type):
        ...
        if circuit_type == "strongly_entangling":
            self.circuit = create_strongly_entangling_circuit(...)
```

### 5.3 Adding a New Pricing Engine

To add a new pricing method:

```python
# src/pricing/neural_operator.py

class NeuralOperatorPricer:
    """Fourier Neural Operator for option pricing."""
    
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def price(self, S, t):
        ...
    
    def greeks(self, S, t):
        ...
```

Register in `src/pricing/__init__.py`:

```python
from .neural_operator import NeuralOperatorPricer

__all__ = [..., "NeuralOperatorPricer"]
```

### 5.4 Adding New Option Types

Extend boundary conditions for new option types:

```python
# src/classical/losses.py

class PINNLoss:
    def __init__(self, params, option_type="european_call"):
        self.option_type = option_type
    
    def terminal_loss(self, model, S):
        t_T = torch.full_like(S, self.params.T)
        V = model(S, t_T)
        
        if self.option_type == "european_call":
            payoff = torch.relu(S - self.params.K)
        elif self.option_type == "european_put":
            payoff = torch.relu(self.params.K - S)
        elif self.option_type == "digital_call":
            payoff = (S > self.params.K).float()
        elif self.option_type == "straddle":
            payoff = torch.abs(S - self.params.K)
        else:
            raise ValueError(f"Unknown option type: {self.option_type}")
        
        return ((V - payoff) ** 2).mean()
```

---

## 6. Configuration Management

### 6.1 Model Configuration

```yaml
# config/classical_pinn.yaml
model:
  type: classical
  hidden_dims: [64, 64, 64, 64]
  use_residual: false
  S_max: 200.0
  T_max: 1.0

training:
  n_epochs: 5000
  lr: 1e-3
  lambda_pde: 1.0
  lambda_bc: 10.0
  lambda_ic: 10.0
  
collocation:
  n_interior: 1000
  n_boundary: 200
  n_terminal: 200
```

```yaml
# config/hybrid_pinn.yaml
model:
  type: hybrid
  n_qubits: 6
  n_layers: 4
  classical_hidden: 32
  circuit_type: hardware_efficient

training:
  n_epochs: 2000
  lr: 5e-3
  ...
```

### 6.2 Loading Configuration

```python
import yaml

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_model_from_config(config):
    if config["model"]["type"] == "classical":
        return PINN(**config["model"])
    elif config["model"]["type"] == "hybrid":
        return HybridPINN(**config["model"])
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
```

---

## 7. Testing Strategy

### 7.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| **Unit tests** | Test individual functions | `tests/test_*.py` |
| **Integration tests** | Test component interactions | `tests/test_pricing.py` |
| **Regression tests** | Ensure accuracy vs benchmarks | `tests/test_classical_pinn.py` |
| **Property tests** | Test invariants (e.g., put-call parity) | `tests/test_pde.py` |

### 7.2 Key Test Cases

```python
# tests/test_pde.py

def test_bs_analytical_call_put_parity():
    """Call - Put = S - K*exp(-rT) for European options."""
    params = BSParams(r=0.05, sigma=0.2, K=100, T=1.0)
    S = torch.linspace(50, 150, 20)
    t = torch.zeros_like(S)
    
    call = bs_analytical(S, t, params, "call")
    put = bs_analytical(S, t, params, "put")
    
    parity = call - put - (S - params.K * np.exp(-params.r * params.T))
    assert torch.allclose(parity, torch.zeros_like(parity), atol=1e-6)

def test_pde_residual_at_analytical_solution():
    """PDE residual should be zero for exact solution."""
    ...
```

```python
# tests/test_classical_pinn.py

def test_pinn_converges():
    """PINN training should reduce total loss."""
    model = PINN()
    trainer = PINNTrainer(model, params)
    history = trainer.train(n_epochs=100)
    
    assert history["total"][-1] < history["total"][0] * 0.1

def test_pinn_accuracy():
    """Trained PINN should match analytical within tolerance."""
    # ... train or load model ...
    pinn_price = pricer.price(S=100, t=0)
    analytical_price = bs_analytical(100, 0, params)
    
    assert abs(pinn_price - analytical_price) / analytical_price < 0.01
```

---

## 8. Performance Considerations

### 8.1 Training Performance

| Bottleneck | Solution |
|------------|----------|
| Quantum circuit evaluation | Use batched simulation, lightning.qubit backend |
| Gradient computation | Use efficient autodiff (retain_graph carefully) |
| Collocation point generation | Generate in parallel, use caching |
| Loss computation | Vectorized operations, avoid Python loops |

### 8.2 Inference Performance

| Scenario | Optimization |
|----------|--------------|
| Single-point pricing | Direct model call, minimal overhead |
| Batch pricing | Vectorized forward pass |
| Greeks computation | Single forward + backward pass |
| Grid evaluation | Pre-allocate tensors, batch processing |

### 8.3 Memory Management

```python
# Efficient Greeks computation
def greeks(self, S, t):
    with torch.enable_grad():
        S_t = torch.tensor(S).requires_grad_(True)
        t_t = torch.tensor(t).requires_grad_(True)
        
        V = self.model(S_t, t_t)
        
        # Compute all Greeks in one backward pass
        grads = torch.autograd.grad(
            V.sum(), [S_t, t_t],
            create_graph=True, retain_graph=True
        )
        delta, theta = grads
        
        gamma = torch.autograd.grad(
            delta.sum(), S_t
        )[0]
    
    return {"price": V, "delta": delta, "gamma": gamma, "theta": theta}
```

---

## 9. Deployment Considerations

### 9.1 Model Serialization

```python
# Save trained model
torch.save({
    "model_state_dict": model.state_dict(),
    "params": asdict(params),
    "config": config,
    "training_history": history,
}, "models/pinn_v1.pt")

# Load for inference
checkpoint = torch.load("models/pinn_v1.pt")
model = PINN(**checkpoint["config"]["model"])
model.load_state_dict(checkpoint["model_state_dict"])
params = BSParams(**checkpoint["params"])
```

### 9.2 Production API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PricingRequest(BaseModel):
    spot: float
    time: float
    include_greeks: bool = False

@app.post("/price")
def price_option(request: PricingRequest):
    if request.include_greeks:
        result = pricer.greeks(request.spot, request.time)
    else:
        result = {"price": pricer.price(request.spot, request.time)}
    return result
```
