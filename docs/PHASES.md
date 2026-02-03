# Project Roadmap: Quantum-Classical Hybrid PINNs for Derivatives Pricing

This document outlines the six-phase development roadmap for exploring quantum-enhanced methods in financial derivatives pricing.

---

## Phase 1: Classical PINN for Black-Scholes

**Status:** 🔄 In Progress

### Goals

Establish a robust classical baseline using Physics-Informed Neural Networks to solve the Black-Scholes PDE for European options. This provides:
- Ground truth for comparing quantum approaches
- Validation of PINN methodology for option pricing
- Baseline computational benchmarks

### Technical Objectives

1. **PDE Implementation**
   - Black-Scholes residual: $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$
   - Boundary conditions for European calls/puts
   - Terminal condition: $$V(S, T) = \max(S - K, 0)$$ for calls

2. **Network Architecture**
   - Multi-layer perceptron with configurable depth/width
   - Residual connections for deeper networks
   - Input normalization for numerical stability

3. **Training Pipeline**
   - Collocation point sampling (interior, boundary, terminal)
   - Weighted loss function: $$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{BC}} + \lambda_3 \mathcal{L}_{\text{IC}}$$
   - Adaptive learning rate scheduling

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1.1 | PDE residual implementation with autodiff | ✅ Complete |
| M1.2 | Analytical Black-Scholes for validation | ✅ Complete |
| M1.3 | PINN architecture with MLP/ResNet | ✅ Complete |
| M1.4 | Training loop with collocation sampling | ✅ Complete |
| M1.5 | Greeks computation via autodiff | ✅ Complete |
| M1.6 | Convergence analysis and hyperparameter tuning | 🔄 In Progress |
| M1.7 | Comprehensive test suite | 🔄 In Progress |
| M1.8 | Benchmark against finite difference methods | ⏳ Pending |

### Success Criteria

- Relative error < 1% vs analytical Black-Scholes across price range
- Greeks (Delta, Gamma, Theta) within 5% of analytical values
- Training convergence in < 5000 epochs
- Documented performance vs finite difference and Monte Carlo

---

## Phase 2: Hybrid Quantum-Classical PINN

**Status:** ⏳ Pending

### Goals

Replace the classical neural network with variational quantum circuits (VQCs) to explore whether quantum expressivity provides advantages for PDE solutions.

### Technical Objectives

1. **Variational Quantum Circuit Design**
   - Hardware-efficient ansatz with trainable rotations
   - Data re-uploading circuits for increased expressivity
   - Ring and all-to-all entanglement topologies

2. **Hybrid Architecture**
   - Classical preprocessing → Quantum circuit → Classical postprocessing
   - Multiple quantum layer configurations (single, stacked, residual)
   - Differentiable quantum-classical interface via PennyLane

3. **Input Encoding Strategies**
   - Angle encoding: $$|x\rangle \rightarrow R_Y(\pi x)|0\rangle$$
   - Amplitude encoding for higher-dimensional inputs
   - Data re-uploading for enhanced function approximation

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M2.1 | Variational quantum circuit implementation | ⏳ Pending |
| M2.2 | PennyLane integration with PyTorch | ⏳ Pending |
| M2.3 | Single-layer hybrid PINN | ⏳ Pending |
| M2.4 | Multi-layer hybrid architectures | ⏳ Pending |
| M2.5 | Quantum residual correction model | ⏳ Pending |
| M2.6 | Barren plateau analysis | ⏳ Pending |
| M2.7 | Expressivity comparison with classical NNs | ⏳ Pending |

### Research Questions

- Does the quantum circuit's expressivity help approximate option pricing functions?
- What is the optimal circuit depth vs. trainability trade-off?
- How do barren plateaus affect convergence for financial PDEs?
- At what problem size does quantum advantage potentially emerge?

---

## Phase 3: Quantum Amplitude Estimation

**Status:** ⏳ Pending

### Goals

Implement Quantum Amplitude Estimation (QAE) for Monte Carlo option pricing, achieving the theoretical $$O(1/\epsilon)$$ vs classical $$O(1/\epsilon^2)$$ speedup.

### Technical Objectives

1. **State Preparation**
   - Encode log-normal distribution in quantum amplitudes
   - Efficient preparation circuits for financial distributions
   - Multi-asset distribution encoding

2. **Payoff Oracle**
   - Encode payoff function as controlled rotations
   - European, Asian, and barrier option payoffs
   - Comparator circuits for digital payoffs

3. **Amplitude Estimation Variants**
   - Canonical QAE with phase estimation
   - Iterative QAE (no QPE required)
   - Maximum likelihood amplitude estimation

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M3.1 | Log-normal state preparation | ⏳ Pending |
| M3.2 | Payoff rotation encoding | ⏳ Pending |
| M3.3 | Canonical QAE implementation | ⏳ Pending |
| M3.4 | Iterative QAE variant | ⏳ Pending |
| M3.5 | Error bound verification | ⏳ Pending |
| M3.6 | Resource estimation for practical problems | ⏳ Pending |
| M3.7 | Comparison with classical Monte Carlo | ⏳ Pending |

### Key Metrics

- Query complexity vs classical sample complexity
- Circuit depth requirements
- Qubit count scaling with precision
- Practical crossover point estimation

---

## Phase 4: Heston & Exotic Options

**Status:** ⏳ Pending

### Goals

Extend PINNs to handle the Heston stochastic volatility model and exotic options with path-dependent features.

### Technical Objectives

1. **Heston Model PDE**
   $$\frac{\partial V}{\partial t} + \frac{1}{2}vS^2\frac{\partial^2 V}{\partial S^2} + \rho\xi vS\frac{\partial^2 V}{\partial S \partial v} + \frac{1}{2}\xi^2 v\frac{\partial^2 V}{\partial v^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta - v)\frac{\partial V}{\partial v} - rV = 0$$

2. **Exotic Options**
   - Barrier options (up-and-out, down-and-in, etc.)
   - Asian options (arithmetic and geometric average)
   - Lookback options

3. **Technical Challenges**
   - 2D PDE with mixed derivatives
   - Non-smooth boundary conditions
   - Path-dependent payoffs requiring augmented state space

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M4.1 | Heston PDE residual with mixed derivatives | ⏳ Pending |
| M4.2 | Classical PINN for Heston | ⏳ Pending |
| M4.3 | Hybrid quantum PINN for 2D PDE | ⏳ Pending |
| M4.4 | Barrier option boundary handling | ⏳ Pending |
| M4.5 | Asian option with path augmentation | ⏳ Pending |
| M4.6 | Calibration to market data | ⏳ Pending |

---

## Phase 5: Tensor Network Methods

**Status:** ⏳ Pending

### Goals

Implement quantum-inspired tensor network methods (MPS, TTN) for high-dimensional option pricing, particularly multi-asset derivatives.

### Technical Objectives

1. **Matrix Product States (MPS)**
   - Efficient representation of multi-variate functions
   - Bond dimension control for accuracy/speed trade-off
   - Variational optimization of MPS parameters

2. **Tree Tensor Networks (TTN)**
   - Hierarchical decomposition for multi-scale problems
   - Natural handling of asset correlation structures
   - Efficient contraction algorithms

3. **Applications**
   - Basket options with many underlyings
   - Rainbow options (best-of, worst-of)
   - High-dimensional American options

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M5.1 | MPS representation layer | ⏳ Pending |
| M5.2 | TTN architecture | ⏳ Pending |
| M5.3 | Tensor train optimization | ⏳ Pending |
| M5.4 | Multi-asset basket pricing | ⏳ Pending |
| M5.5 | Comparison with classical methods | ⏳ Pending |
| M5.6 | Scaling analysis | ⏳ Pending |

### Research Questions

- How does MPS bond dimension relate to option complexity?
- Can TTN naturally capture asset correlation structures?
- What is the practical dimensionality limit?
- Comparison with Monte Carlo for n > 10 assets?

---

## Phase 6: Benchmarks & Analysis

**Status:** ⏳ Pending

### Goals

Comprehensive comparison of all implemented methods with rigorous statistical analysis and practical recommendations.

### Technical Objectives

1. **Benchmark Suite**
   - Standardized test cases across methods
   - Multiple option types and market conditions
   - Varying precision requirements

2. **Metrics**
   - Accuracy (relative error, Greeks error)
   - Computational cost (time, memory, circuit depth)
   - Scalability (dimension, precision, time-to-solution)
   - Robustness (market condition sensitivity)

3. **Analysis**
   - Crossover point analysis for quantum advantage
   - Resource estimation for real quantum hardware
   - Practical deployment recommendations

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M6.1 | Unified benchmarking framework | ⏳ Pending |
| M6.2 | European option benchmark suite | ⏳ Pending |
| M6.3 | Exotic option benchmark suite | ⏳ Pending |
| M6.4 | Multi-asset benchmark suite | ⏳ Pending |
| M6.5 | Statistical analysis and visualization | ⏳ Pending |
| M6.6 | Research paper / report | ⏳ Pending |

### Deliverables

- Benchmark data and analysis notebooks
- Performance comparison tables and figures
- Guidelines for method selection based on use case
- Open-source release with documentation

---

## Timeline & Dependencies

```
Phase 1 ──────────────►
         │
         ├──► Phase 2 ──────────────►
         │             │
         │             └──► Phase 3 ─────────►
         │                          │
         └──► Phase 4 ──────────────┤
                       │            │
                       └──► Phase 5 ┤
                                    │
                                    └──► Phase 6
```

**Estimated Timeline:**
- Phase 1: Weeks 1-4
- Phase 2: Weeks 3-8
- Phase 3: Weeks 6-10
- Phase 4: Weeks 8-14
- Phase 5: Weeks 12-18
- Phase 6: Weeks 16-20
