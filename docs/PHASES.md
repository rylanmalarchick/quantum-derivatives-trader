# Project Roadmap: Quantum-Classical Hybrid PINNs for Derivatives Pricing

This document outlines the implementation roadmap for exploring quantum computing approaches to financial derivatives pricing through physics-informed neural networks.

---

## Phase 1: Classical PINN for Black-Scholes (In Progress)

**Status:** 🔄 In Progress  
**Timeline:** Weeks 1-4

### Goals

1. Establish a robust classical baseline using Physics-Informed Neural Networks
2. Validate PINN approach against analytical Black-Scholes solutions
3. Build infrastructure for training, evaluation, and benchmarking
4. Create reusable components for quantum extensions

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1.1 | Implement Black-Scholes PDE residual computation | ✅ Complete |
| M1.2 | Build MLP and ResidualMLP architectures | ✅ Complete |
| M1.3 | Implement PINN loss function (PDE + BC + IC) | ✅ Complete |
| M1.4 | Create collocation point sampling strategies | ✅ Complete |
| M1.5 | Train and validate against analytical solutions | 🔄 In Progress |
| M1.6 | Implement adaptive loss weighting | ✅ Complete |
| M1.7 | Add Greeks computation via autodiff | ✅ Complete |
| M1.8 | Benchmark against finite difference methods | ⏳ Pending |

### Deliverables

- `src/classical/pinn.py` - PINN model and trainer
- `src/classical/losses.py` - Loss functions with adaptive weighting
- `src/pde/black_scholes.py` - PDE definition and analytical solutions
- `src/pricing/pinn_pricer.py` - Pricing engine interface
- `tests/test_classical_pinn.py` - Comprehensive test suite
- `scripts/train_classical.py` - Training script with logging

### Success Criteria

- [ ] PINN achieves < 1% relative error vs analytical Black-Scholes
- [ ] Greeks (delta, gamma, theta) within 2% of analytical values
- [ ] Training converges within 5000 epochs
- [ ] PDE residual < 1e-4 across domain

---

## Phase 2: Hybrid Quantum-Classical PINN

**Status:** ⏳ Pending  
**Timeline:** Weeks 5-8

### Goals

1. Replace classical MLP with variational quantum circuits (VQC)
2. Explore different quantum circuit architectures for function approximation
3. Compare quantum expressivity with classical networks of similar parameter counts
4. Identify regimes where hybrid models may offer advantages

### Architecture Options

```
Option A: Direct Replacement
┌─────────────────────────────────────────────────────────────┐
│  Input (S,t) → Normalize → VQC → Postprocess → V           │
└─────────────────────────────────────────────────────────────┘

Option B: Quantum Residual
┌─────────────────────────────────────────────────────────────┐
│  Input → Classical MLP → V_classical                        │
│       └→ VQC          → V_quantum   → V_classical + αV_q   │
└─────────────────────────────────────────────────────────────┘

Option C: Interleaved Layers
┌─────────────────────────────────────────────────────────────┐
│  Input → Classical → VQC → Classical → VQC → ... → Output  │
└─────────────────────────────────────────────────────────────┘
```

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M2.1 | Implement hardware-efficient VQC ansatz | ✅ Complete |
| M2.2 | Implement data re-uploading circuit | ✅ Complete |
| M2.3 | Build HybridPINN with preprocessing/postprocessing | ✅ Complete |
| M2.4 | Implement QuantumResidualPINN architecture | ✅ Complete |
| M2.5 | Train hybrid models on Black-Scholes | ⏳ Pending |
| M2.6 | Compare expressivity vs parameter count | ⏳ Pending |
| M2.7 | Analyze barren plateau effects | ⏳ Pending |
| M2.8 | Document optimal circuit configurations | ⏳ Pending |

### Deliverables

- `src/quantum/variational.py` - VQC implementations
- `src/quantum/hybrid_pinn.py` - Hybrid model architectures
- `scripts/train_hybrid.py` - Hybrid training script
- `notebooks/quantum_expressivity.ipynb` - Expressivity analysis

### Success Criteria

- [ ] Hybrid PINN achieves comparable accuracy to classical with fewer parameters
- [ ] Identify optimal qubit count and layer depth for 2D input problems
- [ ] Characterize training dynamics (convergence, variance, barren plateaus)

---

## Phase 3: Quantum Amplitude Estimation

**Status:** ⏳ Pending  
**Timeline:** Weeks 9-12

### Goals

1. Implement Quantum Amplitude Estimation for Monte Carlo pricing
2. Demonstrate quadratic speedup scaling O(1/N) vs O(1/√N)
3. Compare with classical Monte Carlo for various precision targets
4. Analyze resource requirements for practical quantum advantage

### Algorithm Overview

```
Classical MC:    Error ~ O(1/√N)  →  N samples for precision ε
Quantum AE:      Error ~ O(1/M)   →  M queries for precision ε

Speedup: M² = N  →  Quadratic reduction in oracle calls
```

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M3.1 | Implement log-normal distribution encoding | ✅ Complete |
| M3.2 | Implement payoff-as-rotation encoding | ✅ Complete |
| M3.3 | Build basic QAE estimator | ✅ Complete |
| M3.4 | Implement Iterative QAE (no QPE) | ✅ Complete |
| M3.5 | Simulate QAE for European options | ⏳ Pending |
| M3.6 | Compare precision vs query complexity | ⏳ Pending |
| M3.7 | Estimate fault-tolerant resource requirements | ⏳ Pending |
| M3.8 | Extend to path-dependent options | ⏳ Pending |

### Deliverables

- `src/quantum/amplitude_estimation.py` - QAE implementations
- `notebooks/qae_speedup_analysis.ipynb` - Speedup analysis
- Comparison table: Classical MC vs QAE for various precisions

### Success Criteria

- [ ] Demonstrate O(1/N) error scaling in simulation
- [ ] Estimate crossover point for practical quantum advantage
- [ ] Characterize qubit requirements vs precision

---

## Phase 4: Heston & Exotic Options

**Status:** ⏳ Pending  
**Timeline:** Weeks 13-16

### Goals

1. Extend PINNs to stochastic volatility (Heston model)
2. Implement barrier and Asian option pricing
3. Handle path-dependent features in PINN framework
4. Explore quantum advantages for higher-dimensional PDEs

### Heston Model PDE

The Heston model is a 2D PDE in (S, v) with coupled dynamics:

```
∂V/∂t + ½σ²S²∂²V/∂S² + ρσνS∂²V/∂S∂v + ½ν²v∂²V/∂v² 
      + rS∂V/∂S + κ(θ-v)∂V/∂v - rV = 0
```

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M4.1 | Implement Heston PDE residual | ✅ Complete |
| M4.2 | Extend PINN to 3D input (S, v, t) | ⏳ Pending |
| M4.3 | Implement barrier option boundary conditions | ⏳ Pending |
| M4.4 | Implement Asian option averaging | ⏳ Pending |
| M4.5 | Train PINN on Heston model | ⏳ Pending |
| M4.6 | Implement jump-diffusion (Merton model) | ✅ Complete |
| M4.7 | Benchmark against Monte Carlo | ⏳ Pending |

### Deliverables

- `src/pde/heston.py` - Heston PDE implementation
- `src/pde/jump_diffusion.py` - Jump-diffusion models
- Extended pricing engine for exotic options
- Comparative analysis: PINN vs MC for exotics

### Success Criteria

- [ ] Heston PINN within 2% of Monte Carlo benchmark
- [ ] Barrier option pricing handles discontinuous payoffs
- [ ] Demonstrate PINN advantages for high-dimensional problems

---

## Phase 5: Tensor Network Methods

**Status:** ⏳ Pending  
**Timeline:** Weeks 17-20

### Goals

1. Implement Matrix Product States (MPS) for multi-asset options
2. Explore Tree Tensor Networks (TTN) for hierarchical problems
3. Compare tensor network methods with quantum approaches
4. Identify use cases where tensor networks excel

### Tensor Network Structures

```
Matrix Product State (MPS):
○─○─○─○─○─○  (linear chain, good for 1D correlations)

Tree Tensor Network (TTN):
    ○           (hierarchical, good for multi-scale)
   / \
  ○   ○
 / \ / \
○  ○ ○  ○
```

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M5.1 | Implement MPS contraction | ✅ Complete |
| M5.2 | Implement differentiable MPS layer | ✅ Complete |
| M5.3 | Implement Tree Tensor Network | ✅ Complete |
| M5.4 | Implement Tensor Train for multi-asset pricing | ✅ Complete |
| M5.5 | Train MPS-based PINN for basket options | ⏳ Pending |
| M5.6 | Compare expressivity: MPS vs VQC | ⏳ Pending |
| M5.7 | Benchmark computational scaling | ⏳ Pending |

### Deliverables

- `src/quantum/tensor_network.py` - MPS, TTN implementations
- Basket option pricer using tensor networks
- Scaling analysis: complexity vs number of assets

### Success Criteria

- [ ] Tensor networks scale polynomially with number of assets
- [ ] Achieve comparable accuracy to Monte Carlo for 10+ asset options
- [ ] Characterize bond dimension requirements

---

## Phase 6: Benchmarks & Analysis

**Status:** ⏳ Pending  
**Timeline:** Weeks 21-24

### Goals

1. Comprehensive comparison of all implemented methods
2. Publish benchmark results and analysis
3. Document best practices and recommendations
4. Identify promising directions for future research

### Comparison Dimensions

| Dimension | Methods Compared |
|-----------|-----------------|
| Accuracy | Classical PINN, Hybrid PINN, Analytical, MC, FD |
| Speed | Training time, inference time, convergence rate |
| Scalability | Dimension scaling, parameter efficiency |
| Greeks | Autodiff quality, numerical stability |
| Exotic Options | Barrier, Asian, basket, lookback |

### Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| M6.1 | Design comprehensive benchmark suite | ⏳ Pending |
| M6.2 | Run all methods on standard test cases | ⏳ Pending |
| M6.3 | Statistical analysis of results | ⏳ Pending |
| M6.4 | Generate publication-quality figures | ⏳ Pending |
| M6.5 | Write analysis report | ⏳ Pending |
| M6.6 | Document recommendations | ⏳ Pending |

### Deliverables

- `scripts/benchmark.py` - Comprehensive benchmark suite
- `docs/benchmark_results.md` - Results and analysis
- Publication-ready figures and tables
- Recommendations document

### Research Questions to Answer

1. **Does quantum circuit expressivity help for PDE solutions?**
   - Compare parameter efficiency of VQC vs MLP
   - Analyze function approximation quality

2. **Where does hybrid outperform pure classical?**
   - Identify problem regimes favoring quantum
   - Characterize breakeven points

3. **What's the practical crossover point for QAE advantage?**
   - Resource estimation for fault-tolerant QAE
   - Compare with optimized classical MC

4. **Can tensor networks achieve similar benefits classically?**
   - Compare MPS with VQC for expressivity
   - Analyze computational overhead

---

## Dependencies & Prerequisites

### Software Requirements

```
Python >= 3.10
PyTorch >= 2.0
PennyLane >= 0.32
NumPy, SciPy
Matplotlib, Plotly (visualization)
pytest (testing)
```

### Hardware Recommendations

- **Classical Training:** GPU recommended (CUDA support)
- **Quantum Simulation:** 8-12 qubits feasible on CPU
- **Tensor Networks:** Memory scales with bond dimension²

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Barren plateaus in VQC | Use layerwise training, local observables |
| PINN training instability | Adaptive loss weighting, learning rate scheduling |
| Quantum simulation overhead | Limit qubit count, use efficient backends |
| Tensor network scaling | Cap bond dimension, use compression |

---

## Timeline Summary

```
Weeks 1-4:   Phase 1 - Classical PINN ████████████████████
Weeks 5-8:   Phase 2 - Hybrid PINN   ░░░░░░░░░░░░░░░░░░░░
Weeks 9-12:  Phase 3 - QAE           ░░░░░░░░░░░░░░░░░░░░
Weeks 13-16: Phase 4 - Heston/Exotic ░░░░░░░░░░░░░░░░░░░░
Weeks 17-20: Phase 5 - Tensor Nets   ░░░░░░░░░░░░░░░░░░░░
Weeks 21-24: Phase 6 - Benchmarks    ░░░░░░░░░░░░░░░░░░░░

████ = Complete/In Progress
░░░░ = Pending
```
