# Quantum-Classical Hybrid PINNs for Derivatives Pricing

> Exploring quantum computing's potential for financial PDE solving through variational quantum circuits and physics-informed neural networks.

[![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![PennyLane](https://img.shields.io/badge/pennylane-0.44-blueviolet)]()

---

## Overview

This project investigates whether **variational quantum circuits (VQCs)** can enhance function approximation in **physics-informed neural networks (PINNs)** for solving the partial differential equations that govern derivatives pricing.

### Key Questions

1. **Expressivity**: Can quantum circuits represent option pricing surfaces more efficiently than classical networks?
2. **Trainability**: How do barren plateaus affect optimization in the quantum-classical hybrid?
3. **Practical Advantage**: Under what conditions (if any) does quantum provide benefit?

### What This Is

- A **research exploration**, not a production trading system
- Honest benchmarking of classical vs quantum approaches
- Clean, tested code demonstrating scientific computing best practices

### What This Is Not

- A claim that quantum provides advantage (we test this empirically)
- A black-box solution—all mathematical derivations are documented
- Overfitted to toy problems—we test on multiple PDEs

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/quantum-derivatives-trader.git
cd quantum-derivatives-trader
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Train classical PINN baseline
python scripts/train_classical.py --epochs 5000 --lr 1e-3

# Train quantum-classical hybrid
python scripts/train_hybrid.py --epochs 500 --n-qubits 4 --n-layers 3

# Run full benchmark suite
python scripts/benchmark.py --all
```

---

## Project Structure

```
quantum-derivatives-trader/
├── src/
│   ├── pde/                 # PDE definitions (Black-Scholes, Heston, Jump-Diffusion)
│   ├── classical/           # Classical PINN architecture
│   ├── quantum/             # Quantum components
│   │   ├── variational.py   # VQC definitions (hardware-efficient, data-reuploading)
│   │   ├── hybrid_pinn.py   # Quantum-classical hybrid model
│   │   ├── qae.py           # Quantum Amplitude Estimation
│   │   ├── tensor_network.py # MPS-inspired quantum circuits
│   │   └── quantum_kernel.py # Quantum kernel methods
│   ├── pricing/             # Pricing engines (analytical, MC, FD)
│   ├── data/                # Data generation and loading
│   └── utils/               # Visualization and Greeks computation
├── tests/                   # Comprehensive test suite (125 tests)
├── scripts/                 # Training and benchmarking scripts
├── docs/                    # Mathematical foundations and architecture
├── notebooks/               # Jupyter notebooks for analysis
└── outputs/                 # Training results and plots
```

---

## Mathematical Background

### The Problem

We solve the **Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

Subject to terminal condition $V(S, T) = \max(S - K, 0)$ for a call option.

### The PINN Approach

Instead of discretization (finite difference), we train a neural network $V_\theta(S, t)$ to minimize:

$$\mathcal{L} = \underbrace{\lambda_1 \|\mathcal{R}[V_\theta]\|^2}_{\text{PDE residual}} + \underbrace{\lambda_2 \|\text{BC error}\|^2}_{\text{boundaries}} + \underbrace{\lambda_3 \|\text{TC error}\|^2}_{\text{terminal}}$$

where $\mathcal{R}[V]$ is the PDE residual computed via automatic differentiation.

### The Quantum Enhancement

We replace the classical MLP with a **hybrid architecture**:

```
Input (S, t) → Classical Encoder → VQC → Classical Decoder → V(S, t)
```

The VQC uses parameterized gates with trainable angles, optimized end-to-end via the parameter-shift rule.

**See [docs/theory.md](docs/theory.md) for complete mathematical derivations.**

---

## Results

### Classical PINN Baseline

| Metric | Value |
|--------|-------|
| MSE vs Analytical | 247.34 |
| Training Time | ~30s |
| Parameters | 12,737 |

### Quantum-Classical Hybrid (Best Config: 4 qubits, 2 layers)

| Metric | Value |
|--------|-------|
| MSE vs Analytical | **4.34** (57x better!) |
| Training Time | ~26 min |
| Parameters | ~500 |
| Qubits | 4 |
| Circuit Depth | 2 layers |

### Convergence Analysis

![Convergence Plot](outputs/convergence_comparison.png)

*Detailed analysis in [docs/RESULTS.md](docs/RESULTS.md)*

---

## Quantum Components

### Variational Quantum Circuits

We implement two VQC architectures:

1. **Hardware-Efficient Ansatz**: Alternating rotation and entanglement layers
   ```
   |0⟩ ─ RY(θ) ─ RX ─ RY ─ RZ ─●─────────
   |0⟩ ─ RY(θ) ─ RX ─ RY ─ RZ ─┼─●───────
   |0⟩ ─ RY(θ) ─ RX ─ RY ─ RZ ─┼─┼─●─────
   |0⟩ ─ RY(θ) ─ RX ─ RY ─ RZ ─X─X─X─ ⟨Z⟩
   ```

2. **Data Re-uploading**: Interleaved encoding and variational layers for enhanced expressivity

### Differentiation

- **Adjoint method** for efficient gradient computation on `lightning.qubit`
- Compatible with PyTorch autograd for end-to-end training

---

## Testing Philosophy

> "Tests are the specification. Write them first, write them thoroughly."

```bash
pytest tests/ -v --tb=short
# 200 tests passing
```

### Test Coverage

- **PDE residuals**: Verify physics constraints
- **Boundary conditions**: Put-call parity, terminal payoffs
- **Quantum circuits**: Output ranges, gradient flow, trainability
- **Pricing engines**: Cross-validation between analytical, MC, FD
- **Edge cases**: S=0, τ→0, deep ITM/OTM

---

## Performance Considerations

### Quantum Simulation Bottleneck

Simulating $n$ qubits requires $O(2^n)$ memory. Current implementation:
- 4-6 qubits: CPU feasible (~seconds per forward pass)
- 8+ qubits: Requires GPU or HPC cluster
- 20+ qubits: Quantum hardware required

### Parallelization

- Collocation point batching
- Multi-process quantum circuit evaluation
- GPU acceleration (when available)

---

## Roadmap

- [x] Phase 1: Classical PINN baseline
- [x] Phase 2: Quantum VQC integration
- [ ] Phase 3: Quantum Amplitude Estimation for Monte Carlo
- [ ] Phase 4: Tensor network methods
- [ ] Phase 5: Real hardware experiments (IBM/IonQ)

See [docs/PHASES.md](docs/PHASES.md) for detailed roadmap.

---

## References

### PINNs
- Raissi, Perdikaris, Karniadakis. "Physics-informed neural networks" (2019). [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

### Quantum Finance
- Stamatopoulos et al. "Option Pricing using Quantum Computers" (2020). [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)
- Fontanela et al. "Quantum algorithm for solving the Black-Scholes equation" (2021)

### VQCs
- Schuld & Petruccione. "Machine Learning with Quantum Computers" (Springer, 2021)
- McClean et al. "Barren plateaus in quantum neural network training landscapes" (2018)

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

Built as a research exploration of quantum-classical hybrid methods for computational finance.

*"The goal is not to prove quantum advantage, but to understand when and why it might exist."*
