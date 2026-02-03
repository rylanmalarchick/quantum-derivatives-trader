# Quantum-Classical Hybrid PINNs for Derivatives Pricing

> Exploring quantum computing's potential for financial PDE solving through variational quantum circuits and physics-informed neural networks.

[![Tests](https://img.shields.io/badge/tests-240%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![PennyLane](https://img.shields.io/badge/pennylane-0.44-blueviolet)]()

---

## Overview

This project investigates whether **variational quantum circuits (VQCs)** can enhance function approximation in **physics-informed neural networks (PINNs)** for solving the partial differential equations that govern derivatives pricing.

### Key Capabilities

| Problem | Type | Dimension | Status |
|---------|------|-----------|--------|
| Black-Scholes European Call | Forward PDE | 2D | ✅ Working |
| Quantum-Hybrid PINN | Forward PDE | 2D | ✅ 57x better than classical |
| 5-Asset Basket Option | Forward PDE | **6D** | ✅ Beats curse of dimensionality |
| Volatility Calibration | **Inverse Problem** | 2D | ✅ <5% vol recovery error |

### What This Is

- A **research exploration**, not a production trading system
- Honest benchmarking of classical vs quantum approaches
- Clean, tested code demonstrating scientific computing best practices
- **Real quant problems** - high-dimensional pricing, model calibration

### What This Is Not

- A claim that quantum provides advantage (we test this empirically)
- A black-box solution—all mathematical derivations are documented
- Overfitted to toy problems—we solve genuine multi-asset derivatives

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/rylanmalarchick/quantum-derivatives-trader.git
cd quantum-derivatives-trader
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v  # 240 tests

# Train classical PINN baseline (Black-Scholes)
python scripts/train_classical.py --epochs 5000 --lr 1e-3

# Train quantum-classical hybrid
python scripts/train_hybrid.py --epochs 300 --n-qubits 4 --n-layers 2

# Train 5-asset basket PINN (high-dimensional)
python scripts/train_basket.py --epochs 5000 --n_interior 15000 --eval

# Train volatility calibration (inverse problem)
python scripts/train_calibration.py --epochs 2000
```

---

## Project Structure

```
quantum-derivatives-trader/
├── src/
│   ├── pde/                 # PDE definitions
│   │   ├── black_scholes.py # 1D European option PDE
│   │   ├── basket.py        # N-asset basket option PDE
│   │   └── dupire.py        # Local volatility model
│   ├── classical/           # Classical PINN architectures
│   │   ├── pinn.py          # Standard MLP PINN
│   │   ├── pinn_basket.py   # High-dimensional basket PINN
│   │   └── pinn_calibration.py # Inverse problem PINN
│   ├── quantum/             # Quantum components
│   │   ├── variational.py   # VQC definitions
│   │   ├── hybrid_pinn.py   # Quantum-classical hybrid
│   │   └── qae.py           # Quantum Amplitude Estimation
│   ├── pricing/             # Pricing engines (analytical, MC, FD)
│   └── utils/               # Visualization and Greeks
├── tests/                   # 240 comprehensive tests
├── scripts/                 # Training scripts
├── notebooks/               # Analysis notebooks
│   ├── 05_basket_analysis.py      # 5-asset results
│   └── 06_calibration_analysis.py # Vol surface calibration
└── docs/                    # Mathematical documentation
```

---

## Results Summary

### 1. Classical vs Quantum-Hybrid PINN (Black-Scholes)

| Model | MSE | Parameters | Training Time |
|-------|-----|------------|---------------|
| Classical MLP | 247.34 | 12,737 | ~30s |
| **Hybrid 4q/2L** | **4.34** | ~500 | ~26 min |
| FD/MC (reference) | ~0.001 | N/A | <1s |

**Key finding**: Quantum-hybrid achieves **57x better MSE** with **25x fewer parameters**.

### 2. 5-Asset Basket Option (High-Dimensional)

| Aspect | Classical FD | PINN |
|--------|-------------|------|
| Dimension | 6D (5 assets + time) | 6D |
| Grid points needed | 10 billion (100^5) | 15,000 |
| Memory | Impossible | ~3 GB |
| Greeks | Numerical noise | Exact (autodiff) |

**Key finding**: PINNs **solve the curse of dimensionality**. FD is infeasible for 5+ assets.

### 3. Volatility Calibration (Inverse Problem)

| Metric | Value |
|--------|-------|
| Price fit (relative) | 1.45% |
| Vol recovery error | 4.70% |
| Training time | ~30s (500 epochs) |

**Key finding**: PINN recovers volatility surface from option prices—**real quant workflow**.

---

## Mathematical Background

### Forward Problem: Black-Scholes PDE

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

### Forward Problem: Multi-Asset Basket (N-dimensional)

$$\frac{\partial V}{\partial t} + \sum_i rS_i\frac{\partial V}{\partial S_i} + \frac{1}{2}\sum_{i,j} \rho_{ij}\sigma_i\sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} - rV = 0$$

### Inverse Problem: Dupire Calibration

Given market prices $C(K,T)$, find local volatility $\sigma(K,T)$ such that:

$$\sigma^2(K,T) = \frac{2\left(\frac{\partial C}{\partial T} + rK\frac{\partial C}{\partial K}\right)}{K^2 \frac{\partial^2 C}{\partial K^2}}$$

**See [docs/theory.md](docs/theory.md) for complete derivations.**

---

## Why This Matters for Quant Finance

### 1. High-Dimensional Pricing
- Basket options, rainbow options, path-dependent exotics
- Monte Carlo is slow, FD is infeasible
- PINNs scale to 10+ dimensions

### 2. Model Calibration
- Every trading desk calibrates models daily
- Standard approaches use optimization + analytical gradients
- PINNs offer smooth, arbitrage-free surfaces automatically

### 3. Greeks Computation
- Automatic differentiation gives exact derivatives
- No numerical differentiation noise
- All Greeks (delta, gamma, vega, cross-gammas) for free

---

## Testing Philosophy

> "Tests are the specification. Write them first, write them thoroughly."

```bash
pytest tests/ -v --tb=short
# 240 tests passing
```

### Test Coverage

- **PDE residuals**: Physics constraints verified
- **Boundary conditions**: Put-call parity, terminal payoffs
- **Quantum circuits**: Output ranges, gradient flow, trainability
- **Basket pricing**: MC validation, Greeks accuracy
- **Calibration**: Vol recovery, arbitrage constraints
- **Edge cases**: S=0, τ→0, deep ITM/OTM

---

## Roadmap

- [x] Phase 1: Classical PINN baseline
- [x] Phase 2: Quantum VQC integration (57x improvement)
- [x] Phase 3A: High-dimensional basket options (5 assets, 6D)
- [x] Phase 3B: Volatility surface calibration (inverse problem)
- [ ] Phase 4: Hybrid quantum for high-dimensional problems
- [ ] Phase 5: Jump-diffusion and stochastic volatility
- [ ] Phase 6: Real hardware experiments (IBM/IonQ)

---

## References

### PINNs
- Raissi, Perdikaris, Karniadakis. "Physics-informed neural networks" (2019). [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

### Quantum Finance
- Stamatopoulos et al. "Option Pricing using Quantum Computers" (2020). [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)

### Local Volatility
- Dupire. "Pricing with a Smile" (Risk, 1994)

### VQCs
- Schuld & Petruccione. "Machine Learning with Quantum Computers" (Springer, 2021)

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

Built as a research exploration of quantum-classical hybrid methods for computational finance.

*"The goal is not to prove quantum advantage, but to understand when and why it might exist—and solve real problems along the way."*
