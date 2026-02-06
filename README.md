# Quantum-Classical Hybrid PINNs for Derivatives Pricing

> Research exploration of physics-informed neural networks for financial PDEs, with experimental quantum-classical hybrid architectures.

[![Tests](https://img.shields.io/badge/tests-322%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.14-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![PennyLane](https://img.shields.io/badge/pennylane-0.44-blueviolet)]()

---

## Overview

This project uses **physics-informed neural networks (PINNs)** to solve partial differential equations in derivatives pricing, with experimental integration of **variational quantum circuits (VQCs)**.

### What This Project Demonstrates

| Capability | Why It Matters |
|------------|----------------|
| **6D basket option pricing** | Solves curse of dimensionality (FD infeasible) |
| **Volatility surface calibration** | Real quant workflow: inverse problems |
| **Jump-diffusion & stochastic vol** | Beyond textbook Black-Scholes |
| **Greeks via autodiff** | Exact sensitivities, no numerical noise |
| **Comprehensive testing** | 322 tests, production-quality code |

### Honest Assessment

**What works well:**
- High-dimensional pricing (5-asset basket, 6D PDE)
- Volatility calibration with arbitrage constraints
- Clean architecture with proper testing
- Mathematical documentation (944 lines of theory)

**What's still research:**
- Quantum-hybrid component shows promise on simple problems but doesn't yet outperform well-tuned classical on production pricing tasks
- Current quantum results: ~22% relative error (market tolerance is <1%)
- Exploring when/why quantum expressivity might help

---

## Key Results

### 1. High-Dimensional Pricing (5-Asset Basket)

| Method | Feasibility | Memory | Greeks |
|--------|-------------|--------|--------|
| Finite Difference | No -- 10^12 grid points | Impossible | Numerical noise |
| Monte Carlo | Yes, but slow | Moderate | Pathwise |
| **PINN** | **Yes** -- 15K collocation points | ~3 GB | **Exact (autodiff)** |

**Result**: Final loss 1.40, 3.9% error vs Monte Carlo at S₀.

### 2. Volatility Calibration (Inverse Problem)

| Metric | Value |
|--------|-------|
| Price fit | 2.22% mean relative error |
| Vol surface recovery | 4.51% error |
| Training | ~2 min (3000 epochs) |

**Real quant application**: Recovers local volatility σ(K,T) from market prices.

### 3. Advanced Models

| Model | PDE Type | Dimension | Key Feature |
|-------|----------|-----------|-------------|
| Merton | PIDE | 2D | Poisson jumps, Gauss-Hermite quadrature |
| Heston | PDE | 3D | Stochastic variance, vol smile |
| American | Free boundary | 2D | Early exercise, penalty method |

### 4. Quantum-Hybrid (Experimental)

| Observation | Status |
|-------------|--------|
| VQC trains and produces gradients | Yes |
| Hybrid beats poorly-tuned classical | Yes |
| Hybrid beats well-tuned classical | Needs ablation |
| Practical pricing accuracy (<1% error) | Not yet |

**Honest conclusion**: The quantum component is research-stage. We're investigating expressivity and optimization landscape differences, not claiming advantage.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/rylanmalarchick/quantum-derivatives-trader.git
cd quantum-derivatives-trader
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (322 passing)
pytest tests/ -v

# Train basket PINN (high-dimensional - the strong result)
python scripts/train_basket.py --epochs 5000 --n_interior 15000 --eval

# Train volatility calibration (inverse problem)
python scripts/train_calibration.py --epochs 2000

# Train advanced models
python scripts/train_merton.py --epochs 3000 --eval   # Jump-diffusion
python scripts/train_heston.py --epochs 3000 --eval   # Stochastic vol
python scripts/train_american.py --epochs 3000 --eval # Early exercise

# Quantum-hybrid (experimental)
python scripts/train_hybrid.py --epochs 300 --n-qubits 4 --n-layers 2
```

---

## Project Structure

```
quantum-derivatives-trader/
├── src/
│   ├── pde/                 # PDE definitions (6 types)
│   │   ├── black_scholes.py # 1D European option
│   │   ├── basket.py        # N-asset basket (6D)
│   │   ├── dupire.py        # Local volatility calibration
│   │   ├── merton.py        # Jump-diffusion (PIDE)
│   │   ├── heston.py        # Stochastic volatility (3D)
│   │   └── american.py      # Free boundary problem
│   ├── classical/           # PINN architectures
│   ├── quantum/             # VQC integration (experimental)
│   ├── pricing/             # MC, FD, analytical engines
│   └── validation/          # Greeks validation
├── tests/                   # 322 comprehensive tests
├── scripts/                 # Training and benchmarking
├── notebooks/               # Analysis (7 notebooks)
└── docs/
    └── theory.md            # 944 lines of mathematical derivations
```

---

## Mathematical Background

### Forward Problem: Multi-Asset Black-Scholes

$$\frac{\partial V}{\partial t} + \sum_i rS_i\frac{\partial V}{\partial S_i} + \frac{1}{2}\sum_{i,j} \rho_{ij}\sigma_i\sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} - rV = 0$$

### Inverse Problem: Dupire Calibration

Given market prices $C(K,T)$, recover local volatility $\sigma(K,T)$:

$$\sigma^2(K,T) = \frac{2\left(\frac{\partial C}{\partial T} + rK\frac{\partial C}{\partial K}\right)}{K^2 \frac{\partial^2 C}{\partial K^2}}$$

### Jump-Diffusion (Merton)

$$\frac{\partial V}{\partial t} + (r-\lambda\kappa)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV + \lambda\int_0^\infty [V(SJ,t) - V(S,t)]g(J)dJ = 0$$

**See [docs/theory.md](docs/theory.md) for complete derivations (11 sections, 944 lines).**

---

## Why This Matters

### For Quant Finance

1. **High-dimensional pricing**: Basket options, rainbow options require methods that scale beyond 3D
2. **Model calibration**: Every desk calibrates daily; PINNs give smooth, arbitrage-free surfaces
3. **Greeks computation**: Autodiff gives exact Δ, Γ, Θ, ν, cross-gammas—no numerical noise

### For Research

1. **Honest quantum exploration**: Testing VQC expressivity on real problems, not toy examples
2. **Reproducible benchmarks**: Proper baselines, not cherry-picked comparisons
3. **Clean implementation**: 322 tests, documented theory, readable code

---

## Testing

```bash
pytest tests/ -v --tb=short  # 322 tests
```

| Category | Tests | Coverage |
|----------|-------|----------|
| PDE residuals | 45 | Physics constraints, boundary conditions |
| PINN models | 68 | Forward pass, gradients, training |
| Quantum circuits | 32 | Output ranges, parameter updates |
| Greeks validation | 33 | Analytical vs autodiff |
| Basket/calibration | 89 | MC validation, vol recovery |
| Advanced models | 55 | Merton, Heston, American |

---

## Roadmap

- [x] Classical PINN baseline
- [x] High-dimensional basket (5 assets, 6D)
- [x] Volatility calibration (inverse problem)
- [x] Jump-diffusion (Merton PIDE)
- [x] Stochastic volatility (Heston 3D)
- [x] American options (free boundary)
- [x] Greeks validation module
- [x] Speed benchmarks
- [ ] Ablation: quantum vs matched classical baseline
- [ ] Barrier options (path-dependent)
- [ ] Hardware experiments (IBM/IonQ)

---

## References

- Raissi et al. "Physics-informed neural networks" (2019). [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)
- Stamatopoulos et al. "Option Pricing using Quantum Computers" (2020). [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)
- Dupire. "Pricing with a Smile" (Risk, 1994)
- Schuld & Petruccione. "Machine Learning with Quantum Computers" (Springer, 2021)

---

## License

MIT License. See [LICENSE](LICENSE).

---

*"The goal is not to prove quantum advantage, but to rigorously investigate when it might exist—while solving real problems with classical methods that work today."*
