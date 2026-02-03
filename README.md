# Quantum-Classical Hybrid PINNs for Derivatives Pricing

A research project exploring the intersection of quantum computing, physics-informed neural networks, and financial derivatives pricing.

## Overview

This project implements and compares several approaches to options pricing:

1. **Classical PINNs**: Neural networks that encode the Black-Scholes PDE as a physics constraint
2. **Hybrid Quantum-Classical PINNs**: Replace classical NNs with variational quantum circuits
3. **Quantum Amplitude Estimation**: Quadratic speedup for Monte Carlo pricing
4. **Tensor Networks**: Quantum-inspired methods for high-dimensional problems

## Project Status

See [PHASES.md](docs/PHASES.md) for detailed implementation roadmap.

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | 🔄 In Progress | Classical PINN for Black-Scholes |
| 2 | ⏳ Pending | Hybrid Quantum-Classical PINN |
| 3 | ⏳ Pending | Quantum Amplitude Estimation |
| 4 | ⏳ Pending | Heston & Exotic Options |
| 5 | ⏳ Pending | Tensor Network Methods |
| 6 | ⏳ Pending | Benchmarks & Analysis |

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run classical PINN training
python scripts/train_classical.py

# Run hybrid quantum-classical training
python scripts/train_hybrid.py

# Run benchmarks
python scripts/benchmark.py
```

## Project Structure

```
quantum-pinn-derivatives/
├── src/
│   ├── pde/              # PDE definitions (Black-Scholes, Heston, Jump-Diffusion)
│   ├── classical/        # Classical PINN implementation
│   ├── quantum/          # Quantum circuits, hybrid models, QAE
│   ├── pricing/          # Pricing engines (analytical, MC, FD, PINN)
│   ├── data/             # Data generation and collocation
│   └── utils/            # Greeks, visualization
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Unit tests
├── scripts/              # Training and benchmark scripts
├── docs/                 # Documentation
└── ocaml/                # Optional high-performance numerical core
```

## Key Concepts

### Physics-Informed Neural Networks (PINNs)

Instead of learning from labeled data, PINNs learn by satisfying physical laws (PDEs). For options pricing:

```
Loss = λ₁·L_PDE + λ₂·L_boundary + λ₃·L_terminal
```

Where L_PDE enforces the Black-Scholes equation at collocation points.

### Quantum Function Approximation

Replace the classical neural network with a variational quantum circuit:

```
V_θ(S,t) = f_post(⟨ψ(S,t)| U(θ)† M U(θ) |ψ(S,t)⟩)
```

This explores whether quantum expressivity provides advantages for PDE solutions.

### Quantum Amplitude Estimation

For Monte Carlo pricing, QAE achieves O(1/N) error vs classical O(1/√N):
- Encode price distribution in quantum superposition
- Encode payoff as amplitude
- Use phase estimation to extract expectation

## Research Questions

1. Does quantum circuit expressivity help for PDE solutions?
2. Where does hybrid outperform pure classical?
3. What's the practical crossover point for QAE advantage?
4. Can tensor networks achieve similar benefits classically?

## References

- Raissi et al., "Physics-informed neural networks" (2019)
- Stamatopoulos et al., "Option Pricing using Quantum Computers" (2020)
- Schuld & Petruccione, "Machine Learning with Quantum Computers" (2021)

## License

MIT
