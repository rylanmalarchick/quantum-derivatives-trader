# Experimental Results

> This document tracks experimental results as we develop the quantum-classical hybrid PINN.

---

## Experiment Log

### Experiment 1: Classical PINN Baseline

**Date**: 2025-02-02  
**Commit**: `59c78db`

#### Configuration

| Parameter | Value |
|-----------|-------|
| Network | MLP (4 hidden layers, 64 units) |
| Activation | Tanh |
| Epochs | 1000 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Interior Points | 1000 |
| Boundary Points | 200 |
| Terminal Points | 200 |
| Loss Weights (PDE/BC/TC) | 1.0 / 10.0 / 10.0 |

#### Black-Scholes Parameters

| Parameter | Value |
|-----------|-------|
| Strike (K) | 100 |
| Maturity (T) | 1.0 year |
| Risk-free rate (r) | 5% |
| Volatility (σ) | 20% |
| Spot range | [0, 200] |

#### Results

| Metric | Value | Notes |
|--------|-------|-------|
| **MSE** | 247.34 | vs analytical Black-Scholes |
| **MAE** | 8.25 | |
| **Max Abs Error** | 48.99 | Likely at boundaries |
| **Mean Rel Error** | 27.66% | Room for improvement |
| **RMSE** | 15.73 | |
| **Training Time** | ~30s | CPU only |
| **Parameters** | 12,737 | |

#### Training Dynamics

```
Epoch    0: total=22071.70, pde=0.01, bc=0.04, ic=22071.65
Epoch  100: total=11115.68, pde=18.99, bc=40.14, ic=11056.55
Epoch  500: total=3329.43,  pde=27.78, bc=0.22, ic=3301.43
Epoch 1000: total=1665.28,  pde=3.09,  bc=0.06, ic=1662.13
```

**Observations**:
- Initial condition (terminal payoff) dominates the loss
- PDE residual decreases steadily
- Boundary conditions quickly satisfied
- Loss still decreasing at epoch 1000—longer training may help

#### Artifacts

- Training curves: `outputs/classical/20260202_210742/training_history.png`
- Comparison plot: `outputs/classical/20260202_210742/comparison.png`
- Price surface: `outputs/classical/20260202_210742/price_surface.png`

---

### Experiment 2: Quantum-Classical Hybrid

**Date**: 2025-02-02  
**Status**: IN PROGRESS

#### Configuration

| Parameter | Value |
|-----------|-------|
| Qubits | 4 |
| VQC Layers | 3 |
| Circuit Type | Hardware-efficient |
| Diff Method | Adjoint |
| Epochs | 500 |
| Learning Rate | 1e-3 |

#### Results

| Metric | Value | Notes |
|--------|-------|-------|
| **MSE** | TBD | |
| **Training Time** | ~hours | Quantum simulation is expensive |
| **Quantum Parameters** | 36 | 4 qubits × 3 layers × 3 rotations |
| **Total Parameters** | ~5000 | Including classical encoder/decoder |

---

## Comparison Summary

| Model | MSE | MAE | Training Time | Parameters |
|-------|-----|-----|---------------|------------|
| Classical PINN | 247.34 | 8.25 | ~30s | 12,737 |
| Hybrid Quantum | TBD | TBD | TBD | ~5,000 |
| Finite Difference | <0.5 | <0.5 | ~1s | N/A |
| Monte Carlo (100k) | ~0.1 | ~0.2 | ~2s | N/A |

---

## Observations & Insights

### Why Classical PINN Has High Error

1. **Loss balancing**: Terminal condition dominates; may need adaptive weighting
2. **Collocation sampling**: Uniform sampling may miss important regions (near strike)
3. **Training duration**: 1000 epochs may be insufficient
4. **Network capacity**: 4×64 may be undersized for the pricing surface

### Potential Improvements

1. **Adaptive loss weighting** (learn λ values during training)
2. **Importance sampling** for collocation points (more near ATM)
3. **Curriculum learning** (start with simple, increase complexity)
4. **Larger network** or **Fourier features** for high-frequency content
5. **Longer training** with learning rate scheduling

### Quantum-Specific Considerations

1. **Barren plateaus**: Random initialization may lead to vanishing gradients
2. **Expressivity vs trainability**: More layers = more expressive but harder to train
3. **Simulation overhead**: Each forward pass is O(2^n), limiting qubit count
4. **Adjoint differentiation**: Enables gradients but constrains circuit structure

---

## Next Experiments

- [ ] Classical PINN with 5000 epochs and LR scheduling
- [ ] Classical PINN with adaptive loss weighting
- [ ] Hybrid with 6 qubits (compare expressivity)
- [ ] Hybrid with data-reuploading circuit
- [ ] Convergence analysis: error vs collocation points
- [ ] Convergence analysis: error vs VQC depth

---

## Reproducibility

All experiments can be reproduced with:

```bash
# Classical baseline
python scripts/train_classical.py --epochs 1000 --lr 1e-3 --seed 42

# Quantum hybrid
python scripts/train_hybrid.py --epochs 500 --n-qubits 4 --n-layers 3 --seed 42
```

Random seeds are fixed for reproducibility. Hardware: CPU-only simulation.
