# Theoretical Foundations

This document covers the mathematical and physical foundations underlying the quantum-classical hybrid approach to derivatives pricing.

---

## 1. Black-Scholes PDE

### 1.1 Derivation

The Black-Scholes equation describes the evolution of an option price $V(S, t)$ under the following assumptions:
- The underlying follows geometric Brownian motion: $dS = \mu S \, dt + \sigma S \, dW$
- No arbitrage, continuous trading, no dividends
- Constant volatility $\sigma$ and risk-free rate $r$

Using Itô's lemma and delta hedging, we obtain the **Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

This is a backward parabolic PDE solved from the terminal condition (option expiry) back to the present.

### 1.2 Boundary Conditions

For a **European call** option with strike $K$ and maturity $T$:

**Terminal condition (payoff at expiry):**
$$V(S, T) = \max(S - K, 0) = (S - K)^+$$

**Lower boundary ($S = 0$):**
$$V(0, t) = 0$$

**Upper boundary ($S \to \infty$):**
$$V(S, t) \approx S - K e^{-r(T-t)} \quad \text{as } S \to \infty$$

For a **European put**:
$$V(S, T) = \max(K - S, 0) = (K - S)^+$$
$$V(0, t) = K e^{-r(T-t)}$$
$$V(S, t) \to 0 \quad \text{as } S \to \infty$$

### 1.3 Analytical Solution

The Black-Scholes formula for a European call:

$$C(S, t) = S \cdot N(d_1) - K e^{-r\tau} \cdot N(d_2)$$

where $\tau = T - t$ is time to maturity, and:

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}$$

$N(\cdot)$ is the standard normal CDF.

### 1.4 The Greeks

Option sensitivities derived from the solution:

| Greek | Definition | Interpretation |
|-------|------------|----------------|
| Delta | $\Delta = \frac{\partial V}{\partial S}$ | Hedge ratio |
| Gamma | $\Gamma = \frac{\partial^2 V}{\partial S^2}$ | Delta sensitivity |
| Theta | $\Theta = \frac{\partial V}{\partial t}$ | Time decay |
| Vega | $\mathcal{V} = \frac{\partial V}{\partial \sigma}$ | Volatility sensitivity |
| Rho | $\rho = \frac{\partial V}{\partial r}$ | Interest rate sensitivity |

---

## 2. Physics-Informed Neural Networks (PINNs)

### 2.1 Overview

PINNs [1] encode physical laws directly into the neural network loss function, enabling learning from physics rather than labeled data.

For option pricing, the neural network $\hat{V}_\theta(S, t)$ approximates the true solution $V(S, t)$.

### 2.2 Loss Function Components

The total loss is a weighted sum of three components:

$$\mathcal{L}(\theta) = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{BC}} + \lambda_3 \mathcal{L}_{\text{IC}}$$

**PDE Residual Loss** (physics constraint):

Sample collocation points $(S_i, t_i)$ in the interior domain and minimize:

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| \frac{\partial \hat{V}}{\partial t} + \frac{1}{2}\sigma^2 S_i^2 \frac{\partial^2 \hat{V}}{\partial S^2} + r S_i \frac{\partial \hat{V}}{\partial S} - r\hat{V} \right|^2$$

Derivatives are computed via automatic differentiation.

**Boundary Condition Loss:**

$$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{bc}}} \sum_{j=1}^{N_{\text{bc}}} \left| \hat{V}(0, t_j) - 0 \right|^2 + \frac{1}{N_{\text{bc}}} \sum_{k=1}^{N_{\text{bc}}} \left| \hat{V}(S_{\max}, t_k) - (S_{\max} - Ke^{-r(T-t_k)}) \right|^2$$

**Initial/Terminal Condition Loss** (payoff at maturity):

$$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{ic}}} \sum_{l=1}^{N_{\text{ic}}} \left| \hat{V}(S_l, T) - \max(S_l - K, 0) \right|^2$$

### 2.3 Training Considerations

1. **Collocation Point Sampling**: Fresh random points each epoch improve generalization
2. **Loss Weighting**: $\lambda_1, \lambda_2, \lambda_3$ balance competing objectives
3. **Input Normalization**: Scale $(S, t)$ to $[0, 1]$ for stable training
4. **Network Architecture**: Tanh activation preferred for smoothness

### 2.4 Advantages for Option Pricing

- **Mesh-free**: No grid discretization required
- **Continuous solution**: Greeks computed via autodiff, not finite differences
- **Generalization**: Single network valid for all $(S, t)$ in domain
- **Transfer learning**: Pretrained networks adapt to new parameters

---

## 3. Variational Quantum Circuits (VQCs)

### 3.1 Parameterized Quantum Circuits

A VQC implements a unitary transformation $U(\theta)$ parameterized by angles $\theta$:

$$|\psi(\theta)\rangle = U(\theta)|0\rangle^{\otimes n}$$

For function approximation, we encode inputs $x$ and measure an observable:

$$f_\theta(x) = \langle\psi(x, \theta)| \hat{O} |\psi(x, \theta)\rangle$$

### 3.2 Input Encoding

**Angle Encoding:**

Map classical input $x \in [0, 1]$ to rotation angle:

$$|x\rangle = R_Y(\pi x)|0\rangle = \cos\left(\frac{\pi x}{2}\right)|0\rangle + \sin\left(\frac{\pi x}{2}\right)|1\rangle$$

For 2D inputs $(S, t)$, encode on separate qubits or use sequential rotations.

**Data Re-uploading [2]:**

Interleave data encoding with variational layers for enhanced expressivity:

$$U(\theta, x) = \prod_{l=1}^{L} W_l(\theta_l) S(x)$$

where $S(x)$ encodes data and $W_l(\theta_l)$ are trainable unitaries.

### 3.3 Hardware-Efficient Ansatz

A practical circuit structure with single-qubit rotations and nearest-neighbor entanglement:

```
     ┌───────┐     ┌───────┐
q_0: ┤ Ry(x) ├──■──┤ Rx(θ) ├──■──
     ├───────┤┌─┴─┐├───────┤┌─┴─┐
q_1: ┤ Ry(t) ├┤ X ├┤ Ry(θ) ├┤ X ├
     ├───────┤├───┤├───────┤├───┤
q_2: ┤ Ry(x) ├┤ X ├┤ Rz(θ) ├┤ X ├
     └───────┘└───┘└───────┘└───┘
```

### 3.4 Expressivity and Trainability

**Expressivity**: VQCs can approximate any continuous function with sufficient depth [3]. The question is efficiency.

**Barren Plateaus**: Deep random circuits suffer from exponentially vanishing gradients [4]:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_j}\right] \leq O(2^{-n})$$

Mitigation strategies:
- Shallow circuits with structured ansätze
- Layer-wise training
- Initialization near identity

### 3.5 Hybrid Architecture for PINNs

Replace the classical MLP with a quantum layer:

$$\hat{V}_\theta(S, t) = g_{\text{post}}\left( \langle\psi(S, t, \theta)| Z |\psi(S, t, \theta)\rangle \right)$$

where $g_{\text{post}}$ is a classical postprocessing layer for output scaling.

---

## 4. Quantum Amplitude Estimation (QAE)

### 4.1 Monte Carlo Pricing

Classical Monte Carlo estimates option price as:

$$\hat{V} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^N \text{payoff}(S_T^{(i)})$$

The standard error scales as $O(1/\sqrt{N})$, requiring $N = O(1/\epsilon^2)$ samples for error $\epsilon$.

### 4.2 Quantum Speedup

Quantum Amplitude Estimation [5] achieves error $\epsilon$ with $O(1/\epsilon)$ queries, a **quadratic speedup**.

The algorithm:
1. Prepare superposition encoding price distribution
2. Encode payoff as amplitude
3. Use phase estimation to extract the amplitude

### 4.3 State Preparation

Encode the risk-neutral distribution $p(S_T)$ in quantum amplitudes:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \sqrt{p_i} |i\rangle$$

where $|i\rangle$ represents discretized price bin $S_i$.

For log-normal: $\ln S_T \sim \mathcal{N}\left(\ln S_0 + (r - \frac{\sigma^2}{2})T, \sigma^2 T\right)$

### 4.4 Payoff Oracle

Encode discounted payoff in ancilla qubit amplitude:

$$\mathcal{A}|i\rangle|0\rangle = |i\rangle \left( \sqrt{1 - f(S_i)}|0\rangle + \sqrt{f(S_i)}|1\rangle \right)$$

where $f(S_i) = e^{-rT} \cdot \text{payoff}(S_i) / V_{\max}$ is the normalized payoff.

### 4.5 Error Bounds

For canonical QAE with $M$ Grover iterations:

$$|\hat{a} - a| \leq \frac{\pi}{M} + \frac{\pi^2}{M^2}$$

where $a = |\langle\psi|1\rangle|^2$ is the amplitude encoding the option price.

**Iterative QAE [6]** achieves similar scaling without quantum phase estimation:

$$\text{Error} = O\left(\frac{1}{M}\right), \quad \text{Queries} = O(M)$$

### 4.6 Resource Requirements

For practical option pricing with $\epsilon = 0.01\%$ relative error:

| Resource | Estimate |
|----------|----------|
| Price qubits | 10-12 (1024-4096 price bins) |
| Ancilla qubits | 1-2 (payoff encoding) |
| QAE qubits | 8-10 (precision) |
| Total qubits | ~20-25 |
| Circuit depth | $O(10^4 - 10^5)$ gates |

**Practical crossover**: Requires fault-tolerant quantum computers; current NISQ devices have insufficient depth.

---

## 5. Tensor Network Methods

### 5.1 Matrix Product States (MPS)

An MPS represents a function of $n$ variables as:

$$f(x_1, \ldots, x_n) = \sum_{\alpha_1, \ldots, \alpha_{n-1}} A_1^{x_1}[\alpha_1] A_2^{x_2}[\alpha_1, \alpha_2] \cdots A_n^{x_n}[\alpha_{n-1}]$$

where $A_i$ are tensors with bond dimension $D$ controlling expressivity.

**Complexity**: $O(nD^2d)$ vs exponential $O(d^n)$ for full tensor.

### 5.2 Application to Multi-Asset Options

For a basket option on $n$ assets:

$$V(S_1, \ldots, S_n, t) \approx \text{MPS}(S_1, \ldots, S_n, t)$$

The MPS efficiently captures correlations if they decay with "distance" (asset index).

### 5.3 Tree Tensor Networks (TTN)

TTN arranges tensors hierarchically:

```
        [Root]
       /      \
    [T1]      [T2]
    /  \      /  \
  [L1][L2]  [L3][L4]
```

Naturally suited for multi-scale problems and hierarchical asset structures.

### 5.4 Connection to Quantum Computing

Tensor networks arise naturally in quantum computation:
- Quantum states with limited entanglement are efficiently representable
- MPS correspond to 1D quantum states with area-law entanglement
- TTN generalize to tree-like entanglement structures

**Key insight**: Some quantum advantages may be achievable classically via tensor networks.

---

## References

[1] M. Raissi, P. Perdikaris, G.E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, 378:686-707, 2019. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

[2] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J.I. Latorre, "Data re-uploading for a universal quantum classifier," *Quantum*, 4:226, 2020. [arXiv:1907.02085](https://arxiv.org/abs/1907.02085)

[3] M. Schuld, R. Sweke, J.J. Meyer, "Effect of data encoding on the expressive power of variational quantum machine learning models," *Physical Review A*, 103:032430, 2021. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)

[4] J.R. McClean, S. Boixo, V.N. Smelyanskiy, R. Babbush, H. Neven, "Barren plateaus in quantum neural network training landscapes," *Nature Communications*, 9:4812, 2018. [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)

[5] G. Brassard, P. Høyer, M. Mosca, A. Tapp, "Quantum Amplitude Amplification and Estimation," *Contemporary Mathematics*, 305:53-74, 2002. [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)

[6] D. Grinko, J. Gacon, C. Zoufal, S. Woerner, "Iterative quantum amplitude estimation," *npj Quantum Information*, 7:52, 2021. [arXiv:1912.05559](https://arxiv.org/abs/1912.05559)

[7] N. Stamatopoulos et al., "Option Pricing using Quantum Computers," *Quantum*, 4:291, 2020. [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)

[8] M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, Springer, 2021. ISBN: 978-3-030-83097-7

[9] U. Schollwöck, "The density-matrix renormalization group in the age of matrix product states," *Annals of Physics*, 326:96-192, 2011. [arXiv:1008.3477](https://arxiv.org/abs/1008.3477)

[10] F. Black and M. Scholes, "The Pricing of Options and Corporate Liabilities," *Journal of Political Economy*, 81(3):637-654, 1973.
