# Mathematical Foundations

> **Executive Summary**: This document provides the theoretical underpinnings for using quantum-classical hybrid physics-informed neural networks (PINNs) to solve the Black-Scholes PDE. We derive the PDE from first principles, formalize the PINN loss function, explain variational quantum circuit expressivity, and outline quantum amplitude estimation for potential future speedups.

**Target Audience**: Quantitative researchers familiar with derivatives pricing and interested in ML/quantum approaches.

---

## Table of Contents

1. [Black-Scholes PDE](#1-black-scholes-pde)
2. [Physics-Informed Neural Networks](#2-physics-informed-neural-networks-pinns)
3. [Variational Quantum Circuits](#3-variational-quantum-circuits-for-function-approximation)
4. [Quantum Amplitude Estimation](#4-quantum-amplitude-estimation)
5. [Key References](#5-key-references)
6. [Multi-Asset Black-Scholes PDE](#6-multi-asset-black-scholes-pde)
7. [Inverse Problems and Volatility Calibration](#7-inverse-problems-and-volatility-calibration)
8. [Hybrid Quantum-Classical Architectures for High Dimensions](#8-hybrid-quantum-classical-architectures-for-high-dimensions)

---

## 1. Black-Scholes PDE

### 1.1 Derivation

The Black-Scholes partial differential equation describes the evolution of a derivative's price under the following assumptions:

1. The underlying asset follows geometric Brownian motion:
   $$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

2. Markets are frictionless (no transaction costs, continuous trading)
3. No arbitrage opportunities exist
4. Constant volatility $\sigma$ and risk-free rate $r$

Through delta hedging and Itô's lemma, we derive the **Black-Scholes PDE**:

$$\boxed{\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0}$$

where:
- $V(S, t)$ is the option value
- $S$ is the spot price of the underlying
- $t$ is time
- $\sigma$ is volatility
- $r$ is the risk-free interest rate

### 1.2 Boundary Conditions

For a **European call option** with strike $K$ and maturity $T$:

| Condition | Mathematical Form | Physical Meaning |
|-----------|-------------------|------------------|
| **Terminal** | $V(S, T) = \max(S - K, 0)$ | Payoff at expiry |
| **Lower boundary** | $V(0, t) = 0$ | Worthless if $S = 0$ |
| **Upper boundary** | $V(S, t) \to S - Ke^{-r(T-t)}$ as $S \to \infty$ | Deep ITM approaches intrinsic |

For a **European put option**:

| Condition | Mathematical Form |
|-----------|-------------------|
| **Terminal** | $V(S, T) = \max(K - S, 0)$ |
| **Lower boundary** | $V(0, t) = Ke^{-r(T-t)}$ |
| **Upper boundary** | $V(S, t) \to 0$ as $S \to \infty$ |

### 1.3 Analytical Solution

The closed-form Black-Scholes formula for a European call:

$$C(S, t) = S \cdot N(d_1) - Ke^{-r\tau} \cdot N(d_2)$$

where $\tau = T - t$ (time to maturity) and:

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}, \quad d_2 = d_1 - \sigma\sqrt{\tau}$$

$N(\cdot)$ is the standard normal CDF.

### 1.4 Greeks

The Greeks are sensitivities of option price to various parameters:

| Greek | Definition | Formula (Call) |
|-------|------------|----------------|
| **Delta** ($\Delta$) | $\frac{\partial V}{\partial S}$ | $N(d_1)$ |
| **Gamma** ($\Gamma$) | $\frac{\partial^2 V}{\partial S^2}$ | $\frac{N'(d_1)}{S\sigma\sqrt{\tau}}$ |
| **Theta** ($\Theta$) | $\frac{\partial V}{\partial t}$ | $-\frac{SN'(d_1)\sigma}{2\sqrt{\tau}} - rKe^{-r\tau}N(d_2)$ |
| **Vega** ($\nu$) | $\frac{\partial V}{\partial \sigma}$ | $S\sqrt{\tau}N'(d_1)$ |
| **Rho** ($\rho$) | $\frac{\partial V}{\partial r}$ | $K\tau e^{-r\tau}N(d_2)$ |

---

## 2. Physics-Informed Neural Networks (PINNs)

### 2.1 Core Idea

PINNs learn solutions to PDEs by incorporating the physical laws (PDEs) directly into the loss function, rather than learning from labeled data alone.

**Key insight:** The neural network $V_\theta(S, t)$ is trained such that it simultaneously:
1. Satisfies the PDE at interior points
2. Matches boundary conditions
3. Matches terminal/initial conditions

### 2.2 Loss Function Formulation

The total PINN loss is:

$$\boxed{\mathcal{L}_{\text{total}} = \lambda_{\text{PDE}}\mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}}\mathcal{L}_{\text{BC}} + \lambda_{\text{IC}}\mathcal{L}_{\text{IC}}}$$

#### PDE Residual Loss

Sample collocation points $\{(S_i, t_i)\}_{i=1}^{N_{\text{int}}}$ in the interior domain:

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| \mathcal{R}[V_\theta](S_i, t_i) \right|^2$$

where the **residual operator** $\mathcal{R}$ for Black-Scholes is:

$$\mathcal{R}[V] = \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV$$

The derivatives $\frac{\partial V}{\partial t}$, $\frac{\partial V}{\partial S}$, $\frac{\partial^2 V}{\partial S^2}$ are computed via **automatic differentiation**.

#### Boundary Condition Loss

Sample boundary points $\{(S_j^{\text{BC}}, t_j^{\text{BC}})\}$:

$$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{j=1}^{N_{\text{BC}}} \left| V_\theta(S_j^{\text{BC}}, t_j^{\text{BC}}) - V^{\text{BC}}(S_j^{\text{BC}}, t_j^{\text{BC}}) \right|^2$$

#### Terminal Condition Loss (Initial Condition in backward time)

Sample terminal points $\{S_k^{\text{IC}}\}$ at $t = T$:

$$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{k=1}^{N_{\text{IC}}} \left| V_\theta(S_k^{\text{IC}}, T) - \text{payoff}(S_k^{\text{IC}}) \right|^2$$

### 2.3 Adaptive Loss Weighting

Loss terms can have vastly different scales. Adaptive weighting (Wang et al., 2021) adjusts $\lambda$ based on gradient statistics:

$$\lambda_i^{(n+1)} = \alpha \lambda_i^{(n)} + (1-\alpha) \frac{\max_j |\nabla_\theta \mathcal{L}_j|}{|\nabla_\theta \mathcal{L}_i| + \epsilon}$$

This balances gradient magnitudes across loss terms, improving training stability.

### 2.4 Network Architecture

Typical architectures for PINNs:

```
Input (S, t) → Normalize → [MLP / ResNet] → Output V

MLP: Linear → Tanh → Linear → Tanh → ... → Linear
ResNet: Linear → [Tanh → Linear → Tanh → Linear + skip] × N → Linear
```

**Input normalization:** Scale $(S, t)$ to $[0, 1]$ for stable training:
$$\tilde{S} = S / S_{\max}, \quad \tilde{t} = t / T$$

---

## 3. Variational Quantum Circuits for Function Approximation

### 3.1 Quantum Computing Basics

A quantum state of $n$ qubits lives in a $2^n$-dimensional Hilbert space:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \sum_i |\alpha_i|^2 = 1$$

Quantum gates are unitary operations: $U^\dagger U = I$.

### 3.2 Variational Quantum Circuit (VQC)

A VQC is a parameterized quantum circuit $U(\theta)$ that maps classical inputs to quantum measurements:

$$f_\theta(x) = \langle 0 | U^\dagger(x, \theta) \, M \, U(x, \theta) | 0 \rangle$$

where:
- $U(x, \theta)$ combines **data encoding** and **trainable rotations**
- $M$ is a measurement observable (e.g., $\sigma_z$)
- Output is in $[-1, 1]$ for Pauli measurements

### 3.3 Circuit Architecture

**Hardware-Efficient Ansatz:**

```
Layer structure (repeated L times):
┌───────────────────────────────────────────────────┐
│  Data Encoding:  RY(x₀)|0⟩  RY(x₁)|1⟩  ...    │
├───────────────────────────────────────────────────┤
│  Rotations:      RX(θ₀)RY(θ₁)RZ(θ₂) on each   │
├───────────────────────────────────────────────────┤
│  Entanglement:   CNOT ring (0→1→2→...→n→0)    │
└───────────────────────────────────────────────────┘
```

**Data Re-uploading:**

Interleave data encoding with trainable layers for increased expressivity (Pérez-Salinas et al., 2020):

$$U(x, \theta) = \prod_{l=1}^{L} \left[ W_l(\theta_l) \cdot S(x) \right]$$

where $S(x)$ encodes data and $W_l(\theta_l)$ are trainable.

### 3.4 Expressivity

VQCs can approximate arbitrary functions under certain conditions:

**Universal approximation:** A VQC with sufficient depth and data re-uploading can approximate any continuous function (Schuld et al., 2021).

**Fourier perspective:** VQCs compute functions that are sums of Fourier components:

$$f_\theta(x) = \sum_\omega c_\omega(\theta) e^{i\omega x}$$

The frequencies $\omega$ are determined by the data encoding, amplitudes $c_\omega$ by trainable parameters.

### 3.5 Hybrid Architecture

For PINN, we use a hybrid quantum-classical architecture:

```
Classical Preprocessing → VQC → Classical Postprocessing

Input (S, t) → Normalize/Feature expand → [VQC] → Scale to V range → Output
           ↘ Classical layer (optional) ↗
```

**Quantum residual:** Use quantum as a correction to classical:

$$V(S, t) = V_{\text{classical}}(S, t) + \alpha \cdot V_{\text{quantum}}(S, t)$$

---

## 4. Quantum Amplitude Estimation

### 4.1 Monte Carlo Pricing

Classical Monte Carlo estimates option prices as:

$$\hat{V} = e^{-rT} \cdot \frac{1}{N} \sum_{i=1}^{N} \text{payoff}(S_T^{(i)})$$

where $S_T^{(i)}$ are simulated terminal prices under risk-neutral measure.

**Error scaling:** Standard error $\propto 1/\sqrt{N}$, requiring $N \propto 1/\epsilon^2$ samples for precision $\epsilon$.

### 4.2 Quantum Amplitude Estimation Algorithm

QAE achieves **quadratic speedup**: error $\propto 1/M$ with $M$ oracle queries.

**Algorithm outline:**

1. **State preparation:** Encode the price distribution as quantum amplitudes:
   $$|\psi\rangle = \sum_{i} \sqrt{p_i} |i\rangle$$
   where $p_i$ is the probability of price $S_i$.

2. **Payoff encoding:** Apply controlled rotation based on payoff:
   $$|i\rangle|0\rangle \to |i\rangle \left( \sqrt{1 - f_i}|0\rangle + \sqrt{f_i}|1\rangle \right)$$
   where $f_i = \text{payoff}(S_i) / \text{max\_payoff}$.

3. **Amplitude estimation:** The amplitude of $|1\rangle$ on the ancilla is:
   $$a = \sum_i p_i \cdot f_i = \mathbb{E}[\text{payoff}]$$

4. **Phase estimation:** Use Grover iterations and QPE to estimate $a$ with error $O(1/M)$.

### 4.3 Error Scaling Comparison

| Method | Error | Queries for $\epsilon$ precision |
|--------|-------|----------------------------------|
| Classical MC | $O(1/\sqrt{N})$ | $N = O(1/\epsilon^2)$ |
| Quantum AE | $O(1/M)$ | $M = O(1/\epsilon)$ |

**Quadratic speedup:** For precision $\epsilon = 0.01$:
- Classical: ~10,000 samples
- Quantum: ~100 queries

### 4.4 Iterative QAE

Standard QAE requires quantum phase estimation (many controlled operations). **Iterative QAE** (Grinko et al., 2019) avoids this:

1. Run Grover with $k$ iterations
2. Measure and record outcomes
3. Repeat with different $k$ values
4. Use maximum likelihood estimation to combine results

This is more suitable for near-term quantum hardware.

### 4.5 Resource Requirements

For practical quantum advantage:

| Resource | Estimate |
|----------|----------|
| Qubits (price discretization) | $n \approx 10-15$ |
| Qubits (precision) | $m \approx 10-20$ |
| Circuit depth | $O(2^m)$ Grover iterations |
| Error correction | Required for $m > 10$ |

**Crossover point:** Quantum advantage requires fault-tolerant hardware, estimated at $10^3$-$10^6$ physical qubits depending on error rates.

---

## 5. Key References

### PINNs

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
   - *Original PINN paper establishing the framework*

2. **Wang, S., Teng, Y., & Perdikaris, P.** (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055-A3081.
   - *Adaptive loss weighting and training stability*

### Quantum Machine Learning

3. **Schuld, M., & Petruccione, F.** (2021). *Machine Learning with Quantum Computers*. Springer.
   - *Comprehensive textbook on quantum ML*

4. **Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I.** (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
   - *Data re-uploading circuits for function approximation*

5. **Schuld, M., Sweke, R., & Meyer, J. K.** (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103(3), 032430.
   - *Fourier analysis of VQC expressivity*

### Quantum Finance

6. **Stamatopoulos, N., et al.** (2020). Option pricing using quantum computers. *Quantum*, 4, 291.
   - *QAE for option pricing with detailed resource estimates*

7. **Woerner, S., & Egger, D. J.** (2019). Quantum risk analysis. *npj Quantum Information*, 5(1), 15.
   - *Quantum amplitude estimation for risk measures*

8. **Grinko, D., Gacon, J., Zoufal, C., & Woerner, S.** (2021). Iterative quantum amplitude estimation. *npj Quantum Information*, 7(1), 52.
   - *Practical QAE without QPE*

### Tensor Networks

9. **Orús, R.** (2014). A practical introduction to tensor networks: Matrix product states and projected entangled pair states. *Annals of Physics*, 349, 117-158.
   - *Introduction to MPS and tensor network methods*

10. **Stoudenmire, E., & Schwab, D. J.** (2016). Supervised learning with tensor networks. *Advances in Neural Information Processing Systems*, 29.
    - *Tensor networks for machine learning*

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $V(S, t)$ | Option value at spot $S$ and time $t$ |
| $\sigma$ | Volatility |
| $r$ | Risk-free rate |
| $K$ | Strike price |
| $T$ | Maturity |
| $\tau = T - t$ | Time to maturity |
| $N(\cdot)$ | Standard normal CDF |
| $\mathcal{R}[\cdot]$ | PDE residual operator |
| $\lambda$ | Loss weighting coefficients |
| $\vert\psi\rangle$ | Quantum state |
| $U(\theta)$ | Parameterized unitary |
| $\sigma_x, \sigma_y, \sigma_z$ | Pauli matrices |

## Appendix B: Useful Identities

**Black-Scholes PDE from risk-neutral pricing:**

$$V(S, t) = e^{-r(T-t)} \mathbb{E}^{\mathbb{Q}}[\text{payoff}(S_T) | S_t = S]$$

Under $\mathbb{Q}$, the asset follows: $dS = rS\,dt + \sigma S\,dW^{\mathbb{Q}}$

**Feynman-Kac formula:** Solutions to the Black-Scholes PDE equal risk-neutral expectations.

**Quantum amplitude estimation error:**

$$|\hat{a} - a| \leq \frac{\pi}{M} + \frac{\pi^2}{M^2}$$

where $M$ is the number of Grover iterations.

---

## 6. Multi-Asset Black-Scholes PDE

### 6.1 Correlated Geometric Brownian Motion

For a basket of $n$ assets, each asset follows geometric Brownian motion with correlated Wiener processes:

$$dS_i = \mu_i S_i \, dt + \sigma_i S_i \, dW_i, \quad i = 1, \ldots, n$$

where the Wiener processes are correlated:

$$\mathbb{E}[dW_i \cdot dW_j] = \rho_{ij} \, dt$$

The correlation matrix $\boldsymbol{\rho} = (\rho_{ij})$ must be symmetric positive semi-definite. The covariance matrix is:

$$\Sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$$

### 6.2 The $n$-Dimensional Black-Scholes PDE

By applying Itô's lemma to a function $V(S_1, \ldots, S_n, t)$ and constructing a delta-hedged portfolio, we derive the **multi-asset Black-Scholes PDE**:

$$\boxed{\frac{\partial V}{\partial t} + \sum_{i=1}^{n} r S_i \frac{\partial V}{\partial S_i} + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \rho_{ij} \sigma_i \sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} - rV = 0}$$

This is an $(n+1)$-dimensional PDE in the variables $(S_1, \ldots, S_n, t)$.

**Key differences from 1D case:**
- Cross-derivative terms $\frac{\partial^2 V}{\partial S_i \partial S_j}$ for $i \neq j$
- Correlation structure affects the diffusion matrix
- Computational complexity grows exponentially with $n$

### 6.3 Basket Option Payoffs

A **basket option** is an option on a weighted average of underlying assets. Common types:

| Type | Terminal Condition $V(S_1, \ldots, S_n, T)$ |
|------|---------------------------------------------|
| **Equally-weighted call** | $\max\left(\frac{1}{n}\sum_{i=1}^n S_i - K, 0\right)$ |
| **Weighted basket call** | $\max\left(\sum_{i=1}^n w_i S_i - K, 0\right)$ where $\sum w_i = 1$ |
| **Best-of call (rainbow)** | $\max\left(\max_i S_i - K, 0\right)$ |
| **Worst-of call** | $\max\left(\min_i S_i - K, 0\right)$ |

### 6.4 Boundary Conditions

For multi-asset options, boundary conditions are specified when any asset price goes to zero or infinity:

| Boundary | Condition |
|----------|-----------|
| $S_i \to 0$ | Reduces to $(n-1)$-asset problem |
| $S_i \to \infty$ | $V \approx w_i S_i$ (deep ITM contribution) |
| $t = T$ | Payoff function (terminal condition) |

### 6.5 The Curse of Dimensionality

**Finite difference methods fail** for $n > 3$ assets due to exponential scaling:

For a grid with $N$ points per dimension:
$$\text{Grid points} = N^{n+1} = O(N^{n+1})$$

| Assets | Dimensions | Grid Points ($N=100$) | Memory (64-bit) |
|--------|------------|----------------------|-----------------|
| 1 | 2D | $10^4$ | 80 KB |
| 2 | 3D | $10^6$ | 8 MB |
| 3 | 4D | $10^8$ | 800 MB |
| 5 | 6D | $10^{12}$ | 8 TB |
| 10 | 11D | $10^{22}$ | Impossible |

**PINNs circumvent this** by sampling collocation points, not discretizing the entire domain.

### 6.6 Latin Hypercube Sampling

For high-dimensional problems, **Latin Hypercube Sampling (LHS)** provides better coverage than random sampling:

**Algorithm:**
1. Divide each dimension into $N$ equal intervals
2. Place exactly one sample in each interval per dimension
3. Randomly shuffle assignments to create sample matrix

**Advantages:**
- Guarantees uniform marginal distributions
- Avoids clustering in high dimensions
- Variance reduction: $\text{Var}_{\text{LHS}} \leq \text{Var}_{\text{MC}}$ for monotonic functions
- Scales to arbitrary dimensions

For a 5-asset basket with 15,000 collocation points:
- Random sampling: May leave regions unexplored
- LHS: Guaranteed coverage of the entire 6D hypercube

---

## 7. Inverse Problems and Volatility Calibration

### 7.1 Forward vs Inverse Problems

| Problem Type | Given | Find | Example |
|--------------|-------|------|---------|
| **Forward** | Model parameters $\theta$ | Prices $V$ | Price options given volatility |
| **Inverse** | Observed data $V^{\text{obs}}$ | Parameters $\theta$ | Calibrate volatility from prices |

Inverse problems are often **ill-posed** in the sense of Hadamard:
- Solution may not exist (inconsistent data)
- Solution may not be unique (underdetermined)
- Solution may be unstable (sensitive to noise)

**Regularization** makes inverse problems well-posed.

### 7.2 The Dupire Local Volatility Model

Bruno Dupire (1994) showed that European option prices uniquely determine a **local volatility function** $\sigma_{\text{loc}}(K, T)$ such that:

$$dS_t = (r - q) S_t \, dt + \sigma_{\text{loc}}(S_t, t) S_t \, dW_t$$

**The Dupire Equation:**

Given a continuum of European call prices $C(K, T)$ for all strikes $K$ and maturities $T$, the local volatility is:

$$\boxed{\sigma_{\text{loc}}^2(K, T) = \frac{2 \left( \frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC \right)}{K^2 \frac{\partial^2 C}{\partial K^2}}}$$

**Derivation insight:** Apply Itô's lemma to the call option in the forward measure, then differentiate w.r.t. strike.

### 7.3 Calibration as an Inverse Problem

**The calibration problem:** Given observed market prices $\{C^{\text{obs}}(K_i, T_j)\}$, find $\sigma(K, T)$.

**Challenges:**
1. Only discrete $(K, T)$ observations available
2. Market prices contain noise (bid-ask spread)
3. Dupire formula requires derivatives → amplifies noise
4. Surface must satisfy no-arbitrage constraints

### 7.4 PINN Approach to Calibration

PINNs solve the inverse problem by learning $\sigma(K, T)$ as a neural network:

**Architecture:**
$$\sigma_\theta(K, T) : \mathbb{R}^2 \to \mathbb{R}^+$$

**Loss function:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{arb}} \mathcal{L}_{\text{arb}}$$

where:

**Data fidelity:**
$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left| C_\theta(K_i, T_i) - C^{\text{obs}}(K_i, T_i) \right|^2$$

**Smoothness regularization:**
$$\mathcal{L}_{\text{smooth}} = \mathbb{E}\left[ \left| \frac{\partial \sigma}{\partial K} \right|^2 + \left| \frac{\partial \sigma}{\partial T} \right|^2 + \left| \frac{\partial^2 \sigma}{\partial K^2} \right|^2 \right]$$

**Arbitrage-free constraints:**
- Calendar arbitrage: $C$ must be increasing in $T$
- Butterfly arbitrage: $\frac{\partial^2 C}{\partial K^2} \geq 0$ (convexity in $K$)

$$\mathcal{L}_{\text{arb}} = \mathbb{E}\left[ \text{ReLU}\left( -\frac{\partial C}{\partial T} \right) + \text{ReLU}\left( -\frac{\partial^2 C}{\partial K^2} \right) \right]$$

### 7.5 Connection to Implied Volatility

**Implied volatility** $\sigma_{\text{IV}}(K, T)$ is the constant volatility that, when input to Black-Scholes, reproduces the market price:

$$C^{\text{BS}}(S_0, K, T, r, \sigma_{\text{IV}}) = C^{\text{market}}(K, T)$$

**Relationship to local volatility:**

For small maturities:
$$\sigma_{\text{loc}}(K, T) \approx \sigma_{\text{IV}}(K, T) + O(T)$$

For longer maturities, local vol is roughly a "harmonic average" of implied vol along paths.

### 7.6 Synthetic Volatility Surface Model

For testing calibration algorithms, we use a parametric SABR-like surface:

$$\sigma_{\text{IV}}(K, T) = \sigma_0 + \beta_1 \log(K/S_0) + \beta_2 \log^2(K/S_0) + \beta_3 T$$

where:
- $\sigma_0 \approx 0.20$ (ATM base volatility)
- $\beta_1 < 0$ (negative skew for equities)
- $\beta_2 > 0$ (smile curvature)
- $\beta_3 < 0$ (term structure: vol decreases with maturity)

This captures the essential features of equity volatility surfaces:
- ATM volatility level
- Downside skew (higher IV for lower strikes)
- Smile (higher IV for far OTM/ITM)
- Term structure (short-term vol typically higher)

---

## 8. Hybrid Quantum-Classical Architectures for High Dimensions

### 8.1 The Quantum Bottleneck

Current VQCs are limited by:
- **Qubit count:** NISQ devices have 50-100 noisy qubits
- **Gate fidelity:** Error rates of $10^{-3}$ to $10^{-2}$
- **Connectivity:** Limited qubit-qubit interactions

For high-dimensional problems (e.g., 6D basket option), directly encoding all inputs into a VQC is impractical.

### 8.2 Hybrid Architecture Strategies

**Strategy 1: Classical Compression**
$$\text{6D input} \xrightarrow{\text{Classical NN}} \text{2D features} \xrightarrow{\text{VQC}} \text{Output}$$

A classical encoder compresses high-dimensional input to a dimension suitable for the VQC.

**Strategy 2: Ensemble of VQCs**
$$\text{Output} = \sum_{(i,j)} w_{ij} \cdot \text{VQC}_{ij}(S_i, S_j)$$

Deploy multiple VQCs, each handling a pair of assets. For $n$ assets, use $\binom{n}{2}$ VQCs.

**Strategy 3: Quantum Residual Correction**
$$V(S, t) = V_{\text{classical}}(S, t) + \alpha \cdot V_{\text{quantum}}(S, t)$$

Use classical network as base predictor, with VQC providing a learned correction.

### 8.3 Expressivity Considerations

The quantum advantage in hybrid architectures may come from:
- **Inductive bias:** VQC function classes differ from classical NNs
- **Feature interactions:** Entanglement creates non-local correlations
- **Optimization landscape:** Different loss landscape geometry

Current evidence is mixed; rigorous quantum advantage for PINN-like tasks remains an open question.

---

## Appendix C: Extended References

### Multi-Asset Options

11. **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
    - *Standard reference for high-dimensional option pricing*

12. **Broadie, M., & Glasserman, P.** (1997). Pricing American-style securities using simulation. *Journal of Economic Dynamics and Control*, 21(8-9), 1323-1352.
    - *Least-squares Monte Carlo for American options*

### Local Volatility

13. **Dupire, B.** (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
    - *Original Dupire local volatility paper*

14. **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
    - *Comprehensive treatment of volatility modeling*

15. **Fengler, M. R.** (2009). Arbitrage-free smoothing of the implied volatility surface. *Quantitative Finance*, 9(4), 417-428.
    - *Regularization for arbitrage-free surfaces*

### Inverse Problems

16. **Tarantola, A.** (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.
    - *Mathematical foundations of inverse problems*

17. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks for inverse problems. *Computational Mechanics*, 64(5), 1095-1117.
    - *PINNs for parameter identification*

### Latin Hypercube Sampling

18. **McKay, M. D., Beckman, R. J., & Conover, W. J.** (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2), 239-245.
    - *Original LHS paper*

19. **Owen, A. B.** (1998). Latin supercube sampling for very high-dimensional simulations. *ACM Transactions on Modeling and Computer Simulation*, 8(1), 71-102.
    - *LHS for high-dimensional problems*
