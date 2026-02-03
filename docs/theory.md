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
9. [Merton Jump-Diffusion Model](#9-merton-jump-diffusion-model)
10. [Heston Stochastic Volatility Model](#10-heston-stochastic-volatility-model)
11. [American Options and Free Boundary Problems](#11-american-options-and-free-boundary-problems)

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

## 9. Merton Jump-Diffusion Model

### 9.1 Asset Dynamics with Jumps

The classical Black-Scholes model assumes continuous price paths, but empirical evidence shows that asset prices exhibit **discontinuous jumps** (e.g., earnings announcements, market crashes). Merton (1976) extended the geometric Brownian motion to include Poisson-driven jumps:

$$\boxed{\frac{dS}{S} = (r - \lambda\kappa) \, dt + \sigma \, dW + (J-1) \, dN}$$

where:
- $W$ is a standard Brownian motion (continuous component)
- $N$ is a Poisson process with intensity $\lambda$ (jump arrivals per unit time)
- $J$ is the random jump multiplier: $S \to SJ$ upon a jump
- $\sigma$ is the diffusion volatility (between jumps)

### 9.2 Jump Size Distribution

The jump multiplier $J$ follows a **lognormal distribution**:

$$\log(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$$

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| $\lambda$ | Jump intensity (jumps/year) | 0.5 - 2.0 |
| $\mu_J$ | Mean of $\log(J)$ | $-0.10$ (negative for crashes) |
| $\sigma_J$ | Std of $\log(J)$ | 0.10 - 0.25 |

The **expected relative jump size** is:

$$\boxed{\kappa = \mathbb{E}[J-1] = e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1}$$

For typical parameters with $\mu_J < 0$, we have $\kappa < 0$, reflecting the asymmetric nature of market jumps (crashes more common than spikes).

### 9.3 The Pricing PIDE

Option prices under the Merton model satisfy a **partial integro-differential equation (PIDE)**:

$$\boxed{\frac{\partial V}{\partial t} + (r - \lambda\kappa) S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV + \lambda \int_0^\infty \left[ V(SJ, t) - V(S, t) \right] g(J) \, dJ = 0}$$

where $g(J)$ is the lognormal density of the jump size:

$$g(J) = \frac{1}{J\sigma_J\sqrt{2\pi}} \exp\left( -\frac{(\log J - \mu_J)^2}{2\sigma_J^2} \right)$$

**Key differences from Black-Scholes:**
- The drift is adjusted: $r \to r - \lambda\kappa$ to maintain risk-neutrality
- An **integral term** accounts for the expected change in option value due to jumps
- The equation is non-local: the value at $S$ depends on values at all $SJ$

### 9.4 Numerical Approximation of the Integral Term

The integral term cannot be evaluated analytically for general $V$. For PINNs, we use **Gauss-Hermite quadrature**:

$$\int_{-\infty}^{\infty} f(x) e^{-x^2} \, dx \approx \sum_{i=1}^{n} w_i f(x_i)$$

Since $\log(J) \sim \mathcal{N}(\mu_J, \sigma_J^2)$, we substitute $z = (\log J - \mu_J)/(\sigma_J\sqrt{2})$:

$$\int_0^\infty V(SJ) g(J) \, dJ = \frac{1}{\sqrt{\pi}} \sum_{i=1}^{n} w_i \, V\left(S \cdot e^{\mu_J + \sigma_J \sqrt{2} x_i}\right)$$

where $(x_i, w_i)$ are the Gauss-Hermite nodes and weights. Typical choices use $n = 15$-$30$ quadrature points.

### 9.5 Semi-Analytical Pricing Formula

Merton derived a remarkable **series solution** for European options. The call price is a weighted sum of Black-Scholes prices:

$$\boxed{C = \sum_{n=0}^{\infty} \frac{e^{-\lambda' \tau} (\lambda' \tau)^n}{n!} \cdot C^{BS}(S, K, \tau, r_n, \sigma_n)}$$

where $\tau = T - t$ and the adjusted parameters for $n$ jumps are:

| Quantity | Formula |
|----------|---------|
| $\lambda'$ | $\lambda(1 + \kappa)$ |
| $\sigma_n^2$ | $\sigma^2 + n\sigma_J^2 / \tau$ |
| $r_n$ | $r - \lambda\kappa + n\log(1+\kappa)/\tau$ |

**Interpretation:** The price is a Poisson-weighted average over scenarios with 0, 1, 2, ... jumps occurring before expiry. In the $n$-jump scenario:
- Volatility increases by $\sqrt{n}\sigma_J/\sqrt{\tau}$ (jumps add variance)
- The effective interest rate adjusts for the expected drift from $n$ jumps

The series converges rapidly; typically 30-50 terms suffice for machine precision.

### 9.6 PINN Loss Function for Merton

For the PINN approach, the loss includes:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{PIDE}} \mathcal{L}_{\text{PIDE}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}}$$

where the PIDE residual loss is:

$$\mathcal{L}_{\text{PIDE}} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial V}{\partial t} + (r-\lambda\kappa)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV + \lambda \cdot I[V] \right|^2$$

and $I[V]$ is the numerically approximated integral term.

*See `src/pde/merton.py` for implementation details.*

---

## 10. Heston Stochastic Volatility Model

### 10.1 The Stochastic Volatility Framework

The Black-Scholes model assumes constant volatility, contradicting the empirically observed **volatility smile/skew**. Heston (1993) introduced a model where volatility itself is a stochastic process:

$$\boxed{\begin{aligned}
dS &= rS \, dt + \sqrt{v} \, S \, dW_S \\
dv &= \kappa(\theta - v) \, dt + \xi \sqrt{v} \, dW_v
\end{aligned}}$$

with correlated Brownian motions:

$$dW_S \cdot dW_v = \rho \, dt$$

### 10.2 Model Parameters

| Parameter | Symbol | Meaning | Typical Range |
|-----------|--------|---------|---------------|
| Mean reversion speed | $\kappa$ | Rate at which $v$ returns to $\theta$ | 1.0 - 5.0 |
| Long-run variance | $\theta$ | Equilibrium level of variance | 0.02 - 0.10 |
| Volatility of volatility | $\xi$ | How much variance fluctuates | 0.2 - 0.8 |
| Correlation | $\rho$ | Asset-variance correlation | $-0.9$ to $-0.3$ (equity) |
| Initial variance | $v_0$ | Starting variance level | $\approx \theta$ |

**The Feller condition** ensures the variance process remains strictly positive:

$$\boxed{2\kappa\theta > \xi^2}$$

When violated, the variance can hit zero, requiring boundary treatment. For typical equity parameters, the Feller condition is often marginally satisfied or violated.

### 10.3 The 3D Pricing PDE

Option prices $V(S, v, t)$ satisfy a **three-dimensional PDE** (two spatial, one temporal):

$$\boxed{\frac{\partial V}{\partial t} + \frac{1}{2}vS^2 \frac{\partial^2 V}{\partial S^2} + \rho\xi vS \frac{\partial^2 V}{\partial S \partial v} + \frac{1}{2}\xi^2 v \frac{\partial^2 V}{\partial v^2} + rS\frac{\partial V}{\partial S} + \kappa(\theta-v)\frac{\partial V}{\partial v} - rV = 0}$$

**Key features:**
- **Cross-derivative term** $\frac{\partial^2 V}{\partial S \partial v}$: couples the asset and variance dimensions
- **Degenerate at** $v = 0$: all diffusion terms vanish when variance hits zero
- **Mean reversion drift** $\kappa(\theta - v)$: pulls variance toward $\theta$

### 10.4 Boundary Conditions

| Boundary | Condition |
|----------|-----------|
| $S = 0$ | $V(0, v, t) = 0$ (call), $V(0, v, t) = Ke^{-r(T-t)}$ (put) |
| $S \to \infty$ | $V \to S - Ke^{-r(T-t)}$ (call), $V \to 0$ (put) |
| $v = 0$ | PDE degenerates; use limiting behavior or absorbing condition |
| $v \to \infty$ | $V \to S$ (extreme volatility: option has full upside) |
| $t = T$ | Payoff: $\max(S - K, 0)$ or $\max(K - S, 0)$ |

### 10.5 Characteristic Function and Semi-Analytical Pricing

Heston derived a **semi-analytical solution** using Fourier methods. The characteristic function of $\log(S_T/S_0)$ under the risk-neutral measure is:

$$\phi(u; \tau) = \exp\left( C(u; \tau) + D(u; \tau) v_0 + iu \log(S_0) \right)$$

where $C$ and $D$ satisfy Riccati ODEs with closed-form solutions:

$$\begin{aligned}
d &= \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)} \\
g &= \frac{\kappa - \rho\xi iu - d}{\kappa - \rho\xi iu + d} \\
C(u; \tau) &= r \cdot iu \cdot \tau + \frac{\kappa\theta}{\xi^2}\left[ (\kappa - \rho\xi iu - d)\tau - 2\log\left(\frac{1 - ge^{-d\tau}}{1-g}\right) \right] \\
D(u; \tau) &= \frac{\kappa - \rho\xi iu - d}{\xi^2} \cdot \frac{1 - e^{-d\tau}}{1 - ge^{-d\tau}}
\end{aligned}$$

The call price is then obtained via **Fourier inversion**:

$$C = S_0 P_1 - Ke^{-r\tau} P_2$$

where $P_1$ and $P_2$ are computed by numerical integration of the characteristic function.

### 10.6 PINN Architecture for Heston

The PINN takes **3D input** $(S, v, t)$ and outputs the option value $V$:

```
(S, v, t) → Normalize → MLP → V

Normalization:
  S̃ = S / S_max
  ṽ = v / v_max  
  t̃ = t / T
```

**Loss function:**

$$\mathcal{L} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}}$$

**Sampling considerations:**
- Variance $v$ should be sampled from a distribution concentrated around $\theta$, avoiding $v \approx 0$ where the PDE degenerates
- The correlation $\rho < 0$ creates skew: more training points needed for low $S$, high $v$ regions

*See `src/pde/heston.py` for implementation details.*

---

## 11. American Options and Free Boundary Problems

### 11.1 The Early Exercise Feature

Unlike European options (exercisable only at maturity), **American options** can be exercised at any time $t \leq T$. This flexibility has significant pricing implications:

- American call on non-dividend stock: equals European call (never optimal to exercise early)
- American put: strictly more valuable than European put
- The **early exercise premium** is: $V_{\text{American}} - V_{\text{European}} > 0$

### 11.2 The Obstacle Problem Formulation

American option pricing leads to a **free boundary problem** or **obstacle problem**. The option value must satisfy three conditions simultaneously:

$$\boxed{\begin{aligned}
&\text{(i)} \quad V(S, t) \geq \Phi(S) \quad &\text{(no-arbitrage: worth at least intrinsic)} \\
&\text{(ii)} \quad \mathcal{L}V \leq 0 \quad &\text{(PDE inequality)} \\
&\text{(iii)} \quad (V - \Phi) \cdot \mathcal{L}V = 0 \quad &\text{(complementarity)}
\end{aligned}}$$

where:
- $\Phi(S)$ is the payoff function: $\max(K - S, 0)$ for put, $\max(S - K, 0)$ for call
- $\mathcal{L}$ is the Black-Scholes differential operator:

$$\mathcal{L}V = \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV$$

### 11.3 Interpretation of the Conditions

| Region | Condition | $V$ vs $\Phi$ | $\mathcal{L}V$ | Optimal Action |
|--------|-----------|---------------|----------------|----------------|
| **Continuation** | $V > \Phi$ | Strictly greater | $= 0$ | Hold the option |
| **Exercise** | $V = \Phi$ | Equal | $< 0$ | Exercise immediately |

The **complementarity condition** (iii) ensures exactly one of two cases holds:
- If $V > \Phi$: we must have $\mathcal{L}V = 0$ (standard Black-Scholes)
- If $\mathcal{L}V < 0$: we must have $V = \Phi$ (exercise is optimal)

### 11.4 The Early Exercise Boundary

The domain is partitioned by the **early exercise boundary** $S^*(t)$:

**For American put:**
- **Exercise region:** $S < S^*(t)$ — exercise immediately for payoff $K - S$
- **Continuation region:** $S > S^*(t)$ — hold the option
- $S^*(T) = K$ (at maturity, exercise when ITM)
- $S^*(t) < K$ for $t < T$ (exercise threshold decreases as expiry approaches)

**For American call (with dividends):**
- **Exercise region:** $S > S^*(t)$
- **Continuation region:** $S < S^*(t)$

The boundary $S^*(t)$ is **not known a priori** and must be determined as part of the solution — hence "free boundary problem."

### 11.5 Penalty Method for PINNs

Direct enforcement of the complementarity conditions is challenging for neural networks. The **penalty method** provides a smooth approximation:

**Obstacle penalty:**

$$\mathcal{L}_{\text{obstacle}} = \frac{1}{N}\sum_{i=1}^{N} \left[ \text{ReLU}(\Phi(S_i) - V(S_i, t_i)) \right]^2$$

This penalizes violations of $V \geq \Phi$.

**PDE penalty (continuation region):**

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_{i=1}^{N} \left[ \text{ReLU}(\mathcal{L}V) \right]^2$$

This penalizes $\mathcal{L}V > 0$ (only enforce $\mathcal{L}V \leq 0$).

**Complementarity penalty:**

$$\mathcal{L}_{\text{comp}} = \frac{1}{N}\sum_{i=1}^{N} \left[ (V - \Phi) \cdot \text{ReLU}(\mathcal{L}V) \right]$$

This encourages $(V - \Phi) \cdot \mathcal{L}V = 0$.

### 11.6 Network Architecture with Built-in Constraint

A more robust approach **builds the obstacle constraint into the architecture**:

$$V(S, t) = \Phi(S) + \text{softplus}(\text{NN}(S, t))$$

This guarantees $V \geq \Phi$ by construction:
- The network outputs a "time value" which is always non-negative
- Total value = intrinsic value + time value
- At expiry, time value → 0, so $V(S, T) = \Phi(S)$ automatically

### 11.7 American Put Premium

The early exercise premium for American puts can be substantial:

| $S/K$ | European Put | American Put | Premium |
|-------|--------------|--------------|---------|
| 0.8 | 17.5 | 20.0 | 14% |
| 1.0 | 5.6 | 6.1 | 9% |
| 1.2 | 1.4 | 1.5 | 7% |

*Example: $K = 100$, $\sigma = 0.20$, $r = 0.05$, $T = 1$*

The premium is larger for:
- Deep in-the-money puts (high intrinsic value)
- Longer maturities (more opportunities to exercise)
- Higher interest rates (greater benefit from receiving $K$ early)

### 11.8 Numerical Validation

American option PINNs should be validated against:

1. **Binomial tree** (Cox-Ross-Rubinstein): $O(N^2)$ complexity, converges as $N \to \infty$
2. **Finite difference with PSOR**: Projected Successive Over-Relaxation for the obstacle problem
3. **Least-squares Monte Carlo** (Longstaff-Schwartz): Regression-based early exercise approximation

*See `src/pde/american.py` for implementation details.*

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
