"""
Merton Jump-Diffusion Model for Option Pricing.

Reference:
    Merton, R.C. (1976). "Option pricing when underlying stock returns are
    discontinuous." Journal of Financial Economics, 3(1-2), 125-144.

The model extends Geometric Brownian Motion with discontinuous jumps:
    dS/S = (r - λκ)dt + σdW + (J - 1)dN

Where:
    - W is a standard Brownian motion
    - N is a Poisson process with intensity λ (mean number of jumps per year)
    - J = exp(μ_J + σ_J * Z) is the jump multiplier (log-normal)
    - Z ~ N(0,1) is a standard normal random variable
    - κ = E[J - 1] = exp(μ_J + σ_J²/2) - 1 is the expected relative jump size

The corresponding PIDE (Partial Integro-Differential Equation):
    ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + (r - λκ)S(∂V/∂S) - rV
    + λ∫[V(SJ, t) - V(S, t)]g(J)dJ = 0

Where g(J) is the log-normal density of the jump multiplier.

The integral term captures the expected change in option value due to jumps,
making this significantly harder to solve than standard PDEs. PINN-based
approaches discretize the integral using quadrature methods.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class JumpDiffusionParams:
    """
    Parameters for the Merton Jump-Diffusion model.

    Attributes:
        r: Risk-free interest rate (annualized)
        sigma: Diffusion volatility (annualized)
        K: Strike price
        T: Time to maturity (in years)
        lambda_jump: Jump intensity (expected number of jumps per year)
        mu_jump: Mean of the log-jump size distribution
        sigma_jump: Standard deviation of the log-jump size distribution

    Note:
        The jump multiplier J is log-normal: log(J) ~ N(mu_jump, sigma_jump²)
        For downward jumps on average, use negative mu_jump.
        Typical values from Merton (1976): lambda_jump ≈ 1-3, sigma_jump ≈ 0.25-0.45
    """
    r: float
    sigma: float
    K: float
    T: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float


# Alias for backward compatibility
MertonParams = JumpDiffusionParams


def jump_kappa(params: JumpDiffusionParams) -> float:
    """
    Compute the expected relative jump size κ = E[J - 1].

    Under the log-normal assumption for J:
        κ = exp(μ_J + σ_J²/2) - 1

    This appears in the drift adjustment to ensure risk-neutral pricing.

    Args:
        params: Jump-diffusion model parameters

    Returns:
        Expected relative jump size κ

    Reference:
        Merton (1976), Equation (8)
    """
    return np.exp(params.mu_jump + 0.5 * params.sigma_jump**2) - 1


# Alias for backward compatibility
merton_kappa = jump_kappa


def jump_diffusion_pide_residual(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    S: torch.Tensor,
    t: torch.Tensor,
    params: JumpDiffusionParams,
    n_quad: int = 20,
) -> torch.Tensor:
    """
    Compute the PIDE residual for the Merton jump-diffusion model.

    The PIDE is:
        ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + (r - λκ)S(∂V/∂S) - rV
        + λ∫[V(SJ, t) - V(S, t)]g(J)dJ = 0

    The integral term is approximated using Gauss-Hermite quadrature,
    which is optimal for integrals of the form ∫f(x)exp(-x²)dx.
    The log-normal distribution is transformed to this form.

    Args:
        model: Neural network V(S, t) -> option value. Must be differentiable.
        S: Spot prices tensor, shape (batch_size,) or (batch_size, 1)
        t: Time tensor, shape (batch_size,) or (batch_size, 1)
        params: Jump-diffusion model parameters
        n_quad: Number of Gauss-Hermite quadrature points (default 20)

    Returns:
        PIDE residual tensor, same shape as input. Should be ≈0 for correct V.

    Note:
        Higher n_quad improves integral accuracy but increases computation.
        n_quad=20 typically provides good accuracy for reasonable sigma_jump.

    Reference:
        Merton (1976), Equation (17)
    """
    # Ensure gradients are tracked
    S = S.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    V = model(S, t)

    # Compute standard PDE terms via automatic differentiation
    dV_dt = torch.autograd.grad(
        V.sum(), t, create_graph=True, retain_graph=True
    )[0]
    dV_dS = torch.autograd.grad(
        V.sum(), S, create_graph=True, retain_graph=True
    )[0]
    d2V_dS2 = torch.autograd.grad(
        dV_dS.sum(), S, create_graph=True, retain_graph=True
    )[0]

    kappa = jump_kappa(params)

    # Standard Black-Scholes PDE terms with jump-adjusted drift
    pde_part = (
        dV_dt
        + 0.5 * params.sigma**2 * S**2 * d2V_dS2
        + (params.r - params.lambda_jump * kappa) * S * dV_dS
        - params.r * V
    )

    # Integral term: λ∫[V(SJ, t) - V(S, t)]g(J)dJ
    # Use Gauss-Hermite quadrature: ∫f(x)exp(-x²)dx ≈ Σ w_i f(x_i)
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    nodes = torch.tensor(nodes, dtype=S.dtype, device=S.device)
    weights = torch.tensor(weights, dtype=S.dtype, device=S.device)

    # Transform for log-normal: if log(J) = μ + σ*Z where Z~N(0,1)
    # Use substitution x = Z/√2, so Z = √2*x and exp(-x²) is the weight
    sqrt2 = np.sqrt(2.0)
    J = torch.exp(params.mu_jump + params.sigma_jump * sqrt2 * nodes)

    # Compute V(S*J, t) - V(S, t) for each quadrature point
    integral = torch.zeros_like(V)
    for i in range(n_quad):
        S_jumped = S * J[i]
        V_jumped = model(S_jumped, t)
        integral = integral + weights[i] * (V_jumped - V)

    # Normalize by √π (Gauss-Hermite normalization factor)
    integral = integral / np.sqrt(np.pi)

    # Full PIDE residual
    residual = pde_part + params.lambda_jump * integral

    return residual


# Alias for backward compatibility
merton_pide_residual = jump_diffusion_pide_residual


def merton_analytical_call(
    S: torch.Tensor,
    t: torch.Tensor,
    params: JumpDiffusionParams,
    n_terms: int = 50,
) -> torch.Tensor:
    """
    Merton's analytical formula for European call options under jump-diffusion.

    The solution is an infinite series of Black-Scholes prices weighted by
    Poisson probabilities:

        V = Σ_{n=0}^{∞} P(N=n) * BS(S, K, r_n, σ_n, τ)

    Where:
        - P(N=n) is the Poisson probability of n jumps in time τ
        - r_n, σ_n are adjusted parameters accounting for n jumps
        - BS is the standard Black-Scholes formula

    The adjusted parameters are:
        λ' = λ(1 + κ)  (risk-neutral jump intensity)
        σ_n² = σ² + n*σ_J²/τ  (variance including n jump contributions)
        r_n = r - λκ + n*log(1+κ)/τ  (adjusted risk-free rate)

    Args:
        S: Spot price tensor
        t: Current time tensor
        params: Jump-diffusion model parameters
        n_terms: Number of terms in the series (default 50)

    Returns:
        European call option price tensor

    Note:
        The series converges rapidly for typical parameters. n_terms=50
        is usually sufficient for high accuracy.

    Reference:
        Merton (1976), Equation (17) and surrounding derivation
    """
    from scipy.stats import norm, poisson

    S_np = S.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    tau = params.T - t_np  # Time to maturity

    # Handle edge case of zero time to maturity
    tau = np.maximum(tau, 1e-10)

    kappa = jump_kappa(params)

    # Risk-neutral jump intensity (under equivalent martingale measure)
    lambda_prime = params.lambda_jump * (1 + kappa)

    price = np.zeros_like(S_np)

    for n in range(n_terms):
        # Adjusted variance for n jumps
        # Total variance = diffusion variance + jump variance contribution
        sigma_n_sq = params.sigma**2 + n * params.sigma_jump**2 / tau
        sigma_n = np.sqrt(sigma_n_sq)

        # Adjusted risk-free rate for n jumps
        # Accounts for drift adjustment and expected value of n jumps
        r_n = params.r - params.lambda_jump * kappa + n * np.log(1 + kappa) / tau

        # Standard Black-Scholes formula with adjusted parameters
        d1 = (np.log(S_np / params.K) + (r_n + 0.5 * sigma_n_sq) * tau) / (
            sigma_n * np.sqrt(tau)
        )
        d2 = d1 - sigma_n * np.sqrt(tau)

        bs_call = (
            S_np * norm.cdf(d1) 
            - params.K * np.exp(-r_n * tau) * norm.cdf(d2)
        )

        # Poisson probability weight for n jumps in time τ
        poisson_weight = poisson.pmf(n, lambda_prime * tau)

        price = price + poisson_weight * bs_call

    return torch.tensor(price, dtype=S.dtype, device=S.device)


# Alias for backward compatibility
merton_analytical_approx = merton_analytical_call


def merton_analytical_put(
    S: torch.Tensor,
    t: torch.Tensor,
    params: JumpDiffusionParams,
    n_terms: int = 50,
) -> torch.Tensor:
    """
    Merton's analytical formula for European put options under jump-diffusion.

    Uses put-call parity: P = C - S + K*exp(-r*τ)

    Args:
        S: Spot price tensor
        t: Current time tensor
        params: Jump-diffusion model parameters
        n_terms: Number of terms in the series (default 50)

    Returns:
        European put option price tensor

    Reference:
        Merton (1976), with put-call parity
    """
    call_price = merton_analytical_call(S, t, params, n_terms)
    tau = params.T - t
    put_price = call_price - S + params.K * torch.exp(-params.r * tau)
    return put_price


def jump_diffusion_terminal_condition(
    S: torch.Tensor,
    params: JumpDiffusionParams,
    option_type: str = "call",
) -> torch.Tensor:
    """
    Terminal payoff condition for European options under jump-diffusion.

    At maturity T, the option value equals the payoff:
        Call: V(S, T) = max(S - K, 0)
        Put:  V(S, T) = max(K - S, 0)

    Note that the terminal condition is identical to Black-Scholes;
    the jumps affect the dynamics but not the final payoff.

    Args:
        S: Spot price tensor at maturity
        params: Jump-diffusion model parameters (uses K for strike)
        option_type: "call" or "put"

    Returns:
        Terminal payoff tensor

    Reference:
        Merton (1976), Equation (11)
    """
    if option_type == "call":
        return torch.relu(S - params.K)
    elif option_type == "put":
        return torch.relu(params.K - S)
    else:
        raise ValueError(f"Unknown option_type: {option_type}. Use 'call' or 'put'.")


def jump_diffusion_boundary_conditions(
    S_max: float,
    t: torch.Tensor,
    params: JumpDiffusionParams,
    option_type: str = "call",
) -> dict[str, torch.Tensor]:
    """
    Boundary conditions for jump-diffusion option pricing.

    For European call options:
        - V(0, t) = 0 (worthless if stock is worthless)
        - V(S, t) → S - K*exp(-r*τ) as S → ∞ (deep ITM approximation)

    For European put options:
        - V(0, t) = K*exp(-r*τ) (guaranteed payoff if stock is worthless)
        - V(S, t) → 0 as S → ∞ (worthless if deeply OTM)

    Args:
        S_max: Upper boundary for spot price
        t: Time tensor
        params: Jump-diffusion model parameters
        option_type: "call" or "put"

    Returns:
        Dictionary with boundary values:
            - "S_zero": Value at S=0
            - "S_max": Value at S=S_max
    """
    tau = params.T - t

    if option_type == "call":
        return {
            "S_zero": torch.zeros_like(t),
            "S_max": S_max - params.K * torch.exp(-params.r * tau),
        }
    elif option_type == "put":
        return {
            "S_zero": params.K * torch.exp(-params.r * tau),
            "S_max": torch.zeros_like(t),
        }
    else:
        raise ValueError(f"Unknown option_type: {option_type}. Use 'call' or 'put'.")


def simulate_merton_paths(
    S0: float,
    params: JumpDiffusionParams,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate sample paths under the Merton jump-diffusion model.

    Useful for Monte Carlo pricing and model validation.

    The discrete approximation is:
        S_{t+dt} = S_t * exp((r - λκ - σ²/2)dt + σ√dt*Z + Σ_{j=1}^{N_t} Y_j)

    Where:
        - Z ~ N(0,1) is the diffusion component
        - N_t ~ Poisson(λ*dt) is the number of jumps in [t, t+dt]
        - Y_j ~ N(μ_J, σ_J²) are the log-jump sizes

    Args:
        S0: Initial spot price
        params: Jump-diffusion model parameters
        n_paths: Number of simulation paths
        n_steps: Number of time steps
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_paths, n_steps + 1) containing simulated paths

    Reference:
        Glasserman (2003), "Monte Carlo Methods in Financial Engineering", Ch. 3
    """
    if seed is not None:
        np.random.seed(seed)

    dt = params.T / n_steps
    kappa = jump_kappa(params)

    # Initialize paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    # Drift adjustment for log-price process
    drift = (params.r - params.lambda_jump * kappa - 0.5 * params.sigma**2) * dt

    for i in range(n_steps):
        # Diffusion component
        Z = np.random.standard_normal(n_paths)
        diffusion = params.sigma * np.sqrt(dt) * Z

        # Jump component
        # Number of jumps in this time step (Poisson)
        n_jumps = np.random.poisson(params.lambda_jump * dt, n_paths)

        # Total jump size (sum of log-normal jumps)
        jump_component = np.zeros(n_paths)
        for j in range(n_paths):
            if n_jumps[j] > 0:
                log_jumps = np.random.normal(
                    params.mu_jump, params.sigma_jump, n_jumps[j]
                )
                jump_component[j] = np.sum(log_jumps)

        # Update log-price and convert back
        log_return = drift + diffusion + jump_component
        paths[:, i + 1] = paths[:, i] * np.exp(log_return)

    return paths
