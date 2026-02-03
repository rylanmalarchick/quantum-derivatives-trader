"""
Finite Difference pricing methods.

Numerical PDE solvers as baseline for PINN comparison.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Optional


@dataclass
class FDGrid:
    """Finite difference grid specification."""
    S_min: float = 0.0
    S_max: float = 200.0
    n_S: int = 100      # Spatial grid points
    n_t: int = 1000     # Time grid points


class FiniteDifferencePricer:
    """
    Finite difference solver for Black-Scholes PDE.

    Implements explicit, implicit, and Crank-Nicolson schemes.
    """

    def __init__(
        self,
        r: float,
        sigma: float,
        grid: Optional[FDGrid] = None,
    ):
        self.r = r
        self.sigma = sigma
        self.grid = grid or FDGrid()

    def _setup_grid(self, K: float, T: float) -> tuple:
        """Set up spatial and time grids."""
        S = np.linspace(self.grid.S_min, self.grid.S_max, self.grid.n_S)
        t = np.linspace(0, T, self.grid.n_t)

        dS = S[1] - S[0]
        dt = t[1] - t[0]

        return S, t, dS, dt

    def price_explicit(
        self,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Explicit finite difference scheme.

        Simple but requires dt < dS²/(σ²S_max²) for stability.

        Returns:
            (S_grid, V_at_t0) - grid and option values at t=0
        """
        S, t, dS, dt = self._setup_grid(K, T)
        n_S = len(S)
        n_t = len(t)

        # Check stability
        stability = dt * self.sigma**2 * self.grid.S_max**2 / dS**2
        if stability > 0.5:
            print(f"Warning: stability ratio {stability:.2f} > 0.5, may be unstable")

        # Initialize with terminal condition
        if option_type == "call":
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)

        # Time stepping (backward from T to 0)
        for n in range(n_t - 1, 0, -1):
            V_new = V.copy()

            for i in range(1, n_S - 1):
                # Coefficients
                a = 0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
                b = 1 - dt * (self.sigma**2 * i**2 + self.r)
                c = 0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

                V_new[i] = a * V[i - 1] + b * V[i] + c * V[i + 1]

            # Boundary conditions
            if option_type == "call":
                V_new[0] = 0
                V_new[-1] = S[-1] - K * np.exp(-self.r * t[n - 1])
            else:
                V_new[0] = K * np.exp(-self.r * t[n - 1])
                V_new[-1] = 0

            V = V_new

        return S, V

    def price_implicit(
        self,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Implicit finite difference scheme.

        Unconditionally stable but requires solving linear system.

        Returns:
            (S_grid, V_at_t0)
        """
        S, t, dS, dt = self._setup_grid(K, T)
        n_S = len(S)
        n_t = len(t)

        # Initialize with terminal condition
        if option_type == "call":
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)

        # Build tridiagonal matrix
        i = np.arange(1, n_S - 1)

        # Coefficients for implicit scheme
        a = -0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
        b = 1 + dt * (self.sigma**2 * i**2 + self.r)
        c = -0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

        # Create sparse tridiagonal matrix
        diagonals = [a[1:], b, c[:-1]]
        A = sparse.diags(diagonals, [-1, 0, 1], format='csr')

        # Time stepping
        for n in range(n_t - 1, 0, -1):
            # Right-hand side
            rhs = V[1:-1].copy()

            # Boundary conditions
            if option_type == "call":
                V[0] = 0
                V[-1] = S[-1] - K * np.exp(-self.r * t[n - 1])
                rhs[0] -= a[0] * V[0]
                rhs[-1] -= c[-1] * V[-1]
            else:
                V[0] = K * np.exp(-self.r * t[n - 1])
                V[-1] = 0
                rhs[0] -= a[0] * V[0]
                rhs[-1] -= c[-1] * V[-1]

            # Solve system
            V[1:-1] = spsolve(A, rhs)

        return S, V

    def price_crank_nicolson(
        self,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crank-Nicolson scheme (average of explicit and implicit).

        Second-order accurate in both time and space.

        Returns:
            (S_grid, V_at_t0)
        """
        S, t, dS, dt = self._setup_grid(K, T)
        n_S = len(S)
        n_t = len(t)

        # Initialize
        if option_type == "call":
            V = np.maximum(S - K, 0)
        else:
            V = np.maximum(K - S, 0)

        i = np.arange(1, n_S - 1)

        # Coefficients (halved for CN averaging)
        alpha = 0.25 * dt * (self.sigma**2 * i**2 - self.r * i)
        beta = -0.5 * dt * (self.sigma**2 * i**2 + self.r)
        gamma = 0.25 * dt * (self.sigma**2 * i**2 + self.r * i)

        # LHS matrix: (I - 0.5*A)
        M1 = sparse.diags(
            [-alpha[1:], 1 - beta, -gamma[:-1]],
            [-1, 0, 1],
            format='csr'
        )

        # RHS matrix: (I + 0.5*A)
        M2 = sparse.diags(
            [alpha[1:], 1 + beta, gamma[:-1]],
            [-1, 0, 1],
            format='csr'
        )

        # Time stepping
        for n in range(n_t - 1, 0, -1):
            # RHS = M2 @ V[1:-1]
            rhs = M2 @ V[1:-1]

            # Boundary conditions
            if option_type == "call":
                bc_left = 0
                bc_right = S[-1] - K * np.exp(-self.r * t[n - 1])
            else:
                bc_left = K * np.exp(-self.r * t[n - 1])
                bc_right = 0

            rhs[0] += alpha[0] * (V[0] + bc_left)
            rhs[-1] += gamma[-1] * (V[-1] + bc_right)

            V[0] = bc_left
            V[-1] = bc_right

            # Solve
            V[1:-1] = spsolve(M1, rhs)

        return S, V

    def compute_greeks(
        self,
        K: float,
        T: float,
        option_type: str = "call",
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute option value and Greeks via finite differences.

        Returns:
            Dictionary with V, delta, gamma, theta for the grid
        """
        S, V = self.price_crank_nicolson(K, T, option_type)
        dS = S[1] - S[0]

        # Delta: ∂V/∂S (central difference)
        delta = np.zeros_like(V)
        delta[1:-1] = (V[2:] - V[:-2]) / (2 * dS)
        delta[0] = (V[1] - V[0]) / dS
        delta[-1] = (V[-1] - V[-2]) / dS

        # Gamma: ∂²V/∂S²
        gamma = np.zeros_like(V)
        gamma[1:-1] = (V[2:] - 2 * V[1:-1] + V[:-2]) / dS**2

        # Theta: need to price at T - dt
        dt_theta = T / 100
        _, V_later = self.price_crank_nicolson(K, T - dt_theta, option_type)
        theta = (V_later - V) / dt_theta

        return {
            "S": S,
            "V": V,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
        }


class AmericanFDPricer(FiniteDifferencePricer):
    """
    Finite difference for American options.

    Uses projected SOR to handle early exercise constraint.
    """

    def price_american(
        self,
        K: float,
        T: float,
        option_type: str = "put",
        omega: float = 1.2,  # SOR relaxation parameter
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Price American option using implicit FD with PSOR.

        Args:
            K: Strike
            T: Time to maturity
            option_type: "call" or "put"
            omega: SOR relaxation parameter
            tol: Convergence tolerance

        Returns:
            (S_grid, V_at_t0)
        """
        S, t, dS, dt = self._setup_grid(K, T)
        n_S = len(S)
        n_t = len(t)

        # Intrinsic value (exercise payoff)
        if option_type == "put":
            intrinsic = np.maximum(K - S, 0)
        else:
            intrinsic = np.maximum(S - K, 0)

        V = intrinsic.copy()

        i = np.arange(1, n_S - 1)
        a = -0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
        b = 1 + dt * (self.sigma**2 * i**2 + self.r)
        c = -0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

        # Time stepping
        for n in range(n_t - 1, 0, -1):
            V_old = V.copy()

            # Boundary conditions
            if option_type == "put":
                V[0] = K
                V[-1] = 0
            else:
                V[0] = 0
                V[-1] = S[-1] - K * np.exp(-self.r * t[n - 1])

            # PSOR iteration
            for _ in range(1000):
                V_prev = V.copy()

                for j in range(1, n_S - 1):
                    # Gauss-Seidel update
                    rhs = V_old[j]
                    if j > 1:
                        rhs -= a[j - 1] * V[j - 1]
                    if j < n_S - 2:
                        rhs -= c[j - 1] * V_prev[j + 1]

                    V_gs = rhs / b[j - 1]

                    # SOR relaxation
                    V_sor = omega * V_gs + (1 - omega) * V_prev[j]

                    # Project to satisfy early exercise constraint
                    V[j] = max(V_sor, intrinsic[j])

                # Check convergence
                if np.max(np.abs(V - V_prev)) < tol:
                    break

        return S, V
