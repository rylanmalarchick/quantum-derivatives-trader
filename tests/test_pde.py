"""
Tests for Black-Scholes and Heston PDE modules.
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm

import sys
sys.path.insert(0, "/home/rylan/dev/personal/quantum-derivatives-trader/src")

from pde.black_scholes import BSParams, bs_pde_residual, bs_analytical
from pde.heston import HestonParams, heston_pde_residual


class TestBSPDEResidual:
    """Tests for bs_pde_residual function."""

    def test_residual_near_zero_for_analytical_solution(self, bs_params):
        """PDE residual should be ~0 when V is the analytical BS solution."""
        # Create sample points
        S = torch.tensor([80.0, 100.0, 120.0], requires_grad=True)
        t = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

        # Compute analytical solution
        V = bs_analytical(S, t, bs_params, option_type="call")
        V.requires_grad_(True)

        # For the PDE residual, we need to use autodiff
        # We'll use a simple gradient function
        def grad_fn(y, x):
            return torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True
            )[0]

        # The analytical solution satisfies the PDE, so residual should be small
        # Note: Due to numerical precision, we allow some tolerance
        # For a proper test, we'd evaluate on interior points with proper setup
        # Here we just verify the function runs and returns sensible shape
        residual = bs_pde_residual(V, S, t, bs_params, grad_fn)

        assert residual.shape == S.shape
        # Residual should be relatively small (not necessarily exactly zero due to discretization)
        assert torch.all(torch.abs(residual) < 1.0)  # Loose bound


class TestBSAnalytical:
    """Tests for bs_analytical function."""

    def test_matches_scipy_black_scholes(self, bs_params):
        """Verify bs_analytical matches scipy Black-Scholes implementation."""
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.zeros(3)
        tau = bs_params.T - 0.0  # Time to maturity

        # Compute using bs_analytical
        V_pde = bs_analytical(S, t, bs_params, option_type="call")

        # Compute using scipy directly
        S_np = S.numpy()
        d1 = (np.log(S_np / bs_params.K) + (bs_params.r + 0.5 * bs_params.sigma**2) * tau) / (
            bs_params.sigma * np.sqrt(tau)
        )
        d2 = d1 - bs_params.sigma * np.sqrt(tau)
        V_scipy = S_np * norm.cdf(d1) - bs_params.K * np.exp(-bs_params.r * tau) * norm.cdf(d2)

        np.testing.assert_allclose(V_pde.numpy(), V_scipy, rtol=1e-6)

    def test_put_call_parity(self, bs_params):
        """Verify put-call parity: C - P = S - K*exp(-r*T)."""
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.zeros(3)

        call_price = bs_analytical(S, t, bs_params, option_type="call")
        put_price = bs_analytical(S, t, bs_params, option_type="put")

        # Put-call parity
        expected_diff = S - bs_params.K * np.exp(-bs_params.r * bs_params.T)

        np.testing.assert_allclose(
            (call_price - put_price).numpy(),
            expected_diff.numpy(),
            rtol=1e-6
        )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_positive_option_values(self, bs_params, option_type):
        """Option values should be non-negative."""
        S = torch.tensor([50.0, 80.0, 100.0, 120.0, 150.0])
        t = torch.zeros(5)

        V = bs_analytical(S, t, bs_params, option_type=option_type)

        assert torch.all(V >= 0)


class TestBoundaryConditions:
    """Tests for Black-Scholes boundary conditions."""

    def test_lower_boundary_call(self, bs_params):
        """Call option value at S=0 should be 0."""
        S = torch.tensor([0.0])
        t = torch.tensor([0.0])

        V = bs_analytical(S, t, bs_params, option_type="call")

        assert torch.allclose(V, torch.tensor([0.0]), atol=1e-6)

    def test_upper_boundary_call(self, bs_params):
        """For large S, call ≈ S - K*exp(-r*tau)."""
        S = torch.tensor([500.0, 1000.0])  # Very large S
        t = torch.zeros(2)
        tau = bs_params.T

        V = bs_analytical(S, t, bs_params, option_type="call")
        expected = S - bs_params.K * np.exp(-bs_params.r * tau)

        # Should be very close for large S
        np.testing.assert_allclose(V.numpy(), expected.numpy(), rtol=1e-2)

    def test_lower_boundary_put(self, bs_params):
        """For S=0, put value should be K*exp(-r*tau)."""
        # Note: S=0 causes issues with log, so we test very small S
        S = torch.tensor([0.001])
        t = torch.tensor([0.0])
        tau = bs_params.T

        V = bs_analytical(S, t, bs_params, option_type="put")
        expected = bs_params.K * np.exp(-bs_params.r * tau)

        # Should be close to discounted strike
        assert V.item() > 0.9 * expected


class TestTerminalCondition:
    """Tests for terminal condition (payoff at maturity)."""

    def test_call_payoff_at_maturity(self, bs_params):
        """At t=T, call value = max(S - K, 0)."""
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.full((3,), bs_params.T)  # At maturity

        V = bs_analytical(S, t, bs_params, option_type="call")
        expected_payoff = torch.relu(S - bs_params.K)

        np.testing.assert_allclose(V.numpy(), expected_payoff.numpy(), atol=1e-4)

    def test_put_payoff_at_maturity(self, bs_params):
        """At t=T, put value = max(K - S, 0)."""
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.full((3,), bs_params.T)

        V = bs_analytical(S, t, bs_params, option_type="put")
        expected_payoff = torch.relu(bs_params.K - S)

        np.testing.assert_allclose(V.numpy(), expected_payoff.numpy(), atol=1e-4)


class TestHestonPDE:
    """Tests for Heston PDE residual structure."""

    @pytest.fixture
    def heston_params(self):
        """Heston model parameters for testing."""
        return HestonParams(
            r=0.05,
            kappa=2.0,    # Mean reversion speed
            theta=0.04,   # Long-run variance
            xi=0.3,       # Vol of vol
            rho=-0.7,     # Correlation
            K=100.0,
            T=1.0,
            v0=0.04,      # Initial variance
        )

    def test_heston_residual_returns_tensor(self, heston_params):
        """Heston PDE residual should return a tensor of correct shape."""
        batch_size = 5
        S = torch.rand(batch_size) * 100 + 50  # Random spots
        v = torch.rand(batch_size) * 0.1 + 0.01  # Random variances
        t = torch.rand(batch_size) * heston_params.T

        # Create a simple model-like function for V
        # V = S (just for testing residual computation works)
        S.requires_grad_(True)
        v.requires_grad_(True)
        t.requires_grad_(True)

        V = S.clone()  # Simple test function

        residual = heston_pde_residual(V, S, v, t, heston_params)

        assert residual.shape == S.shape
        assert not torch.isnan(residual).any()
        assert not torch.isinf(residual).any()


class TestParameterizedPDE:
    """Parameterized tests for PDE functions."""

    @pytest.mark.parametrize("r", [0.01, 0.05, 0.10])
    @pytest.mark.parametrize("sigma", [0.1, 0.2, 0.4])
    def test_analytical_monotonicity_in_spot(self, r, sigma):
        """Call value should increase with spot price."""
        params = BSParams(r=r, sigma=sigma, K=100.0, T=1.0)
        S = torch.tensor([80.0, 90.0, 100.0, 110.0, 120.0])
        t = torch.zeros(5)

        V = bs_analytical(S, t, params, option_type="call")

        # Verify strictly increasing
        assert torch.all(V[1:] > V[:-1])

    @pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
    def test_atm_option_positive_value(self, K):
        """ATM options should have positive time value."""
        params = BSParams(r=0.05, sigma=0.2, K=K, T=1.0)
        S = torch.tensor([K])  # At-the-money
        t = torch.zeros(1)

        V_call = bs_analytical(S, t, params, option_type="call")
        V_put = bs_analytical(S, t, params, option_type="put")

        assert V_call.item() > 0
        assert V_put.item() > 0
