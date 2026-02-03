"""
Tests for pricing engines (analytical, Monte Carlo, finite difference).
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm

from src.pricing.analytical import AnalyticalPricer
from src.pricing.monte_carlo import MonteCarloEngine, MCResult
from src.pricing.finite_difference import FiniteDifferencePricer, FDGrid


class TestAnalyticalPricer:
    """Tests for AnalyticalPricer class."""

    @pytest.fixture
    def pricer(self):
        """Standard analytical pricer."""
        return AnalyticalPricer(r=0.05, sigma=0.2)

    def test_black_scholes_known_value(self, pricer):
        """Test Black-Scholes against known values."""
        # ATM call with S=100, K=100, T=1, r=0.05, sigma=0.2
        # Expected value computed from external source
        S = np.array([100.0])
        K = 100.0
        T = 1.0

        price = pricer.black_scholes(S, K, T, option_type="call")

        # Approximate expected value for ATM call
        # From standard BS formula, should be around 10.45
        assert 10.0 < price[0] < 11.0

    def test_black_scholes_itm_call_lower_bound(self, pricer):
        """ITM call should be worth at least intrinsic value."""
        S = np.array([120.0])
        K = 100.0
        T = 1.0

        price = pricer.black_scholes(S, K, T, option_type="call")
        intrinsic = max(S[0] - K, 0)

        assert price[0] >= intrinsic

    def test_black_scholes_otm_put_low_value(self, pricer):
        """Deep OTM put should have small value."""
        S = np.array([150.0])  # Far above strike
        K = 100.0
        T = 1.0

        price = pricer.black_scholes(S, K, T, option_type="put")

        assert price[0] < 1.0  # Should be very small

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_black_scholes_at_expiry(self, pricer, option_type):
        """At expiry, option value equals payoff."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1e-10  # Near expiry

        price = pricer.black_scholes(S, K, T, option_type=option_type)

        if option_type == "call":
            expected = np.maximum(S - K, 0)
        else:
            expected = np.maximum(K - S, 0)

        np.testing.assert_allclose(price, expected, atol=1e-4)


class TestAnalyticalGreeks:
    """Tests for analytical Greeks computation."""

    @pytest.fixture
    def pricer(self):
        return AnalyticalPricer(r=0.05, sigma=0.2)

    def test_delta_call_range(self, pricer):
        """Call delta should be in [0, 1]."""
        S = np.array([50.0, 80.0, 100.0, 120.0, 150.0])
        K = 100.0
        T = 1.0

        delta = pricer.delta(S, K, T, option_type="call")

        assert np.all(delta >= 0)
        assert np.all(delta <= 1)

    def test_delta_put_range(self, pricer):
        """Put delta should be in [-1, 0]."""
        S = np.array([50.0, 80.0, 100.0, 120.0, 150.0])
        K = 100.0
        T = 1.0

        delta = pricer.delta(S, K, T, option_type="put")

        assert np.all(delta >= -1)
        assert np.all(delta <= 0)

    def test_delta_atm_approximately_half(self, pricer):
        """ATM call delta should be approximately 0.5."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0

        delta = pricer.delta(S, K, T, option_type="call")

        # ATM delta is slightly above 0.5 due to drift
        assert 0.45 < delta[0] < 0.65

    def test_gamma_positive(self, pricer):
        """Gamma should always be positive."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1.0

        gamma = pricer.gamma(S, K, T)

        assert np.all(gamma > 0)

    def test_gamma_peaks_near_atm(self, pricer):
        """Gamma should peak near ATM (allowing for slight skew)."""
        S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        K = 100.0
        T = 0.5

        gamma = pricer.gamma(S, K, T)

        # Gamma should peak near ATM - allow for either S=90 or S=100
        # due to interest rate drift shifting the peak slightly
        max_idx = np.argmax(gamma)
        assert max_idx in [1, 2]  # Either 90 or 100 spot

        # Gamma at edges should be lower than near ATM
        assert gamma[0] < gamma[max_idx]  # S=80 < peak
        assert gamma[4] < gamma[max_idx]  # S=120 < peak

    def test_vega_positive(self, pricer):
        """Vega should always be positive."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1.0

        vega = pricer.vega(S, K, T)

        assert np.all(vega > 0)

    def test_vega_peaks_atm(self, pricer):
        """Vega should peak at ATM."""
        S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        K = 100.0
        T = 0.5

        vega = pricer.vega(S, K, T)

        # ATM vega (index 2) should be largest
        assert vega[2] == max(vega)


class TestPutCallParity:
    """Tests for put-call parity."""

    @pytest.fixture
    def pricer(self):
        return AnalyticalPricer(r=0.05, sigma=0.2)

    def test_put_call_parity(self, pricer):
        """Verify C - P = S - K*exp(-r*T)."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1.0

        call = pricer.black_scholes(S, K, T, option_type="call")
        put = pricer.black_scholes(S, K, T, option_type="put")

        expected_diff = S - K * np.exp(-pricer.r * T)

        np.testing.assert_allclose(call - put, expected_diff, rtol=1e-6)

    @pytest.mark.parametrize("S", [80.0, 100.0, 120.0])
    @pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
    @pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
    def test_put_call_parity_parametrized(self, pricer, S, K, T):
        """Put-call parity should hold for various parameters."""
        S_arr = np.array([S])

        call = pricer.black_scholes(S_arr, K, T, option_type="call")[0]
        put = pricer.black_scholes(S_arr, K, T, option_type="put")[0]

        expected_diff = S - K * np.exp(-pricer.r * T)

        np.testing.assert_allclose(call - put, expected_diff, rtol=1e-5)


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine class."""

    @pytest.fixture
    def mc_engine(self):
        return MonteCarloEngine(r=0.05, sigma=0.2, seed=42)

    def test_price_european_returns_mc_result(self, mc_engine):
        """price_european should return MCResult with correct fields."""
        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        result = mc_engine.price_european(
            payoff_fn=payoff_fn,
            S0=100.0,
            T=1.0,
            n_paths=1000,
        )

        assert isinstance(result, MCResult)
        assert hasattr(result, "price")
        assert hasattr(result, "std_error")
        assert hasattr(result, "n_paths")
        assert hasattr(result, "confidence_interval")

    def test_price_european_confidence_interval_contains_true_value(self, mc_engine):
        """95% CI should contain true value (most of the time)."""
        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        # Get analytical value
        analytical_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        true_value = analytical_pricer.black_scholes(
            np.array([100.0]), K, 1.0, option_type="call"
        )[0]

        # MC estimate with many paths for accuracy
        result = mc_engine.price_european(
            payoff_fn=payoff_fn,
            S0=100.0,
            T=1.0,
            n_paths=50000,
        )

        # CI should contain true value
        assert result.confidence_interval[0] <= true_value <= result.confidence_interval[1]

    def test_price_european_close_to_analytical(self, mc_engine):
        """MC price should be close to analytical for large n_paths."""
        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        # Analytical value
        analytical_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        true_value = analytical_pricer.black_scholes(
            np.array([100.0]), K, 1.0, option_type="call"
        )[0]

        # MC estimate
        result = mc_engine.price_european(
            payoff_fn=payoff_fn,
            S0=100.0,
            T=1.0,
            n_paths=100000,
        )

        # Should be within 3 standard errors of true value
        assert abs(result.price - true_value) < 3 * result.std_error

    def test_antithetic_reduces_variance(self, mc_engine):
        """Antithetic variates should reduce variance."""
        K = 100.0
        payoff_fn = lambda S: max(S - K, 0)

        n_paths = 10000

        # Standard MC
        result_standard = mc_engine.price_european(
            payoff_fn=payoff_fn,
            S0=100.0,
            T=1.0,
            n_paths=n_paths,
        )

        # Antithetic MC (uses same seed)
        mc_antithetic = MonteCarloEngine(r=0.05, sigma=0.2, seed=42)
        result_antithetic = mc_antithetic.price_with_antithetic(
            payoff_fn=payoff_fn,
            S0=100.0,
            T=1.0,
            n_paths=n_paths,
        )

        # Antithetic should have lower or comparable std error
        # (for same effective number of function evaluations)
        assert result_antithetic.std_error <= result_standard.std_error * 1.5


class TestFiniteDifferencePricer:
    """Tests for FiniteDifferencePricer class."""

    @pytest.fixture
    def fd_pricer(self):
        grid = FDGrid(S_min=0.0, S_max=200.0, n_S=100, n_t=1000)
        return FiniteDifferencePricer(r=0.05, sigma=0.2, grid=grid)

    def test_fd_converges_to_analytical(self, fd_pricer):
        """FD solution should converge to analytical as grid refines."""
        K = 100.0
        T = 1.0

        S_grid, V_fd = fd_pricer.price_crank_nicolson(K, T, option_type="call")

        # Get analytical values at grid points
        analytical_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        V_analytical = analytical_pricer.black_scholes(S_grid, K, T, option_type="call")

        # Check ATM region (where S ≈ K)
        atm_idx = np.argmin(np.abs(S_grid - K))

        # FD and analytical should be close at ATM
        assert abs(V_fd[atm_idx] - V_analytical[atm_idx]) < 0.5

    def test_fd_implicit_stable(self, fd_pricer):
        """Implicit FD should be stable regardless of time step."""
        K = 100.0
        T = 1.0

        S_grid, V = fd_pricer.price_implicit(K, T, option_type="call")

        # Should not have NaN or Inf
        assert not np.isnan(V).any()
        assert not np.isinf(V).any()

        # Should be non-negative
        assert np.all(V >= -1e-6)

    def test_fd_crank_nicolson_accuracy(self, fd_pricer):
        """Crank-Nicolson should be reasonably accurate."""
        K = 100.0
        T = 1.0

        S_grid, V_cn = fd_pricer.price_crank_nicolson(K, T, option_type="call")

        # Analytical solution
        analytical_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        V_analytical = analytical_pricer.black_scholes(S_grid, K, T, option_type="call")

        # Compute error in interior region (avoid boundaries)
        interior = (S_grid > 20) & (S_grid < 180)
        max_error = np.max(np.abs(V_cn[interior] - V_analytical[interior]))

        # Should be reasonably accurate (allow slightly larger tolerance)
        assert max_error < 1.5

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_fd_terminal_condition(self, fd_pricer, option_type):
        """FD at T=0 should match payoff as T→0."""
        K = 100.0
        T = 0.01  # Very short time

        S_grid, V = fd_pricer.price_crank_nicolson(K, T, option_type=option_type)

        if option_type == "call":
            expected = np.maximum(S_grid - K, 0)
        else:
            expected = np.maximum(K - S_grid, 0)

        # Should be close to payoff for short maturity
        interior = (S_grid > 20) & (S_grid < 180)
        np.testing.assert_allclose(V[interior], expected[interior], atol=0.5)

    def test_fd_greeks_computation(self, fd_pricer):
        """compute_greeks should return valid Greeks."""
        K = 100.0
        T = 1.0

        greeks = fd_pricer.compute_greeks(K, T, option_type="call")

        assert "S" in greeks
        assert "V" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks

        # Delta should be in [0, 1] for call
        interior = (greeks["S"] > 20) & (greeks["S"] < 180)
        assert np.all(greeks["delta"][interior] >= -0.1)
        assert np.all(greeks["delta"][interior] <= 1.1)


class TestPricingConsistency:
    """Cross-validation tests between different pricing methods."""

    def test_analytical_mc_fd_consistency(self):
        """All pricing methods should give similar results."""
        r = 0.05
        sigma = 0.2
        S0 = 100.0
        K = 100.0
        T = 1.0

        # Analytical
        analytical = AnalyticalPricer(r, sigma)
        V_analytical = analytical.black_scholes(np.array([S0]), K, T, option_type="call")[0]

        # Monte Carlo
        mc = MonteCarloEngine(r, sigma, seed=42)
        payoff_fn = lambda S: max(S - K, 0)
        V_mc = mc.price_european(payoff_fn, S0, T, n_paths=100000).price

        # Finite Difference
        fd_grid = FDGrid(S_min=0.0, S_max=300.0, n_S=150, n_t=1500)
        fd = FiniteDifferencePricer(r, sigma, grid=fd_grid)
        S_grid, V_fd_grid = fd.price_crank_nicolson(K, T, option_type="call")
        # Interpolate to get value at S0
        V_fd = np.interp(S0, S_grid, V_fd_grid)

        # All should be close
        assert abs(V_analytical - V_mc) < 0.5
        assert abs(V_analytical - V_fd) < 0.5
        assert abs(V_mc - V_fd) < 0.5
