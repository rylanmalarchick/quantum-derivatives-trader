"""
Tests for Greeks validation module.

Tests analytical Greeks against known values, PINN Greeks computation
(shape and sign checks), and integration tests with trained models.
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm

from src.validation.greeks import (
    compute_pinn_delta,
    compute_pinn_gamma,
    compute_pinn_theta,
    compute_pinn_vega,
    compute_all_greeks,
    analytical_delta,
    analytical_gamma,
    analytical_theta,
    analytical_vega,
    validate_greeks,
    GreeksValidationResult,
)
from src.classical.pinn import PINN
from src.pde.black_scholes import BSParams


class TestAnalyticalGreeks:
    """Tests for analytical Greeks formulas against known values."""

    def test_analytical_delta_call_atm(self):
        """ATM call delta should be approximately 0.5 + small adjustment."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = analytical_delta(S, K, T, r, sigma, option_type="call")
        
        # ATM call delta is slightly above 0.5 due to drift
        assert 0.5 < delta[0] < 0.7
        
    def test_analytical_delta_call_deep_itm(self):
        """Deep ITM call delta should approach 1."""
        S = np.array([150.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = analytical_delta(S, K, T, r, sigma, option_type="call")
        
        assert delta[0] > 0.95

    def test_analytical_delta_call_deep_otm(self):
        """Deep OTM call delta should approach 0."""
        S = np.array([50.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = analytical_delta(S, K, T, r, sigma, option_type="call")
        
        assert delta[0] < 0.05

    def test_analytical_delta_put_atm(self):
        """ATM put delta should be approximately -0.5 + small adjustment."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta = analytical_delta(S, K, T, r, sigma, option_type="put")
        
        # ATM put delta is slightly above -0.5
        assert -0.5 < delta[0] < -0.3

    def test_analytical_delta_put_call_parity(self):
        """Delta(call) - Delta(put) = 1 (put-call parity for delta)."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        delta_call = analytical_delta(S, K, T, r, sigma, option_type="call")
        delta_put = analytical_delta(S, K, T, r, sigma, option_type="put")
        
        np.testing.assert_allclose(delta_call - delta_put, 1.0, rtol=1e-10)

    def test_analytical_gamma_positive(self):
        """Gamma should always be positive."""
        S = np.array([50.0, 75.0, 100.0, 125.0, 150.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        gamma = analytical_gamma(S, K, T, r, sigma)
        
        assert np.all(gamma > 0)

    def test_analytical_gamma_peak_atm(self):
        """Gamma should peak near ATM."""
        S = np.linspace(50, 150, 101)
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        gamma = analytical_gamma(S, K, T, r, sigma)
        
        # Max gamma should be near S=100
        max_idx = np.argmax(gamma)
        assert 35 <= max_idx <= 65  # Roughly centered

    def test_analytical_gamma_known_value(self):
        """Test gamma against hand-calculated value."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        # d1 = (ln(1) + (0.05 + 0.02)*1) / (0.2*1) = 0.35
        d1 = 0.35
        expected_gamma = norm.pdf(d1) / (100 * 0.2 * 1.0)
        
        gamma = analytical_gamma(S, K, T, r, sigma)
        
        np.testing.assert_allclose(gamma[0], expected_gamma, rtol=1e-6)

    def test_analytical_theta_call_negative(self):
        """Theta for long call should be negative (time decay)."""
        S = np.array([80.0, 100.0, 120.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        theta = analytical_theta(S, K, T, r, sigma, option_type="call")
        
        # Theta should be negative for long positions
        assert np.all(theta < 0)

    def test_analytical_theta_increases_near_expiry(self):
        """Theta magnitude should increase as expiry approaches (for ATM)."""
        S = np.array([100.0])
        K = 100.0
        r = 0.05
        sigma = 0.2
        
        theta_1y = analytical_theta(S, K, 1.0, r, sigma, tau=np.array([1.0]))
        theta_1m = analytical_theta(S, K, 1.0, r, sigma, tau=np.array([1/12]))
        
        # Magnitude of theta increases near expiry
        assert abs(theta_1m[0]) > abs(theta_1y[0])

    def test_analytical_vega_positive(self):
        """Vega should always be positive."""
        S = np.array([50.0, 75.0, 100.0, 125.0, 150.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        vega = analytical_vega(S, K, T, r, sigma)
        
        assert np.all(vega > 0)

    def test_analytical_vega_peak_atm(self):
        """Vega should peak near ATM."""
        S = np.linspace(50, 150, 101)
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        vega = analytical_vega(S, K, T, r, sigma)
        
        # Max vega should be near S=100
        max_idx = np.argmax(vega)
        assert 40 < max_idx < 60

    def test_analytical_vega_known_value(self):
        """Test vega against hand-calculated value."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        # d1 = 0.35
        d1 = 0.35
        expected_vega = 100 * norm.pdf(d1) * np.sqrt(1.0)
        
        vega = analytical_vega(S, K, T, r, sigma)
        
        np.testing.assert_allclose(vega[0], expected_vega, rtol=1e-6)

    def test_analytical_greeks_with_tau_array(self):
        """Test that tau array is handled correctly."""
        S = np.array([100.0, 100.0, 100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        tau = np.array([0.25, 0.5, 1.0])
        
        delta = analytical_delta(S, K, T, r, sigma, tau=tau)
        gamma = analytical_gamma(S, K, T, r, sigma, tau=tau)
        vega = analytical_vega(S, K, T, r, sigma, tau=tau)
        
        # All should have correct shape
        assert delta.shape == (3,)
        assert gamma.shape == (3,)
        assert vega.shape == (3,)
        
        # Gamma should be higher for shorter tau (more curvature near expiry)
        assert gamma[0] > gamma[2]
        
        # Vega should be lower for shorter tau
        assert vega[0] < vega[2]


class TestPINNGreeksComputation:
    """Tests for PINN Greeks computation (shape and sign checks)."""

    @pytest.fixture
    def simple_pinn(self, device):
        """Create a simple PINN for testing."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        model.to(device)
        return model

    def test_compute_pinn_delta_shape(self, simple_pinn, bs_params, device):
        """Delta computation should return correct shape."""
        S = torch.tensor([80.0, 100.0, 120.0], device=device)
        t = torch.tensor([0.0, 0.5, 0.5], device=device)
        
        delta = compute_pinn_delta(simple_pinn, S, t, bs_params)
        
        assert delta.shape == (3,)
        assert not torch.isnan(delta).any()

    def test_compute_pinn_gamma_shape(self, simple_pinn, bs_params, device):
        """Gamma computation should return correct shape."""
        S = torch.tensor([80.0, 100.0, 120.0], device=device)
        t = torch.tensor([0.0, 0.5, 0.5], device=device)
        
        gamma = compute_pinn_gamma(simple_pinn, S, t, bs_params)
        
        assert gamma.shape == (3,)
        assert not torch.isnan(gamma).any()

    def test_compute_pinn_theta_shape(self, simple_pinn, bs_params, device):
        """Theta computation should return correct shape."""
        S = torch.tensor([80.0, 100.0, 120.0], device=device)
        t = torch.tensor([0.0, 0.5, 0.5], device=device)
        
        theta = compute_pinn_theta(simple_pinn, S, t, bs_params)
        
        assert theta.shape == (3,)
        assert not torch.isnan(theta).any()

    def test_compute_all_greeks_shape(self, simple_pinn, bs_params, device):
        """compute_all_greeks should return dict with correct shapes."""
        S = torch.tensor([80.0, 100.0, 120.0], device=device)
        t = torch.tensor([0.0, 0.5, 0.5], device=device)
        
        greeks = compute_all_greeks(simple_pinn, S, t, bs_params)
        
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        
        for key in ["delta", "gamma", "theta"]:
            assert greeks[key].shape == (3,)
            assert not torch.isnan(greeks[key]).any()

    def test_compute_all_greeks_no_vega_without_sigma_fn(self, simple_pinn, bs_params, device):
        """Vega should not be in result if forward_with_sigma not provided."""
        S = torch.tensor([100.0], device=device)
        t = torch.tensor([0.0], device=device)
        
        greeks = compute_all_greeks(simple_pinn, S, t, bs_params)
        
        assert "vega" not in greeks

    def test_pinn_delta_increases_with_spot(self, simple_pinn, bs_params, device):
        """For a reasonable PINN, delta should generally increase with S for calls."""
        # Note: This is a weak test since the PINN is untrained
        S = torch.linspace(50.0, 150.0, 11, device=device)
        t = torch.zeros_like(S)
        
        delta = compute_pinn_delta(simple_pinn, S, t, bs_params)
        
        # Just check we get reasonable values (finite, correct shape)
        assert delta.shape == (11,)
        assert not torch.isnan(delta).any()
        assert not torch.isinf(delta).any()

    def test_compute_pinn_vega_with_wrapper(self, simple_pinn, bs_params, device):
        """Test vega computation with a custom forward function."""
        S = torch.tensor([100.0], device=device)
        t = torch.tensor([0.0], device=device)
        sigma = torch.tensor([0.2], device=device)
        
        # Simple wrapper that doesn't actually use sigma (just for testing the interface)
        def forward_with_sigma(S, t, sigma):
            return simple_pinn(S, t) * sigma / sigma  # Identity transformation
        
        vega = compute_pinn_vega(simple_pinn, S, t, sigma, forward_with_sigma)
        
        assert vega.shape == (1,)
        assert not torch.isnan(vega).any()

    def test_greeks_detached(self, simple_pinn, bs_params, device):
        """Returned Greeks should be detached from computation graph."""
        S = torch.tensor([100.0], device=device)
        t = torch.tensor([0.0], device=device)
        
        greeks = compute_all_greeks(simple_pinn, S, t, bs_params)
        
        for key in ["delta", "gamma", "theta"]:
            assert not greeks[key].requires_grad


class TestValidateGreeks:
    """Tests for the validate_greeks function."""

    @pytest.fixture
    def simple_pinn(self, device):
        """Create a simple PINN for testing."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        model.to(device)
        return model

    def test_validate_greeks_returns_result(self, simple_pinn, bs_params, device):
        """validate_greeks should return a GreeksValidationResult."""
        result = validate_greeks(
            simple_pinn,
            bs_params,
            S_range=(80.0, 120.0),
            n_spots=10,
            device=device,
        )
        
        assert isinstance(result, GreeksValidationResult)
        assert result.delta_mae >= 0
        assert result.delta_max_error >= 0
        assert result.gamma_mae >= 0
        assert result.gamma_max_error >= 0
        assert result.theta_mae >= 0
        assert result.theta_max_error >= 0

    def test_validate_greeks_max_error_ge_mae(self, simple_pinn, bs_params, device):
        """Max error should always be >= MAE."""
        result = validate_greeks(
            simple_pinn,
            bs_params,
            S_range=(80.0, 120.0),
            n_spots=10,
            device=device,
        )
        
        assert result.delta_max_error >= result.delta_mae
        assert result.gamma_max_error >= result.gamma_mae
        assert result.theta_max_error >= result.theta_mae

    def test_validate_greeks_custom_t_values(self, simple_pinn, bs_params, device):
        """Should work with custom t_values."""
        t_values = np.array([0.0, 0.5])
        
        result = validate_greeks(
            simple_pinn,
            bs_params,
            S_range=(80.0, 120.0),
            t_values=t_values,
            n_spots=5,
            device=device,
        )
        
        assert isinstance(result, GreeksValidationResult)

    def test_validate_greeks_repr(self, simple_pinn, bs_params, device):
        """GreeksValidationResult should have nice repr."""
        result = validate_greeks(
            simple_pinn,
            bs_params,
            S_range=(80.0, 120.0),
            n_spots=5,
            device=device,
        )
        
        repr_str = repr(result)
        assert "Delta" in repr_str
        assert "Gamma" in repr_str
        assert "Theta" in repr_str
        assert "MAE" in repr_str
        assert "Max" in repr_str


class TestGreeksIntegration:
    """Integration tests with trained PINN (if checkpoint available)."""

    @pytest.fixture
    def checkpoint_path(self):
        """Path to trained BS PINN checkpoint."""
        import os
        paths = [
            "checkpoints/bs_pinn_best.pt",
            "checkpoints/bs_pinn.pt",
            "models/bs_pinn.pt",
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def test_trained_pinn_delta_range(self, checkpoint_path, bs_params, device):
        """Trained PINN delta should be in [0, 1] for calls."""
        if checkpoint_path is None:
            pytest.skip("No trained PINN checkpoint found")
        
        model = PINN(hidden_dims=[64, 64, 64, 64], S_max=200.0, T_max=1.0)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        S = torch.linspace(50.0, 150.0, 21, device=device)
        t = torch.zeros_like(S)
        
        delta = compute_pinn_delta(model, S, t, bs_params)
        
        # For a well-trained call option PINN, delta should be in [0, 1]
        assert torch.all(delta >= -0.1)  # Allow small numerical errors
        assert torch.all(delta <= 1.1)

    def test_trained_pinn_gamma_positive(self, checkpoint_path, bs_params, device):
        """Trained PINN gamma should be positive."""
        if checkpoint_path is None:
            pytest.skip("No trained PINN checkpoint found")
        
        model = PINN(hidden_dims=[64, 64, 64, 64], S_max=200.0, T_max=1.0)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        S = torch.linspace(50.0, 150.0, 21, device=device)
        t = torch.zeros_like(S)
        
        gamma = compute_pinn_gamma(model, S, t, bs_params)
        
        # For a well-trained PINN, gamma should be mostly positive
        # Allow some small negative values due to numerical issues
        assert torch.sum(gamma < -0.01) < len(gamma) // 4

    def test_trained_pinn_validation_errors(self, checkpoint_path, bs_params, device):
        """Trained PINN should have reasonable validation errors."""
        if checkpoint_path is None:
            pytest.skip("No trained PINN checkpoint found")
        
        model = PINN(hidden_dims=[64, 64, 64, 64], S_max=200.0, T_max=1.0)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        result = validate_greeks(
            model,
            bs_params,
            S_range=(70.0, 130.0),
            n_spots=30,
            device=device,
        )
        
        # For a well-trained PINN, errors should be reasonable
        # These thresholds are illustrative; adjust based on actual training quality
        print(f"\nValidation result: {result}")
        
        # Just ensure we got valid numbers
        assert result.delta_mae < 1.0
        assert result.gamma_mae < 1.0
        assert result.theta_mae < 100.0  # Theta has different scale


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_analytical_greeks_near_expiry(self):
        """Greeks should handle near-expiry correctly."""
        S = np.array([100.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        tau = np.array([0.001])  # Very close to expiry
        
        delta = analytical_delta(S, K, T, r, sigma, tau=tau)
        gamma = analytical_gamma(S, K, T, r, sigma, tau=tau)
        theta = analytical_theta(S, K, T, r, sigma, tau=tau)
        vega = analytical_vega(S, K, T, r, sigma, tau=tau)
        
        # Should not have NaN or Inf
        assert np.isfinite(delta).all()
        assert np.isfinite(gamma).all()
        assert np.isfinite(theta).all()
        assert np.isfinite(vega).all()

    def test_analytical_greeks_deep_itm_otm(self):
        """Greeks should handle deep ITM/OTM correctly."""
        S_deep_itm = np.array([200.0])
        S_deep_otm = np.array([10.0])
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        
        # Deep ITM call
        delta_itm = analytical_delta(S_deep_itm, K, T, r, sigma)
        gamma_itm = analytical_gamma(S_deep_itm, K, T, r, sigma)
        
        # Deep OTM call
        delta_otm = analytical_delta(S_deep_otm, K, T, r, sigma)
        gamma_otm = analytical_gamma(S_deep_otm, K, T, r, sigma)
        
        # Deep ITM: delta near 1, gamma near 0
        assert delta_itm[0] > 0.99
        assert gamma_itm[0] < 0.001
        
        # Deep OTM: delta near 0, gamma near 0
        assert delta_otm[0] < 0.01
        assert gamma_otm[0] < 0.001

    def test_pinn_greeks_batch_sizes(self, device):
        """PINN Greeks should work with various batch sizes."""
        model = PINN(hidden_dims=[16, 16], S_max=200.0, T_max=1.0)
        model.to(device)
        
        params = BSParams(r=0.05, sigma=0.2, K=100.0, T=1.0)
        
        for batch_size in [1, 5, 100]:
            S = torch.rand(batch_size, device=device) * 100 + 50
            t = torch.rand(batch_size, device=device)
            
            greeks = compute_all_greeks(model, S, t, params)
            
            assert greeks["delta"].shape == (batch_size,)
            assert greeks["gamma"].shape == (batch_size,)
            assert greeks["theta"].shape == (batch_size,)

    def test_validation_result_with_vega(self):
        """Test GreeksValidationResult with vega values."""
        result = GreeksValidationResult(
            delta_mae=0.01,
            delta_max_error=0.05,
            gamma_mae=0.001,
            gamma_max_error=0.005,
            theta_mae=0.5,
            theta_max_error=2.0,
            vega_mae=0.1,
            vega_max_error=0.5,
        )
        
        repr_str = repr(result)
        assert "Vega" in repr_str
        assert "0.1" in repr_str or "0.10" in repr_str
