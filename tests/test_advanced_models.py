"""
Tests for advanced pricing models:
- Merton jump-diffusion
- Heston stochastic volatility  
- American options with early exercise
"""

import pytest
import torch
import numpy as np

from src.pde.merton import (
    MertonParams, MertonPINN, MertonPINNTrainer,
    merton_analytical_call, merton_pide_residual
)
from src.pde.heston import (
    HestonParams, HestonPINN, HestonPINNTrainer,
    heston_call_price, heston_pde_residual
)
from src.pde.american import (
    AmericanParams, AmericanPINN, AmericanPINNTrainer,
    american_put_binomial, american_call_binomial,
    american_payoff, find_early_exercise_boundary
)


class TestMertonModel:
    """Tests for Merton jump-diffusion."""
    
    def test_merton_params_kappa(self):
        """Verify expected jump size calculation."""
        params = MertonParams(mu_J=-0.10, sigma_J=0.15)
        # κ = exp(μ_J + σ_J²/2) - 1
        expected = np.exp(-0.10 + 0.5 * 0.15**2) - 1
        assert np.isclose(params.kappa, expected, rtol=1e-6)
    
    def test_merton_params_drift(self):
        """Verify risk-neutral drift."""
        params = MertonParams(r=0.05, lam=0.5, mu_J=-0.10, sigma_J=0.15)
        expected = 0.05 - 0.5 * params.kappa
        assert np.isclose(params.drift, expected, rtol=1e-6)
    
    def test_merton_analytical_atm(self):
        """Analytical price should be reasonable for ATM option."""
        params = MertonParams(K=100.0, T=1.0, sigma=0.20)
        price = merton_analytical_call(100.0, params, tau=1.0)
        # Should be positive and reasonable (BS would give ~10)
        assert 5 < price < 25
    
    def test_merton_analytical_deep_itm(self):
        """Deep ITM should approach intrinsic + time value."""
        params = MertonParams(K=100.0)
        price = merton_analytical_call(150.0, params, tau=0.5)
        intrinsic = 50.0
        assert price > intrinsic
    
    def test_merton_analytical_deep_otm(self):
        """Deep OTM should be small but positive."""
        params = MertonParams(K=100.0)
        price = merton_analytical_call(50.0, params, tau=1.0)
        assert 0 < price < 5
    
    def test_merton_analytical_terminal(self):
        """At τ=0, should equal payoff."""
        params = MertonParams(K=100.0)
        S = np.array([80, 100, 120])
        prices = merton_analytical_call(S, params, tau=0)
        expected = np.maximum(S - 100, 0)
        np.testing.assert_allclose(prices, expected, rtol=1e-6)
    
    def test_merton_pinn_forward_shape(self):
        """PINN forward pass returns correct shape."""
        params = MertonParams()
        model = MertonPINN(params, hidden_dims=[32, 32])
        
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.tensor([0.5, 0.5, 0.5])
        V = model(S, t)
        
        assert V.shape == (3,)
    
    def test_merton_pinn_positive_output(self):
        """PINN should output positive values."""
        params = MertonParams()
        model = MertonPINN(params, hidden_dims=[32, 32])
        
        S = torch.rand(100) * 200
        t = torch.rand(100)
        V = model(S, t)
        
        assert (V >= 0).all()
    
    def test_merton_pinn_terminal_condition(self):
        """Terminal condition should match payoff."""
        params = MertonParams(K=100.0)
        model = MertonPINN(params)
        
        S = torch.tensor([80.0, 100.0, 120.0])
        payoff = model.terminal_condition(S)
        
        expected = torch.tensor([0.0, 0.0, 20.0])
        torch.testing.assert_close(payoff, expected)
    
    def test_merton_trainer_loss_decreases(self):
        """Training should reduce loss."""
        params = MertonParams()
        model = MertonPINN(params, hidden_dims=[32, 32])
        trainer = MertonPINNTrainer(model, lr=1e-2, n_quad=10)
        
        # Initial loss
        S_int = torch.rand(100) * 200
        t_int = torch.rand(100)
        S_term = torch.rand(50) * 200
        
        initial_losses = trainer.compute_loss(S_int, t_int, S_term)
        initial_total = initial_losses["total"].item()
        
        # Train briefly
        for _ in range(50):
            trainer.train_step(S_int, t_int, S_term)
        
        final_losses = trainer.compute_loss(S_int, t_int, S_term)
        final_total = final_losses["total"].item()
        
        assert final_total < initial_total


class TestHestonModel:
    """Tests for Heston stochastic volatility."""
    
    def test_heston_params_feller(self):
        """Test Feller condition check."""
        # Should satisfy: 2κθ > ξ²
        params_good = HestonParams(kappa=2.0, theta=0.04, xi=0.3)
        assert params_good.feller_satisfied  # 2*2*0.04 = 0.16 > 0.09
        
        params_bad = HestonParams(kappa=0.5, theta=0.04, xi=0.5)
        assert not params_bad.feller_satisfied  # 2*0.5*0.04 = 0.04 < 0.25
    
    def test_heston_params_initial_vol(self):
        """Initial vol should be sqrt of v0."""
        params = HestonParams(v0=0.04)
        assert np.isclose(params.initial_vol, 0.2)
    
    def test_heston_call_price_atm(self):
        """Heston price should be reasonable for ATM."""
        params = HestonParams(K=100.0, T=1.0, v0=0.04)
        price = heston_call_price(100.0, params, tau=1.0)
        # Should be similar to BS with 20% vol
        assert 5 < price < 20
    
    def test_heston_call_price_terminal(self):
        """At τ→0, should approach payoff."""
        params = HestonParams(K=100.0)
        
        # ITM
        price_itm = heston_call_price(120.0, params, tau=0.001)
        assert abs(price_itm - 20.0) < 1.0
        
        # OTM
        price_otm = heston_call_price(80.0, params, tau=0.001)
        assert price_otm < 1.0
    
    def test_heston_pinn_forward_shape(self):
        """PINN forward pass returns correct shape."""
        params = HestonParams()
        model = HestonPINN(params, hidden_dims=[32, 32])
        
        S = torch.tensor([80.0, 100.0, 120.0])
        v = torch.tensor([0.04, 0.04, 0.04])
        t = torch.tensor([0.5, 0.5, 0.5])
        V = model(S, v, t)
        
        assert V.shape == (3,)
    
    def test_heston_pinn_positive_output(self):
        """PINN should output positive values."""
        params = HestonParams()
        model = HestonPINN(params, hidden_dims=[32, 32])
        
        S = torch.rand(100) * 200
        v = torch.rand(100) * 0.1 + 0.01
        t = torch.rand(100)
        V = model(S, v, t)
        
        assert (V >= 0).all()
    
    def test_heston_pinn_terminal_condition(self):
        """Terminal condition should match payoff."""
        params = HestonParams(K=100.0)
        model = HestonPINN(params)
        
        S = torch.tensor([80.0, 100.0, 120.0])
        payoff = model.terminal_condition(S)
        
        expected = torch.tensor([0.0, 0.0, 20.0])
        torch.testing.assert_close(payoff, expected)
    
    def test_heston_trainer_loss_decreases(self):
        """Training should reduce loss."""
        params = HestonParams()
        model = HestonPINN(params, hidden_dims=[32, 32])
        trainer = HestonPINNTrainer(model, lr=1e-2)
        
        device = next(model.parameters()).device
        
        S_int = torch.rand(100, device=device) * 200
        v_int = trainer.sample_variance(100, device)
        t_int = torch.rand(100, device=device)
        S_term = torch.rand(50, device=device) * 200
        v_term = trainer.sample_variance(50, device)
        
        initial_losses = trainer.compute_loss(S_int, v_int, t_int, S_term, v_term)
        initial_total = initial_losses["total"].item()
        
        for _ in range(50):
            trainer.train_step(S_int, v_int, t_int, S_term, v_term)
        
        final_losses = trainer.compute_loss(S_int, v_int, t_int, S_term, v_term)
        final_total = final_losses["total"].item()
        
        assert final_total < initial_total


class TestAmericanOptions:
    """Tests for American options with early exercise."""
    
    def test_american_payoff_put(self):
        """Put payoff should be max(K-S, 0)."""
        params = AmericanParams(K=100.0, option_type="put")
        S = torch.tensor([80.0, 100.0, 120.0])
        payoff = american_payoff(S, params)
        
        expected = torch.tensor([20.0, 0.0, 0.0])
        torch.testing.assert_close(payoff, expected)
    
    def test_american_payoff_call(self):
        """Call payoff should be max(S-K, 0)."""
        params = AmericanParams(K=100.0, option_type="call")
        S = torch.tensor([80.0, 100.0, 120.0])
        payoff = american_payoff(S, params)
        
        expected = torch.tensor([0.0, 0.0, 20.0])
        torch.testing.assert_close(payoff, expected)
    
    def test_binomial_put_positive(self):
        """Binomial put price should be positive."""
        params = AmericanParams(K=100.0, option_type="put")
        price = american_put_binomial(100.0, params, n_steps=100)
        assert price > 0
    
    def test_binomial_put_exceeds_european(self):
        """American put should exceed European put."""
        from src.pricing.analytical import AnalyticalPricer
        
        params = AmericanParams(K=100.0, sigma=0.3, option_type="put")
        am_price = american_put_binomial(100.0, params, n_steps=200)
        
        # European put via BS
        euro_pricer = AnalyticalPricer(r=params.r, sigma=params.sigma)
        euro_price = euro_pricer.black_scholes(100.0, params.K, params.T, "put")
        
        assert am_price >= euro_price
    
    def test_binomial_put_deep_itm(self):
        """Deep ITM put should be close to intrinsic."""
        params = AmericanParams(K=100.0, option_type="put")
        price = american_put_binomial(50.0, params, n_steps=200)
        intrinsic = 50.0
        # Should be at least intrinsic
        assert price >= intrinsic * 0.99
    
    def test_binomial_call_equals_european(self):
        """American call = European call for non-dividend stock."""
        from src.pricing.analytical import AnalyticalPricer
        
        params = AmericanParams(K=100.0, sigma=0.2, option_type="call")
        am_price = american_call_binomial(100.0, params, n_steps=200)
        
        euro_pricer = AnalyticalPricer(r=params.r, sigma=params.sigma)
        euro_price = euro_pricer.black_scholes(100.0, params.K, params.T, "call")
        
        # Should be approximately equal
        assert abs(am_price - euro_price) < 0.5
    
    def test_american_pinn_forward_shape(self):
        """PINN forward pass returns correct shape."""
        params = AmericanParams(option_type="put")
        model = AmericanPINN(params, hidden_dims=[32, 32])
        
        S = torch.tensor([80.0, 100.0, 120.0])
        t = torch.tensor([0.5, 0.5, 0.5])
        V = model(S, t)
        
        assert V.shape == (3,)
    
    def test_american_pinn_exceeds_payoff(self):
        """American option value should always exceed payoff."""
        params = AmericanParams(K=100.0, option_type="put")
        model = AmericanPINN(params, hidden_dims=[32, 32])
        
        S = torch.rand(100) * 200
        t = torch.rand(100) * 0.9  # Not at terminal
        
        V = model(S, t)
        payoff = american_payoff(S, params)
        
        # V >= payoff (with small tolerance for soft constraint)
        assert (V >= payoff - 0.1).all()
    
    def test_american_pinn_terminal_equals_payoff(self):
        """At terminal, value should equal payoff."""
        params = AmericanParams(K=100.0, option_type="put")
        model = AmericanPINN(params, hidden_dims=[32, 32])
        
        S = torch.tensor([80.0, 100.0, 120.0])
        payoff = model.terminal_condition(S)
        
        expected = torch.tensor([20.0, 0.0, 0.0])
        torch.testing.assert_close(payoff, expected)
    
    def test_american_trainer_loss_decreases(self):
        """Training should reduce loss."""
        params = AmericanParams(option_type="put")
        model = AmericanPINN(params, hidden_dims=[32, 32])
        trainer = AmericanPINNTrainer(model, lr=1e-2)
        
        S_int = torch.rand(100) * 200
        t_int = torch.rand(100)
        S_term = torch.rand(50) * 200
        
        initial_losses = trainer.compute_loss(S_int, t_int, S_term)
        initial_total = initial_losses["total"].item()
        
        for _ in range(50):
            trainer.train_step(S_int, t_int, S_term)
        
        final_losses = trainer.compute_loss(S_int, t_int, S_term)
        final_total = final_losses["total"].item()
        
        assert final_total < initial_total
    
    def test_early_exercise_boundary_exists(self):
        """Should find non-trivial exercise boundary."""
        params = AmericanParams(K=100.0, option_type="put")
        model = AmericanPINN(params, hidden_dims=[32, 32])
        
        # Quick training  
        trainer = AmericanPINNTrainer(model)
        trainer.train(n_epochs=200, n_interior=500, log_every=1000)
        
        t_values = np.linspace(0.1, 0.9, 5)
        boundary = find_early_exercise_boundary(model, t_values, S_range=(30, 120))
        
        # Boundary should exist and be reasonable (between 30 and 120)
        assert boundary.min() >= 30
        assert boundary.max() <= 120


class TestModelConsistency:
    """Cross-model consistency tests."""
    
    def test_merton_reduces_to_bs(self):
        """Merton with λ=0 should equal Black-Scholes."""
        from src.pricing.analytical import AnalyticalPricer
        
        # Merton with no jumps
        merton_params = MertonParams(sigma=0.2, K=100.0, lam=0.0)
        merton_price = merton_analytical_call(100.0, merton_params, tau=1.0)
        
        # Black-Scholes
        bs_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        bs_price = bs_pricer.black_scholes(100.0, 100.0, 1.0, "call")
        
        assert abs(merton_price - bs_price) < 0.1
    
    def test_heston_constant_vol_approximates_bs(self):
        """Heston with very low vol-of-vol should approximate BS."""
        from src.pricing.analytical import AnalyticalPricer
        
        # Heston with minimal stochastic volatility
        heston_params = HestonParams(
            v0=0.04, theta=0.04, kappa=10.0, xi=0.01, rho=0.0, K=100.0
        )
        heston_price = heston_call_price(100.0, heston_params, tau=1.0)
        
        # Black-Scholes with 20% vol
        bs_pricer = AnalyticalPricer(r=0.05, sigma=0.2)
        bs_price = bs_pricer.black_scholes(100.0, 100.0, 1.0, "call")
        
        # Should be close (within 10%)
        assert abs(heston_price - bs_price) / bs_price < 0.1
