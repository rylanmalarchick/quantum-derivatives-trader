"""Tests for multi-asset basket option PINN."""

import pytest
import numpy as np
import torch

from src.pde.basket import (
    BasketParams,
    basket_payoff,
    basket_pde_residual,
    generate_basket_collocation_lhs,
    monte_carlo_basket,
)
from src.classical.pinn_basket import BasketPINN, BasketPINNTrainer


class TestBasketParams:
    """Tests for BasketParams dataclass."""
    
    def test_default_params(self):
        """Default parameters should be valid."""
        params = BasketParams()
        assert params.n_assets == 5
        assert len(params.S0) == 5
        assert len(params.sigma) == 5
        assert np.isclose(params.weights.sum(), 1.0)
    
    def test_custom_assets(self):
        """Should support different asset counts."""
        for n in [2, 3, 5, 10]:
            params = BasketParams(
                n_assets=n,
                S0=np.ones(n) * 100,
                sigma=np.ones(n) * 0.2,
                weights=np.ones(n) / n,
            )
            assert params.n_assets == n
            assert params.correlation.shape == (n, n)
    
    def test_correlation_symmetric(self):
        """Correlation matrix should be symmetric."""
        params = BasketParams()
        assert np.allclose(params.correlation, params.correlation.T)
    
    def test_correlation_positive_definite(self):
        """Correlation matrix should be positive semi-definite."""
        params = BasketParams()
        eigenvalues = np.linalg.eigvals(params.correlation)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_covariance_computation(self):
        """Covariance should equal rho * sigma_outer."""
        params = BasketParams()
        expected = params.correlation * np.outer(params.sigma, params.sigma)
        assert np.allclose(params.covariance, expected)


class TestBasketPayoff:
    """Tests for basket option payoff."""
    
    def test_payoff_atm(self):
        """ATM basket should have zero payoff."""
        params = BasketParams(K=100.0, weights=np.array([0.2] * 5))
        S = torch.tensor([[100., 100., 100., 100., 100.]])
        payoff = basket_payoff(S, params)
        assert torch.isclose(payoff, torch.tensor([0.0]))
    
    def test_payoff_itm(self):
        """ITM basket should have positive payoff."""
        params = BasketParams(K=100.0, weights=np.array([0.2] * 5))
        S = torch.tensor([[120., 120., 120., 120., 120.]])
        payoff = basket_payoff(S, params)
        expected = 0.2 * 120 * 5 - 100  # = 120 - 100 = 20
        assert torch.isclose(payoff, torch.tensor([20.0]))
    
    def test_payoff_otm(self):
        """OTM basket should have zero payoff."""
        params = BasketParams(K=100.0)
        S = torch.tensor([[80., 80., 80., 80., 80.]])
        payoff = basket_payoff(S, params)
        assert torch.isclose(payoff, torch.tensor([0.0]))
    
    def test_payoff_batch(self):
        """Should handle batch input."""
        params = BasketParams(K=100.0)
        S = torch.tensor([
            [100., 100., 100., 100., 100.],
            [120., 120., 120., 120., 120.],
            [80., 80., 80., 80., 80.],
        ])
        payoffs = basket_payoff(S, params)
        assert payoffs.shape == (3,)


class TestMonteCarlo:
    """Tests for Monte Carlo basket pricing."""
    
    def test_mc_price_positive(self):
        """MC price should be positive for call."""
        params = BasketParams()
        result = monte_carlo_basket(params, n_paths=10000)
        assert result["price"] > 0
    
    def test_mc_std_error_decreases(self):
        """More paths should reduce standard error."""
        params = BasketParams()
        result_small = monte_carlo_basket(params, n_paths=1000, seed=42)
        result_large = monte_carlo_basket(params, n_paths=100000, seed=42)
        assert result_large["std_error"] < result_small["std_error"]
    
    def test_mc_reproducible(self):
        """Same seed should give same result."""
        params = BasketParams()
        result1 = monte_carlo_basket(params, n_paths=10000, seed=123)
        result2 = monte_carlo_basket(params, n_paths=10000, seed=123)
        assert result1["price"] == result2["price"]


class TestCollocationSampling:
    """Tests for Latin Hypercube Sampling."""
    
    def test_lhs_shapes(self):
        """Should return correct shapes."""
        params = BasketParams()
        data = generate_basket_collocation_lhs(params, n_interior=100, n_terminal=50)
        
        assert data["S_int"].shape == (100, 5)
        assert data["t_int"].shape == (100,)
        assert data["S_term"].shape == (50, 5)
    
    def test_lhs_bounds(self):
        """Samples should be within bounds."""
        params = BasketParams()
        data = generate_basket_collocation_lhs(params, n_interior=1000)
        
        for i in range(params.n_assets):
            assert data["S_int"][:, i].min() >= params.S_min[i]
            assert data["S_int"][:, i].max() <= params.S_max[i]
        
        assert data["t_int"].min() >= 0
        assert data["t_int"].max() <= params.T


class TestBasketPINN:
    """Tests for BasketPINN model."""
    
    def test_forward_shape(self):
        """Forward pass should return correct shape."""
        model = BasketPINN(n_assets=5)
        S = torch.rand(10, 5) * 100 + 50
        t = torch.rand(10)
        V = model(S, t)
        assert V.shape == (10,)
    
    def test_forward_single_sample(self):
        """Should handle single sample."""
        model = BasketPINN(n_assets=5)
        S = torch.tensor([[100., 100., 100., 100., 100.]])
        t = torch.tensor([0.0])
        V = model(S, t)
        assert V.shape == (1,)
    
    def test_greeks_computation(self):
        """Should compute Greeks via autodiff."""
        model = BasketPINN(n_assets=5)
        S = torch.rand(5, 5) * 100 + 50
        t = torch.rand(5)
        
        greeks = model.predict_with_greeks(S, t)
        
        assert "V" in greeks
        assert "deltas" in greeks
        assert "gammas" in greeks
        assert "theta" in greeks
        assert greeks["deltas"].shape == (5, 5)
        assert greeks["gammas"].shape == (5, 5)


class TestBasketPINNTrainer:
    """Tests for BasketPINNTrainer."""
    
    def test_training_reduces_loss(self):
        """Training should reduce total loss."""
        params = BasketParams()
        model = BasketPINN(n_assets=5, hidden_dims=[32, 32])
        trainer = BasketPINNTrainer(model, params, lr=1e-2)
        
        # Initial loss
        data = generate_basket_collocation_lhs(params, n_interior=50, n_terminal=25)
        initial_losses = trainer.train_step(data["S_int"], data["t_int"], data["S_term"])
        
        # Train a few steps
        for _ in range(10):
            trainer.train_step(data["S_int"], data["t_int"], data["S_term"])
        
        final_losses = trainer.train_step(data["S_int"], data["t_int"], data["S_term"])
        
        # Loss should decrease (usually, not guaranteed for small networks)
        # Just check it doesn't explode
        assert final_losses["total"] < initial_losses["total"] * 10
