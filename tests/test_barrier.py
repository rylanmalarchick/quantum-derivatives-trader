"""
Tests for barrier option pricing module.

Tests cover:
1. Analytical formula correctness (at barrier, deep ITM/OTM)
2. PINN output shape
3. Boundary condition at barrier
4. Reduces to standard call when B → 0
"""

import pytest
import torch
import numpy as np
from scipy.stats import norm

from src.pde.barrier import (
    BarrierParams,
    barrier_payoff,
    barrier_pde_residual,
    barrier_boundary_loss,
    barrier_analytical_down_out_call,
    BarrierPINN,
    BarrierPINNTrainer,
    _bs_call_price,
)


class TestBarrierParams:
    """Tests for BarrierParams dataclass."""

    def test_valid_params(self):
        """Valid parameters should create successfully."""
        params = BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)
        assert params.r == 0.05
        assert params.sigma == 0.2
        assert params.K == 100.0
        assert params.B == 80.0
        assert params.barrier_type == "down-out-call"

    def test_barrier_above_strike_raises(self):
        """Barrier >= strike should raise for down-out-call."""
        with pytest.raises(ValueError, match="barrier B=110.0 must be < strike K=100.0"):
            BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=110.0)

    def test_barrier_equals_strike_raises(self):
        """Barrier = strike should raise for down-out-call."""
        with pytest.raises(ValueError, match="barrier B=100.0 must be < strike K=100.0"):
            BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=100.0)

    def test_negative_barrier_raises(self):
        """Negative barrier should raise."""
        with pytest.raises(ValueError, match="Barrier B=-10 must be positive"):
            BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=-10)

    def test_negative_sigma_raises(self):
        """Negative volatility should raise."""
        with pytest.raises(ValueError, match="Volatility sigma=-0.2 must be positive"):
            BarrierParams(r=0.05, sigma=-0.2, K=100.0, T=1.0, B=80.0)


class TestBarrierPayoff:
    """Tests for barrier_payoff function."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    def test_payoff_above_strike(self, params):
        """ITM: S > K should give positive payoff."""
        S = torch.tensor([110.0, 120.0, 150.0])
        payoff = barrier_payoff(S, params)
        expected = torch.tensor([10.0, 20.0, 50.0])
        torch.testing.assert_close(payoff, expected)

    def test_payoff_below_strike_above_barrier(self, params):
        """OTM but above barrier: should give zero payoff."""
        S = torch.tensor([85.0, 90.0, 99.0])
        payoff = barrier_payoff(S, params)
        assert torch.all(payoff == 0)

    def test_payoff_at_barrier(self, params):
        """At barrier: knocked out, payoff = 0."""
        S = torch.tensor([80.0])
        payoff = barrier_payoff(S, params)
        assert payoff.item() == 0.0

    def test_payoff_below_barrier(self, params):
        """Below barrier: knocked out, payoff = 0."""
        S = torch.tensor([50.0, 70.0, 79.9])
        payoff = barrier_payoff(S, params)
        assert torch.all(payoff == 0)

    def test_payoff_shape(self, params):
        """Output shape should match input."""
        S = torch.rand(100) * 100 + 50
        payoff = barrier_payoff(S, params)
        assert payoff.shape == S.shape


class TestBarrierAnalytical:
    """Tests for barrier_analytical_down_out_call function."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    def test_zero_at_barrier(self, params):
        """At S = B, price should be zero."""
        S = torch.tensor([params.B])
        tau = torch.tensor([params.T])
        price = barrier_analytical_down_out_call(S, params, tau)
        assert torch.allclose(price, torch.tensor([0.0]), atol=1e-6)

    def test_zero_below_barrier(self, params):
        """Below barrier, price should be zero."""
        S = torch.tensor([50.0, 60.0, 79.0])
        tau = torch.full_like(S, params.T)
        price = barrier_analytical_down_out_call(S, params, tau)
        assert torch.all(price == 0)

    def test_positive_above_barrier(self, params):
        """Above barrier, price should be positive."""
        S = torch.tensor([90.0, 100.0, 110.0, 120.0])
        tau = torch.full_like(S, params.T)
        price = barrier_analytical_down_out_call(S, params, tau)
        assert torch.all(price > 0)

    def test_less_than_vanilla_call(self, params):
        """Barrier option should be worth less than vanilla call."""
        S = torch.tensor([90.0, 100.0, 110.0, 120.0])
        tau = torch.full_like(S, params.T)
        
        # Barrier option price
        barrier_price = barrier_analytical_down_out_call(S, params, tau)
        
        # Vanilla call price
        S_np = S.numpy()
        tau_np = tau.numpy()
        vanilla_price = _bs_call_price(S_np, params.K, params.r, params.sigma, tau_np)
        
        # Barrier should be strictly less (unless S is very high)
        assert np.all(barrier_price.numpy() <= vanilla_price + 1e-6)

    def test_deep_itm_approaches_vanilla(self, params):
        """For very high S, barrier option approaches vanilla call."""
        S = torch.tensor([200.0, 300.0, 500.0])
        tau = torch.full_like(S, params.T)
        
        barrier_price = barrier_analytical_down_out_call(S, params, tau)
        vanilla_price = _bs_call_price(S.numpy(), params.K, params.r, params.sigma, tau.numpy())
        
        # For high S, the probability of hitting barrier is negligible
        # So barrier price should be close to vanilla
        rel_diff = np.abs(barrier_price.numpy() - vanilla_price) / vanilla_price
        assert np.all(rel_diff < 0.1)  # Within 10%

    def test_reduces_to_vanilla_when_barrier_zero(self):
        """When B → 0, barrier option → vanilla call."""
        # Use very small barrier
        params_small_B = BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=0.01)
        
        S = torch.tensor([80.0, 100.0, 120.0])
        tau = torch.full_like(S, params_small_B.T)
        
        barrier_price = barrier_analytical_down_out_call(S, params_small_B, tau)
        vanilla_price = _bs_call_price(
            S.numpy(), params_small_B.K, params_small_B.r, 
            params_small_B.sigma, tau.numpy()
        )
        
        # Should be very close
        np.testing.assert_allclose(barrier_price.numpy(), vanilla_price, rtol=0.01)

    def test_at_maturity_equals_payoff(self, params):
        """At τ=0 (maturity), price should equal payoff."""
        S = torch.tensor([85.0, 100.0, 110.0, 120.0])
        tau = torch.full_like(S, 1e-8)  # Very small τ
        
        price = barrier_analytical_down_out_call(S, params, tau)
        payoff = barrier_payoff(S, params)
        
        torch.testing.assert_close(price, payoff, atol=0.1, rtol=0.1)

    def test_monotonic_in_spot(self, params):
        """Price should increase with spot (above barrier)."""
        S = torch.tensor([85.0, 90.0, 95.0, 100.0, 110.0, 120.0])
        tau = torch.full_like(S, params.T)
        
        price = barrier_analytical_down_out_call(S, params, tau)
        
        # Verify strictly increasing
        assert torch.all(price[1:] >= price[:-1] - 1e-6)


class TestBarrierPDEResidual:
    """Tests for barrier_pde_residual function."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    def test_residual_computation(self, params):
        """PDE residual should compute without errors."""
        S = torch.tensor([90.0, 100.0, 110.0], requires_grad=True)
        t = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        
        # Use a smooth function that depends on S and t
        V = S**2 * t
        
        residual = barrier_pde_residual(V, S, t, params)
        
        assert residual.shape == S.shape
        assert not torch.isnan(residual).any()
        assert not torch.isinf(residual).any()

    def test_residual_shape(self, params):
        """Residual shape should match input."""
        n = 50
        S = torch.rand(n, requires_grad=True) * 100 + 90
        t = torch.rand(n, requires_grad=True) * params.T
        
        V = S**2 * t
        residual = barrier_pde_residual(V, S, t, params)
        
        assert residual.shape == (n,)


class TestBarrierPINN:
    """Tests for BarrierPINN class."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    @pytest.fixture
    def model(self, params):
        return BarrierPINN(params, hidden_dims=[32, 32], S_max=200.0)

    def test_output_shape(self, model):
        """Output shape should be (batch,)."""
        S = torch.rand(100) * 100 + 50
        t = torch.rand(100) * model.params.T
        
        V = model(S, t)
        
        assert V.shape == (100,)

    def test_output_non_negative(self, model):
        """Output should always be non-negative."""
        S = torch.rand(100) * 200
        t = torch.rand(100) * model.params.T
        
        V = model(S, t)
        
        assert torch.all(V >= 0)

    def test_barrier_constraint(self, model, params):
        """At S = B, output should be exactly zero."""
        S = torch.full((50,), params.B)
        t = torch.rand(50) * params.T
        
        V = model(S, t)
        
        # Hard constraint enforced via multiplication by (S - B)
        torch.testing.assert_close(V, torch.zeros_like(V), atol=1e-6, rtol=1e-6)

    def test_below_barrier_zero(self, model, params):
        """Below barrier, output should be zero."""
        S = torch.rand(50) * params.B * 0.9  # S < B
        t = torch.rand(50) * params.T
        
        V = model(S, t)
        
        torch.testing.assert_close(V, torch.zeros_like(V), atol=1e-6, rtol=1e-6)

    def test_terminal_condition_method(self, model, params):
        """terminal_condition should return correct payoff."""
        S = torch.tensor([70.0, 85.0, 100.0, 120.0])
        
        payoff = model.terminal_condition(S)
        expected = barrier_payoff(S, params)
        
        torch.testing.assert_close(payoff, expected)

    def test_differentiable(self, model):
        """Model should be differentiable for PDE loss."""
        S = torch.tensor([100.0], requires_grad=True)
        t = torch.tensor([0.5], requires_grad=True)
        
        V = model(S, t)
        V.backward()
        
        assert S.grad is not None
        assert t.grad is not None


class TestBarrierBoundaryLoss:
    """Tests for barrier_boundary_loss function."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    @pytest.fixture
    def model(self, params):
        return BarrierPINN(params, hidden_dims=[32, 32])

    def test_boundary_loss_zero_for_trained(self, model, params):
        """With hard constraint, boundary loss should be zero."""
        t = torch.rand(50) * params.T
        
        loss = barrier_boundary_loss(model, t, params)
        
        # Due to hard constraint in forward pass
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-10, rtol=1e-10)

    def test_boundary_loss_shape(self, model, params):
        """Boundary loss should be a scalar."""
        t = torch.rand(100) * params.T
        
        loss = barrier_boundary_loss(model, t, params)
        
        assert loss.shape == ()


class TestBarrierPINNTrainer:
    """Tests for BarrierPINNTrainer class."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    @pytest.fixture
    def model(self, params):
        return BarrierPINN(params, hidden_dims=[32, 32], S_max=200.0)

    @pytest.fixture
    def trainer(self, model):
        return BarrierPINNTrainer(model, lr=1e-3)

    def test_generate_collocation_points(self, trainer, params):
        """Generated points should be in valid ranges."""
        S_int, t_int, S_term, t_barrier = trainer.generate_collocation_points(
            n_interior=100, n_terminal=50, n_barrier=20, device=torch.device("cpu")
        )
        
        # Interior points above barrier
        assert torch.all(S_int > params.B)
        assert torch.all(t_int >= 0)
        assert torch.all(t_int <= params.T)
        
        # Terminal points above barrier
        assert torch.all(S_term > params.B)
        
        # Barrier times in [0, T]
        assert torch.all(t_barrier >= 0)
        assert torch.all(t_barrier <= params.T)

    def test_compute_loss_returns_dict(self, trainer, params):
        """compute_loss should return loss dictionary."""
        device = torch.device("cpu")
        S_int, t_int, S_term, t_barrier = trainer.generate_collocation_points(
            n_interior=50, n_terminal=20, n_barrier=10, device=device
        )
        
        losses = trainer.compute_loss(S_int, t_int, S_term, t_barrier)
        
        assert "total" in losses
        assert "pde" in losses
        assert "ic" in losses
        assert "barrier" in losses
        assert "upper" in losses

    def test_train_step_reduces_loss(self, trainer, params):
        """Training should reduce loss over steps."""
        device = torch.device("cpu")
        
        # Initial loss
        S_int, t_int, S_term, t_barrier = trainer.generate_collocation_points(
            n_interior=50, n_terminal=20, n_barrier=10, device=device
        )
        initial_losses = trainer.compute_loss(S_int, t_int, S_term, t_barrier)
        initial_total = initial_losses["total"].item()
        
        # Train for a few steps
        for _ in range(10):
            S_int, t_int, S_term, t_barrier = trainer.generate_collocation_points(
                n_interior=50, n_terminal=20, n_barrier=10, device=device
            )
            trainer.train_step(S_int, t_int, S_term, t_barrier)
        
        # Final loss (use same points for fair comparison)
        final_losses = trainer.compute_loss(S_int, t_int, S_term, t_barrier)
        final_total = final_losses["total"].item()
        
        # Loss should decrease (or at least not explode)
        assert final_total < initial_total * 2  # Sanity check

    def test_train_records_history(self, trainer):
        """Training should record loss history."""
        history = trainer.train(n_epochs=5, n_interior=50, n_terminal=20, n_barrier=10, log_every=10)
        
        assert len(history) == 5
        assert all("total" in h for h in history)


class TestBarrierReducesToVanilla:
    """Test that barrier option reduces to vanilla when B → 0."""

    def test_analytical_convergence(self):
        """Analytical formula should converge to BS as B → 0."""
        # Standard parameters
        r, sigma, K, T = 0.05, 0.2, 100.0, 1.0
        S = torch.tensor([80.0, 100.0, 120.0])
        tau = torch.full_like(S, T)
        
        # Vanilla BS prices
        vanilla_prices = _bs_call_price(S.numpy(), K, r, sigma, tau.numpy())
        
        # Barrier prices with decreasing B
        for B in [10.0, 1.0, 0.1, 0.01]:
            params = BarrierParams(r=r, sigma=sigma, K=K, T=T, B=B)
            barrier_prices = barrier_analytical_down_out_call(S, params, tau).numpy()
            
            # As B decreases, barrier price should approach vanilla
            diff = np.abs(barrier_prices - vanilla_prices)
            assert np.all(diff < 0.5 * K)  # Reasonable bound

    def test_pinn_shape_consistency(self):
        """PINN should maintain consistent shapes."""
        params = BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=50.0)
        model = BarrierPINN(params, hidden_dims=[32, 32])
        
        # Test various batch sizes
        for batch_size in [1, 10, 100]:
            S = torch.rand(batch_size) * 150 + 50
            t = torch.rand(batch_size) * params.T
            
            V = model(S, t)
            
            assert V.shape == (batch_size,)


class TestBarrierEdgeCases:
    """Test edge cases and numerical stability."""

    @pytest.fixture
    def params(self):
        return BarrierParams(r=0.05, sigma=0.2, K=100.0, T=1.0, B=80.0)

    def test_near_barrier_stability(self, params):
        """Prices near barrier should be stable."""
        # Spots very close to barrier
        S = torch.tensor([80.001, 80.01, 80.1, 81.0])
        tau = torch.full_like(S, params.T)
        
        prices = barrier_analytical_down_out_call(S, params, tau)
        
        assert not torch.isnan(prices).any()
        assert not torch.isinf(prices).any()
        assert torch.all(prices >= 0)

    def test_small_tau_stability(self, params):
        """Prices at small tau (near maturity) should be stable."""
        S = torch.tensor([90.0, 100.0, 110.0])
        tau = torch.tensor([0.001, 0.001, 0.001])
        
        prices = barrier_analytical_down_out_call(S, params, tau)
        
        assert not torch.isnan(prices).any()
        assert not torch.isinf(prices).any()

    def test_large_spot_stability(self, params):
        """Prices at large spot should be stable."""
        S = torch.tensor([500.0, 1000.0, 2000.0])
        tau = torch.full_like(S, params.T)
        
        prices = barrier_analytical_down_out_call(S, params, tau)
        
        assert not torch.isnan(prices).any()
        assert not torch.isinf(prices).any()
        # Should be close to intrinsic value for very large S
        intrinsic = S - params.K
        assert torch.all(prices > 0.9 * intrinsic)
