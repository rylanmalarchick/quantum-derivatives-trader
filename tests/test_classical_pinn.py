"""
Tests for classical PINN implementation.
"""

import pytest
import torch
import torch.nn as nn

from src.classical.pinn import PINN, PINNTrainer
from src.classical.networks import MLP, ResidualMLP
from src.classical.losses import PINNLoss
from src.pde.black_scholes import BSParams


class TestPINNForwardPass:
    """Tests for PINN forward pass."""

    def test_pinn_output_shape(self, bs_params, device):
        """PINN should output tensor of correct shape."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        model.to(device)

        batch_size = 10
        S = torch.rand(batch_size, device=device) * 200
        t = torch.rand(batch_size, device=device)

        V = model(S, t)

        assert V.shape == (batch_size,)
        assert not torch.isnan(V).any()

    def test_pinn_with_residual_mlp(self, device):
        """PINN with residual connections should work."""
        model = PINN(
            hidden_dims=[32, 32, 32],
            use_residual=True,
            S_max=200.0,
            T_max=1.0
        )
        model.to(device)

        S = torch.rand(5, device=device) * 200
        t = torch.rand(5, device=device)

        V = model(S, t)

        assert V.shape == (5,)

    def test_pinn_handles_1d_and_2d_inputs(self, device):
        """PINN should handle both 1D and 2D tensor inputs."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)

        # 1D input
        S_1d = torch.tensor([100.0, 110.0], device=device)
        t_1d = torch.tensor([0.5, 0.5], device=device)
        V_1d = model(S_1d, t_1d)

        # 2D input (batch, 1)
        S_2d = S_1d.unsqueeze(-1)
        t_2d = t_1d.unsqueeze(-1)
        V_2d = model(S_2d, t_2d)

        assert V_1d.shape == (2,)
        assert V_2d.shape == (2,)

    def test_pinn_predict_with_greeks(self, device):
        """Test predict_with_greeks returns all expected keys."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        model.to(device)

        S = torch.rand(5, device=device) * 200
        t = torch.rand(5, device=device)

        result = model.predict_with_greeks(S, t)

        assert "V" in result
        assert "delta" in result
        assert "gamma" in result
        assert "theta" in result

        # All should have correct shape
        for key in ["V", "delta", "gamma", "theta"]:
            assert result[key].shape == (5,)


class TestNetworkArchitectures:
    """Tests for MLP and ResidualMLP networks."""

    @pytest.mark.parametrize("hidden_dims", [[32], [64, 64], [32, 32, 32, 32]])
    def test_mlp_output_shape(self, hidden_dims):
        """MLP should produce correct output shape."""
        model = MLP(in_dim=2, out_dim=1, hidden_dims=hidden_dims)

        x = torch.rand(10, 2)
        y = model(x)

        assert y.shape == (10, 1)

    @pytest.mark.parametrize("activation", ["tanh", "swish", "gelu"])
    def test_mlp_activations(self, activation):
        """MLP should work with different activations."""
        model = MLP(in_dim=2, out_dim=1, hidden_dims=[32, 32], activation=activation)

        x = torch.rand(5, 2)
        y = model(x)

        assert y.shape == (5, 1)
        assert not torch.isnan(y).any()

    @pytest.mark.parametrize("n_blocks", [1, 2, 4])
    def test_residual_mlp_output_shape(self, n_blocks):
        """ResidualMLP should produce correct output shape."""
        model = ResidualMLP(in_dim=2, out_dim=1, hidden_dim=32, n_blocks=n_blocks)

        x = torch.rand(10, 2)
        y = model(x)

        assert y.shape == (10, 1)

    def test_mlp_gradient_flow(self):
        """Verify gradients flow through MLP."""
        model = MLP(in_dim=2, out_dim=1, hidden_dims=[32, 32])

        x = torch.rand(5, 2, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist and are not zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestPINNLoss:
    """Tests for PINNLoss function."""

    def test_loss_returns_expected_keys(self, bs_params):
        """PINNLoss should return dict with pde, bc, ic, and total keys."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        loss_fn = PINNLoss(bs_params)

        # Create sample points
        S_interior = torch.rand(50) * 200
        t_interior = torch.rand(50)
        S_boundary = torch.cat([torch.zeros(25), torch.full((25,), 200.0)])
        t_boundary = torch.rand(50)
        S_terminal = torch.rand(50) * 200

        losses = loss_fn(
            model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal
        )

        assert "pde" in losses
        assert "bc" in losses
        assert "ic" in losses
        assert "total" in losses

        # Total should be sum of components (weighted)
        assert losses["total"].item() >= 0

    def test_loss_values_are_positive(self, bs_params):
        """All loss components should be non-negative."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        loss_fn = PINNLoss(bs_params)

        S_interior = torch.rand(50) * 200
        t_interior = torch.rand(50)
        S_boundary = torch.cat([torch.zeros(25), torch.full((25,), 200.0)])
        t_boundary = torch.rand(50)
        S_terminal = torch.rand(50) * 200

        losses = loss_fn(
            model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal
        )

        for key in ["pde", "bc", "ic", "total"]:
            assert losses[key].item() >= 0

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_loss_for_different_option_types(self, bs_params, option_type):
        """Loss should work for both call and put options."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        loss_fn = PINNLoss(bs_params, option_type=option_type)

        S_interior = torch.rand(20) * 200
        t_interior = torch.rand(20)
        S_boundary = torch.cat([torch.zeros(10), torch.full((10,), 200.0)])
        t_boundary = torch.rand(20)
        S_terminal = torch.rand(20) * 200

        losses = loss_fn(
            model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal
        )

        assert losses["total"].item() >= 0


class TestGradientComputation:
    """Tests for gradient computation in PINN."""

    def test_pde_residual_requires_grad(self, bs_params):
        """PDE residual computation should maintain gradient graph."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        loss_fn = PINNLoss(bs_params)

        S = torch.rand(10) * 200
        t = torch.rand(10)

        # Compute residual
        residual = loss_fn.pde_residual(model, S, t)

        # Should be differentiable
        assert residual.requires_grad

    def test_backward_pass_computes_gradients(self, bs_params):
        """Backward pass should compute gradients for all parameters."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        loss_fn = PINNLoss(bs_params)

        S_interior = torch.rand(20) * 200
        t_interior = torch.rand(20)
        S_boundary = torch.cat([torch.zeros(10), torch.full((10,), 200.0)])
        t_boundary = torch.rand(20)
        S_terminal = torch.rand(20) * 200

        losses = loss_fn(
            model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal
        )

        losses["total"].backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestPINNTrainer:
    """Tests for PINNTrainer class."""

    def test_trainer_initialization(self, bs_params):
        """Trainer should initialize correctly."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        trainer = PINNTrainer(model, bs_params, lr=1e-3)

        assert trainer.model is model
        assert trainer.params is bs_params
        assert len(trainer.history["total"]) == 0

    def test_train_step_decreases_loss(self, bs_params):
        """A few training steps should generally decrease loss."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        trainer = PINNTrainer(model, bs_params, lr=1e-2)

        # Generate collocation points
        S_int = torch.rand(100) * 200
        t_int = torch.rand(100)
        S_bc = torch.cat([torch.zeros(50), torch.full((50,), 200.0)])
        t_bc = torch.rand(100)
        S_term = torch.rand(100) * 200

        # Initial loss
        initial_losses = trainer.compute_loss(S_int, t_int, S_bc, t_bc, S_term)
        initial_total = initial_losses["total"].item()

        # Run a few training steps
        for _ in range(10):
            trainer.train_step(S_int, t_int, S_bc, t_bc, S_term)

        # Final loss (from history)
        final_total = trainer.history["total"][-1]

        # Loss should decrease (or at least not explode)
        assert final_total < initial_total * 2  # Reasonable bound

    def test_trainer_records_history(self, bs_params):
        """Trainer should record loss history."""
        model = PINN(hidden_dims=[32, 32], S_max=200.0, T_max=1.0)
        trainer = PINNTrainer(model, bs_params, lr=1e-3)

        S_int = torch.rand(50) * 200
        t_int = torch.rand(50)
        S_bc = torch.cat([torch.zeros(25), torch.full((25,), 200.0)])
        t_bc = torch.rand(50)
        S_term = torch.rand(50) * 200

        n_steps = 5
        for _ in range(n_steps):
            trainer.train_step(S_int, t_int, S_bc, t_bc, S_term)

        assert len(trainer.history["total"]) == n_steps
        assert len(trainer.history["pde"]) == n_steps
        assert len(trainer.history["bc"]) == n_steps
        assert len(trainer.history["ic"]) == n_steps

    def test_trainer_runs_without_error(self, bs_params):
        """Training loop should complete without errors."""
        model = PINN(hidden_dims=[16, 16], S_max=200.0, T_max=1.0)
        trainer = PINNTrainer(model, bs_params, lr=1e-3)

        # Run a few epochs with minimal points
        history = trainer.train(
            n_epochs=3,
            n_interior=50,
            n_boundary=20,
            n_terminal=20,
            print_every=100,  # Don't print during test
        )

        assert "total" in history
        assert len(history["total"]) == 3
