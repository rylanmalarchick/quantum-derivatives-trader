"""Tests for volatility calibration PINN."""

import pytest
import numpy as np
import torch

from src.pde.dupire import (
    DupireParams,
    generate_synthetic_vol_surface,
    black_scholes_call,
    black_scholes_call_torch,
    LocalVolNetwork,
    CallPriceNetwork,
    generate_calibration_data,
)
from src.classical.pinn_calibration import (
    VolCalibrationPINN,
    VolCalibrationTrainer,
    evaluate_calibration,
)


class TestDupireParams:
    """Test Dupire parameter configuration."""
    
    def test_default_params(self):
        """Test default parameter creation."""
        params = DupireParams()
        assert params.r == 0.05
        assert params.S0 == 100.0
        assert params.K_min == 50.0
        assert params.K_max == 200.0
    
    def test_custom_params(self):
        """Test custom parameter creation."""
        params = DupireParams(
            r=0.03,
            S0=120.0,
            vol_base=0.25,
        )
        assert params.r == 0.03
        assert params.S0 == 120.0
        assert params.vol_base == 0.25


class TestSyntheticVolSurface:
    """Test synthetic volatility surface generation."""
    
    def test_surface_shape(self):
        """Test surface dimensions."""
        params = DupireParams()
        surface = generate_synthetic_vol_surface(
            params, n_strikes=10, n_maturities=5
        )
        
        assert len(surface["strikes"]) == 50  # 10 * 5
        assert len(surface["maturities"]) == 50
        assert len(surface["implied_vols"]) == 50
        assert len(surface["call_prices"]) == 50
    
    def test_vol_positivity(self):
        """Test all vols are positive."""
        params = DupireParams()
        surface = generate_synthetic_vol_surface(params)
        
        assert np.all(surface["implied_vols"] > 0)
    
    def test_price_positivity(self):
        """Test all prices are positive."""
        params = DupireParams()
        surface = generate_synthetic_vol_surface(params)
        
        assert np.all(surface["call_prices"] > 0)
    
    def test_vol_skew(self):
        """Test negative skew (lower strikes have higher vol)."""
        params = DupireParams(vol_skew=-0.1)
        surface = generate_synthetic_vol_surface(
            params, n_strikes=20, n_maturities=1
        )
        
        # For fixed maturity, vol should decrease with strike
        # (due to negative skew)
        iv = surface["implied_vols"]
        K = surface["strikes"]
        
        # Low strikes should have higher vol than high strikes on average
        low_K_mask = K < params.S0
        high_K_mask = K > params.S0
        
        if np.any(low_K_mask) and np.any(high_K_mask):
            assert np.mean(iv[low_K_mask]) > np.mean(iv[high_K_mask])
    
    def test_noise_addition(self):
        """Test that noise is added correctly."""
        params = DupireParams()
        
        surface_clean = generate_synthetic_vol_surface(params, noise_std=0.0, seed=42)
        surface_noisy = generate_synthetic_vol_surface(params, noise_std=0.02, seed=42)
        
        # Noisy surface should differ from clean
        assert not np.allclose(surface_clean["implied_vols"], surface_noisy["implied_vols"])


class TestBlackScholes:
    """Test Black-Scholes pricing functions."""
    
    def test_call_price_positivity(self):
        """Test call prices are positive."""
        price = black_scholes_call(
            S=100.0,
            K=np.array([80, 100, 120]),
            T=np.array([1.0, 1.0, 1.0]),
            r=0.05,
            sigma=np.array([0.2, 0.2, 0.2]),
        )
        assert np.all(price > 0)
    
    def test_call_price_monotonicity(self):
        """Test call price decreases with strike."""
        K = np.array([80, 100, 120, 140])
        T = np.ones_like(K)
        sigma = np.ones_like(K) * 0.2
        
        price = black_scholes_call(S=100.0, K=K, T=T, r=0.05, sigma=sigma)
        
        # Price should decrease with strike for fixed maturity
        assert np.all(np.diff(price) < 0)
    
    def test_torch_numpy_consistency(self):
        """Test PyTorch and NumPy implementations match."""
        S = 100.0
        K = np.array([80, 100, 120])
        T = np.array([0.5, 1.0, 1.5])
        sigma = np.array([0.2, 0.25, 0.3])
        r = 0.05
        
        price_np = black_scholes_call(S, K, T, r, sigma)
        price_torch = black_scholes_call_torch(
            torch.tensor(S),
            torch.tensor(K, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
            r,
            torch.tensor(sigma, dtype=torch.float32),
        ).numpy()
        
        np.testing.assert_allclose(price_np, price_torch, rtol=1e-5)


class TestLocalVolNetwork:
    """Test LocalVolNetwork architecture."""
    
    def test_output_shape(self):
        """Test output has correct shape."""
        net = LocalVolNetwork()
        K = torch.tensor([80., 100., 120.])
        T = torch.tensor([0.5, 1.0, 1.5])
        
        sigma = net(K, T)
        assert sigma.shape == (3,)
    
    def test_output_positivity(self):
        """Test volatility is always positive (Softplus)."""
        net = LocalVolNetwork()
        K = torch.rand(100) * 150 + 50
        T = torch.rand(100) * 2
        
        sigma = net(K, T)
        assert torch.all(sigma > 0)
    
    def test_output_range(self):
        """Test volatility is in reasonable range."""
        net = LocalVolNetwork()
        K = torch.rand(100) * 150 + 50
        T = torch.rand(100) * 2
        
        sigma = net(K, T)
        assert torch.all(sigma >= 0.05)
        assert torch.all(sigma <= 1.5)


class TestVolCalibrationPINN:
    """Test VolCalibrationPINN model."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create model and test data."""
        params = DupireParams()
        model = VolCalibrationPINN(params, hidden_dims=[32, 32])
        data = generate_calibration_data(params, n_points=25)
        return model, data, params
    
    def test_forward_pass(self, model_and_data):
        """Test forward pass returns prices and vols."""
        model, data, _ = model_and_data
        
        C, sigma = model(data["K"], data["T"])
        
        assert C.shape == data["K"].shape
        assert sigma.shape == data["K"].shape
        assert torch.all(C >= -1e-5)  # Deep OTM options can be ~0, allow numerical tolerance
        assert torch.all(sigma > 0)
    
    def test_vol_surface(self, model_and_data):
        """Test getting full vol surface."""
        model, _, params = model_and_data
        
        K = torch.linspace(params.K_min, params.K_max, 10)
        T = torch.linspace(params.T_min, params.T_max, 5)
        K_grid, T_grid = torch.meshgrid(K, T, indexing='ij')
        
        sigma_surface = model.get_vol_surface(K_grid, T_grid)
        
        assert sigma_surface.shape == K_grid.shape


class TestVolCalibrationTrainer:
    """Test VolCalibrationTrainer."""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer with synthetic data."""
        params = DupireParams()
        model = VolCalibrationPINN(params, hidden_dims=[32, 32])
        data = generate_calibration_data(params, n_points=25)
        
        return VolCalibrationTrainer(
            model=model,
            market_data=data,
            lr=1e-3,
            lambda_smooth=0.01,
            lambda_arb=0.1,
        )
    
    def test_train_step(self, trainer):
        """Test single training step."""
        metrics = trainer.train_step()
        
        assert "total" in metrics
        assert "data" in metrics
        assert "smooth" in metrics
        assert "arb" in metrics
        assert metrics["total"] > 0
    
    def test_short_training(self, trainer):
        """Test short training run converges."""
        history = trainer.train(n_epochs=50, log_every=50)
        
        assert len(history["total"]) == 50
        # Loss should decrease
        assert history["total"][-1] < history["total"][0]
    
    def test_data_loss(self, trainer):
        """Test data loss computation."""
        loss = trainer.compute_data_loss(
            trainer.K_market,
            trainer.T_market,
            trainer.C_market,
        )
        assert loss.item() >= 0
    
    def test_smoothness_loss(self, trainer):
        """Test smoothness loss is finite."""
        loss = trainer.compute_smoothness_loss(
            trainer.K_market[:10],
            trainer.T_market[:10],
        )
        assert torch.isfinite(loss)


class TestEvaluation:
    """Test calibration evaluation."""
    
    def test_evaluation_metrics(self):
        """Test evaluation returns expected metrics."""
        params = DupireParams()
        model = VolCalibrationPINN(params, hidden_dims=[32, 32])
        data = generate_calibration_data(params, n_points=25)
        
        metrics = evaluate_calibration(model, data)
        
        assert "price_mse" in metrics
        assert "price_mae" in metrics
        assert "vol_mse" in metrics
        assert "vol_mae" in metrics
        assert metrics["price_mse"] >= 0
        assert metrics["vol_mse"] >= 0


class TestEndToEnd:
    """End-to-end calibration tests."""
    
    def test_perfect_data_calibration(self):
        """Test calibration on perfect synthetic data."""
        params = DupireParams(
            vol_base=0.20,
            vol_skew=-0.05,
            vol_smile=0.02,
            vol_term=0.0,
        )
        
        model = VolCalibrationPINN(params, hidden_dims=[64, 64, 64])
        data = generate_calibration_data(params, n_points=100)
        
        trainer = VolCalibrationTrainer(
            model=model,
            market_data=data,
            lr=5e-3,
            lambda_smooth=0.001,
            lambda_arb=0.01,
        )
        
        # Train for enough epochs
        history = trainer.train(n_epochs=500, log_every=500)
        
        # Should achieve reasonable fit
        metrics = evaluate_calibration(model, data)
        
        # Price fit should be good
        assert metrics["mean_rel_error_pct"] < 10.0, f"Price error too high: {metrics['mean_rel_error_pct']}"
    
    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        params = DupireParams()
        model = VolCalibrationPINN(params, hidden_dims=[32, 32])
        data = generate_calibration_data(params, n_points=16)
        
        K = data["K"].requires_grad_(True)
        T = data["T"].requires_grad_(True)
        
        C, sigma = model(K, T)
        loss = C.sum()
        loss.backward()
        
        # Gradients should exist
        assert K.grad is not None
        assert T.grad is not None
        
        # Gradients should be non-zero
        assert torch.any(K.grad != 0)
