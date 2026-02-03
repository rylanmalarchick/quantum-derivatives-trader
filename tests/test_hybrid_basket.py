"""Tests for hybrid quantum basket PINN."""

import pytest
import numpy as np
import torch

from src.quantum.hybrid_basket import (
    HybridBasketPINN,
    EnsembleHybridBasketPINN,
    QuantumEnhancedBasketPINN,
    create_hybrid_basket_pinn,
)


class TestHybridBasketPINN:
    """Test compressed hybrid basket PINN."""
    
    @pytest.fixture
    def model(self):
        return HybridBasketPINN(
            n_assets=5,
            n_qubits=4,
            n_layers=2,
            encoder_dims=[32, 16],
            decoder_dims=[16, 32],
        )
    
    def test_forward_shape(self, model):
        """Test output shape is correct."""
        S = torch.rand(10, 5) * 100 + 50  # 10 samples, 5 assets
        t = torch.rand(10)
        
        V = model(S, t)
        assert V.shape == (10,)
    
    def test_output_positive(self, model):
        """Test option values are positive."""
        S = torch.rand(5, 5) * 100 + 50
        t = torch.rand(5) * 0.5
        
        V = model(S, t)
        assert torch.all(V >= 0)
    
    def test_gradient_flow(self, model):
        """Test gradients flow through model."""
        S = torch.rand(3, 5) * 100 + 50
        S.requires_grad_(True)
        t = torch.rand(3)
        t.requires_grad_(True)
        
        V = model(S, t)
        loss = V.sum()
        loss.backward()
        
        assert S.grad is not None
        assert t.grad is not None
        assert torch.any(S.grad != 0)
    
    def test_single_sample(self, model):
        """Test with single sample."""
        S = torch.rand(1, 5) * 100 + 50
        t = torch.rand(1)
        
        V = model(S, t)
        assert V.shape == (1,)


class TestEnsembleHybridBasketPINN:
    """Test ensemble hybrid basket PINN."""
    
    @pytest.fixture
    def model(self):
        return EnsembleHybridBasketPINN(
            n_assets=5,
            n_qubits_per_vqc=4,
            n_layers=2,
            hidden_dim=16,
        )
    
    def test_correct_number_of_pairs(self, model):
        """Test we have the right number of VQCs for pairs."""
        # 5 assets = 10 pairs
        assert len(model.pairs) == 10
        assert len(model.vqcs) == 10
    
    def test_forward_shape(self, model):
        """Test output shape."""
        S = torch.rand(8, 5) * 100 + 50
        t = torch.rand(8)
        
        V = model(S, t)
        assert V.shape == (8,)
    
    def test_output_positive(self, model):
        """Test positive outputs."""
        S = torch.rand(4, 5) * 100 + 50
        t = torch.rand(4)
        
        V = model(S, t)
        assert torch.all(V >= 0)


class TestQuantumEnhancedBasketPINN:
    """Test quantum-enhanced (residual) basket PINN."""
    
    @pytest.fixture
    def model(self):
        return QuantumEnhancedBasketPINN(
            n_assets=5,
            n_qubits=4,
            n_layers=2,
            classical_dims=[32, 32],
        )
    
    def test_forward_shape(self, model):
        """Test output shape."""
        S = torch.rand(6, 5) * 100 + 50
        t = torch.rand(6)
        
        V = model(S, t)
        assert V.shape == (6,)
    
    def test_alpha_parameter(self, model):
        """Test alpha is learnable."""
        assert hasattr(model, 'alpha')
        assert model.alpha.requires_grad
    
    def test_classical_and_quantum_branches(self, model):
        """Test both branches contribute to output."""
        S = torch.rand(3, 5) * 100 + 50
        t = torch.rand(3)
        
        # Get outputs before and after zeroing alpha
        V1 = model(S, t).detach().clone()
        
        with torch.no_grad():
            old_alpha = model.alpha.clone()
            model.alpha.fill_(0)
        
        V2 = model(S, t).detach()
        
        with torch.no_grad():
            model.alpha.fill_(old_alpha.item())
        
        # Outputs should differ when alpha != 0
        if old_alpha.item() != 0:
            assert not torch.allclose(V1, V2)


class TestFactoryFunction:
    """Test create_hybrid_basket_pinn factory."""
    
    def test_create_compressed(self):
        """Test creating compressed architecture."""
        model = create_hybrid_basket_pinn(
            n_assets=5,
            architecture="compressed",
            n_qubits=4,
        )
        assert isinstance(model, HybridBasketPINN)
    
    def test_create_ensemble(self):
        """Test creating ensemble architecture."""
        model = create_hybrid_basket_pinn(
            n_assets=5,
            architecture="ensemble",
            n_qubits=4,
        )
        assert isinstance(model, EnsembleHybridBasketPINN)
    
    def test_create_enhanced(self):
        """Test creating enhanced architecture."""
        model = create_hybrid_basket_pinn(
            n_assets=5,
            architecture="enhanced",
            n_qubits=4,
        )
        assert isinstance(model, QuantumEnhancedBasketPINN)
    
    def test_invalid_architecture(self):
        """Test error on invalid architecture."""
        with pytest.raises(ValueError):
            create_hybrid_basket_pinn(architecture="invalid")


class TestDifferentAssetCounts:
    """Test models with different numbers of assets."""
    
    @pytest.mark.parametrize("n_assets", [2, 3, 5, 7])
    def test_compressed_various_assets(self, n_assets):
        """Test compressed model with various asset counts."""
        model = HybridBasketPINN(n_assets=n_assets, n_qubits=4, n_layers=1)
        S = torch.rand(4, n_assets) * 100 + 50
        t = torch.rand(4)
        
        V = model(S, t)
        assert V.shape == (4,)
    
    @pytest.mark.parametrize("n_assets", [3, 4, 5])
    def test_ensemble_various_assets(self, n_assets):
        """Test ensemble model with various asset counts."""
        model = EnsembleHybridBasketPINN(
            n_assets=n_assets, 
            n_qubits_per_vqc=4, 
            n_layers=1
        )
        S = torch.rand(4, n_assets) * 100 + 50
        t = torch.rand(4)
        
        V = model(S, t)
        assert V.shape == (4,)
        
        # Check number of pairs
        expected_pairs = n_assets * (n_assets - 1) // 2
        assert len(model.pairs) == expected_pairs
