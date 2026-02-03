"""
Hybrid Quantum-Classical PINN for Multi-Asset Basket Options.

Challenge: 5 assets + time = 6D input, but VQCs typically handle 2-8 qubits efficiently.

Approach:
1. Classical encoder compresses 6D → low-dim representation
2. VQC processes compressed representation
3. Classical decoder expands to option value

Alternative: Multi-VQC ensemble where each VQC handles asset pairs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .variational import QuantumLayer


class HybridBasketPINN(nn.Module):
    """
    Hybrid quantum-classical PINN for basket options.
    
    Architecture:
        6D input → Classical Encoder → 2D → VQC → Classical Decoder → 1D output
    
    The classical encoder learns to compress the high-dimensional asset space
    into a low-dimensional quantum-compatible representation.
    """
    
    def __init__(
        self,
        n_assets: int = 5,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoder_dims: list[int] = [64, 32, 16],
        decoder_dims: list[int] = [32, 64],
        S_max: float = 200.0,
        T_max: float = 1.0,
        circuit_type: str = "hardware_efficient",
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.S_max = S_max
        self.T_max = T_max
        
        # Encoder: (n_assets + 1) → 2 for quantum input
        encoder_layers = []
        in_dim = n_assets + 1  # S1..Sn + t
        for h in encoder_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h),
                nn.Tanh(),
            ])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, 2))  # Output 2D for VQC
        encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Quantum layer
        self.quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            circuit_type=circuit_type,
        )
        
        # Decoder: 1 (from VQC) + context → 1
        # We also pass encoded context to decoder for richer representation
        decoder_layers = []
        in_dim = 1 + encoder_dims[-1]  # quantum output + last encoder hidden
        for h in decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
            ])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, 1))
        decoder_layers.append(nn.Softplus())  # Positive output
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store encoder hidden for skip connection
        self.encoder_hidden = None
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid network.
        
        Args:
            S: Asset prices, shape (batch, n_assets)
            t: Time values, shape (batch,)
            
        Returns:
            Option values, shape (batch,)
        """
        batch_size = S.shape[0]
        
        # Normalize inputs
        S_norm = S / self.S_max
        t_norm = t.view(-1, 1) / self.T_max
        
        # Combine inputs
        x = torch.cat([S_norm, t_norm], dim=-1)  # (batch, n_assets + 1)
        
        # Encoder with skip connection capture
        for i, layer in enumerate(self.encoder[:-2]):
            x = layer(x)
            if i == len(self.encoder) - 4:  # Save before final linear
                encoder_hidden = x.clone()
        
        x = self.encoder[-2](x)  # Linear to 2D
        x = self.encoder[-1](x)  # Tanh
        
        # Scale for quantum encoding
        x_quantum = x * np.pi
        
        # Quantum circuit
        q_out = self.quantum(x_quantum)  # (batch,)
        
        # Decoder with skip connection
        decoder_input = torch.cat([q_out.unsqueeze(-1), encoder_hidden], dim=-1)
        V = self.decoder(decoder_input).squeeze(-1)
        
        # Scale by weighted average of asset prices (approximate intrinsic value scale)
        S_weighted = S.mean(dim=-1)  # Simple average
        V = V * S_weighted
        
        return V


class EnsembleHybridBasketPINN(nn.Module):
    """
    Ensemble of VQCs for basket options.
    
    Each VQC handles a subset of assets (e.g., pairs), and outputs
    are combined via classical aggregation.
    
    This avoids the dimensionality bottleneck by distributing
    the high-dimensional problem across multiple quantum circuits.
    """
    
    def __init__(
        self,
        n_assets: int = 5,
        n_qubits_per_vqc: int = 4,
        n_layers: int = 2,
        hidden_dim: int = 32,
        S_max: float = 200.0,
        T_max: float = 1.0,
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.S_max = S_max
        self.T_max = T_max
        
        # Create VQCs for each pair of assets
        # For 5 assets: 10 pairs
        self.n_pairs = n_assets * (n_assets - 1) // 2
        self.pairs = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
        
        # Pre-processing for each pair
        self.pair_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3, hidden_dim),  # 2 assets + time
                nn.Tanh(),
                nn.Linear(hidden_dim, 2),
                nn.Tanh(),
            )
            for _ in range(self.n_pairs)
        ])
        
        # VQC for each pair
        self.vqcs = nn.ModuleList([
            QuantumLayer(n_qubits_per_vqc, n_layers)
            for _ in range(self.n_pairs)
        ])
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(self.n_pairs + n_assets + 1, hidden_dim * 2),  # VQC outputs + raw features
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble network.
        
        Args:
            S: Asset prices, shape (batch, n_assets)
            t: Time values, shape (batch,)
            
        Returns:
            Option values, shape (batch,)
        """
        batch_size = S.shape[0]
        
        # Normalize
        S_norm = S / self.S_max
        t_norm = t.view(-1, 1) / self.T_max
        
        # Process each pair through its VQC
        vqc_outputs = []
        for idx, (i, j) in enumerate(self.pairs):
            # Extract pair features
            pair_input = torch.cat([
                S_norm[:, i:i+1],
                S_norm[:, j:j+1],
                t_norm,
            ], dim=-1)
            
            # Encode and run VQC
            encoded = self.pair_encoders[idx](pair_input) * np.pi
            vqc_out = self.vqcs[idx](encoded)
            vqc_outputs.append(vqc_out)
        
        # Stack VQC outputs
        vqc_stack = torch.stack(vqc_outputs, dim=-1)  # (batch, n_pairs)
        
        # Combine with raw features for aggregation
        agg_input = torch.cat([vqc_stack, S_norm, t_norm], dim=-1)
        V = self.aggregator(agg_input).squeeze(-1)
        
        # Scale by average asset price
        V = V * S.mean(dim=-1)
        
        return V


class QuantumEnhancedBasketPINN(nn.Module):
    """
    Classical basket PINN with quantum correction.
    
    V(S,t) = V_classical(S,t) + α * V_quantum(S,t)
    
    The quantum circuit provides a learned correction to the
    classical PINN solution.
    """
    
    def __init__(
        self,
        n_assets: int = 5,
        n_qubits: int = 4,
        n_layers: int = 2,
        classical_dims: list[int] = [128, 128, 128, 128],
        S_max: float = 200.0,
        T_max: float = 1.0,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.S_max = S_max
        self.T_max = T_max
        
        # Classical PINN branch (standard architecture)
        layers = []
        in_dim = n_assets + 1
        for h in classical_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.classical = nn.Sequential(*layers)
        
        # Quantum correction branch
        self.quantum_encoder = nn.Sequential(
            nn.Linear(n_assets + 1, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Tanh(),
        )
        self.quantum = QuantumLayer(n_qubits, n_layers)
        self.quantum_decoder = nn.Linear(1, 1)
        
        # Learnable mixing coefficient
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum correction."""
        batch_size = S.shape[0]
        
        # Normalize
        S_norm = S / self.S_max
        t_norm = t.view(-1, 1) / self.T_max
        x = torch.cat([S_norm, t_norm], dim=-1)
        
        # Classical prediction
        V_classical = self.classical(x).squeeze(-1)
        
        # Quantum correction
        q_encoded = self.quantum_encoder(x) * np.pi
        q_out = self.quantum(q_encoded)
        V_quantum = self.quantum_decoder(q_out.unsqueeze(-1)).squeeze(-1)
        
        # Combined with scaling
        V = (V_classical + self.alpha * V_quantum)
        V = torch.nn.functional.softplus(V) * S.mean(dim=-1)
        
        return V


def create_hybrid_basket_pinn(
    n_assets: int = 5,
    architecture: str = "compressed",
    n_qubits: int = 4,
    n_layers: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function for hybrid basket PINNs.
    
    Args:
        n_assets: Number of assets in basket
        architecture: One of "compressed", "ensemble", "enhanced"
        n_qubits: Qubits per VQC
        n_layers: VQC layers
        
    Returns:
        Hybrid basket PINN model
    """
    if architecture == "compressed":
        return HybridBasketPINN(
            n_assets=n_assets,
            n_qubits=n_qubits,
            n_layers=n_layers,
            **kwargs,
        )
    elif architecture == "ensemble":
        return EnsembleHybridBasketPINN(
            n_assets=n_assets,
            n_qubits_per_vqc=n_qubits,
            n_layers=n_layers,
            **kwargs,
        )
    elif architecture == "enhanced":
        return QuantumEnhancedBasketPINN(
            n_assets=n_assets,
            n_qubits=n_qubits,
            n_layers=n_layers,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
