"""
PINN for Volatility Surface Calibration (Inverse Problem).

This solves the inverse problem: given market option prices, find the
local volatility surface σ(K,T) that generates them.

The approach:
1. Learn a neural network σ(K,T) for local volatility
2. Use Black-Scholes with local vol to compute model prices
3. Minimize: data_loss (fit market) + physics_loss (Dupire PDE) + reg_loss (smooth vol)

This is fundamentally different from forward PINN:
- Forward: Given σ, solve PDE for V
- Inverse: Given V (prices), infer σ
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import time

from src.pde.dupire import (
    DupireParams,
    LocalVolNetwork,
    CallPriceNetwork,
    generate_calibration_data,
    black_scholes_call_torch,
)


class VolCalibrationPINN(nn.Module):
    """
    PINN for local volatility calibration.
    
    Architecture:
    - LocalVolNetwork: outputs σ(K, T)
    - Uses Black-Scholes to compute model call prices
    - Calibrates by matching market prices
    """
    
    def __init__(
        self,
        params: DupireParams,
        hidden_dims: list[int] = [64, 64, 64],
    ):
        super().__init__()
        
        self.params = params
        
        # Local volatility network
        self.vol_net = LocalVolNetwork(
            hidden_dims=hidden_dims,
            K_range=(params.K_min, params.K_max),
            T_range=(params.T_min, params.T_max),
        )
        
    def forward(self, K: torch.Tensor, T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute model call price and local vol.
        
        Args:
            K: Strike prices
            T: Maturities
            
        Returns:
            (call_prices, local_vols)
        """
        # Get local vol from network
        sigma = self.vol_net(K, T)
        
        # Compute Black-Scholes call with this vol
        S0 = torch.tensor(self.params.S0, dtype=K.dtype, device=K.device)
        C = black_scholes_call_torch(S0, K, T, self.params.r, sigma)
        
        return C, sigma
    
    def get_vol_surface(
        self,
        K_grid: torch.Tensor,
        T_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Get volatility surface on a grid."""
        K_flat = K_grid.flatten()
        T_flat = T_grid.flatten()
        
        with torch.no_grad():
            sigma = self.vol_net(K_flat, T_flat)
        
        return sigma.reshape(K_grid.shape)


class VolCalibrationTrainer:
    """
    Trainer for volatility calibration PINN.
    
    Loss components:
    1. Data loss: ||C_model - C_market||²
    2. Smoothness loss: ||∇σ||² + ||∇²σ||² (regularization)
    3. Arbitrage loss: enforce σ > 0, d²C/dK² > 0 (convexity)
    """
    
    def __init__(
        self,
        model: VolCalibrationPINN,
        market_data: dict[str, torch.Tensor],
        lr: float = 1e-3,
        lambda_smooth: float = 0.01,
        lambda_arb: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        
        # Market data
        self.K_market = market_data["K"].to(device)
        self.T_market = market_data["T"].to(device)
        self.C_market = market_data["C_market"].to(device)
        self.IV_true = market_data.get("IV_true", None)
        if self.IV_true is not None:
            self.IV_true = self.IV_true.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=200, factor=0.5, min_lr=1e-6
        )
        
        # Loss weights
        self.lambda_smooth = lambda_smooth
        self.lambda_arb = lambda_arb
        
        # History
        self.history = {
            "total": [],
            "data": [],
            "smooth": [],
            "arb": [],
            "vol_mse": [],
            "time": [],
        }
    
    def compute_data_loss(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
        C_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between model and market prices."""
        C_model, _ = self.model(K, T)
        
        # Relative error (prices vary in magnitude)
        rel_error = (C_model - C_target) / (C_target + 1e-6)
        loss = (rel_error ** 2).mean()
        
        return loss
    
    def compute_smoothness_loss(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Regularize volatility surface for smoothness.
        
        Penalizes large gradients (first and second order).
        """
        K = K.requires_grad_(True)
        T = T.requires_grad_(True)
        
        sigma = self.model.vol_net(K, T)
        
        # First derivatives
        dsigma_dK = torch.autograd.grad(
            sigma.sum(), K, create_graph=True, retain_graph=True
        )[0]
        dsigma_dT = torch.autograd.grad(
            sigma.sum(), T, create_graph=True, retain_graph=True
        )[0]
        
        # L2 norm of gradients
        grad_loss = (dsigma_dK ** 2).mean() + (dsigma_dT ** 2).mean()
        
        # Second derivatives (curvature)
        d2sigma_dK2 = torch.autograd.grad(
            dsigma_dK.sum(), K, create_graph=True, retain_graph=True
        )[0]
        d2sigma_dT2 = torch.autograd.grad(
            dsigma_dT.sum(), T, create_graph=True, retain_graph=True
        )[0]
        
        curv_loss = (d2sigma_dK2 ** 2).mean() + (d2sigma_dT2 ** 2).mean()
        
        return grad_loss + 0.1 * curv_loss
    
    def compute_arbitrage_loss(
        self,
        K: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enforce no-arbitrage constraints.
        
        Key constraints:
        1. σ(K,T) > 0 (already enforced by Softplus)
        2. ∂²C/∂K² > 0 (call prices are convex in strike)
        3. ∂C/∂T > 0 (calendar arbitrage)
        """
        K = K.requires_grad_(True)
        T = T.requires_grad_(True)
        
        C, sigma = self.model(K, T)
        
        # Convexity: d²C/dK² > 0
        dC_dK = torch.autograd.grad(
            C.sum(), K, create_graph=True, retain_graph=True
        )[0]
        d2C_dK2 = torch.autograd.grad(
            dC_dK.sum(), K, create_graph=True, retain_graph=True
        )[0]
        
        # Penalize negative butterfly (arbitrage)
        convexity_violation = torch.relu(-d2C_dK2)
        
        # Calendar: dC/dT > 0 (longer maturity = higher value)
        dC_dT = torch.autograd.grad(
            C.sum(), T, create_graph=True, retain_graph=True
        )[0]
        calendar_violation = torch.relu(-dC_dT)
        
        loss = convexity_violation.mean() + calendar_violation.mean()
        
        return loss
    
    def train_step(self) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Data loss
        data_loss = self.compute_data_loss(
            self.K_market, self.T_market, self.C_market
        )
        
        # Smoothness regularization (on market points)
        smooth_loss = self.compute_smoothness_loss(
            self.K_market, self.T_market
        )
        
        # Arbitrage constraints
        arb_loss = self.compute_arbitrage_loss(
            self.K_market, self.T_market
        )
        
        # Total loss
        total_loss = (
            data_loss 
            + self.lambda_smooth * smooth_loss
            + self.lambda_arb * arb_loss
        )
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Compute vol MSE if we have ground truth
        vol_mse = 0.0
        if self.IV_true is not None:
            with torch.no_grad():
                sigma_pred = self.model.vol_net(self.K_market, self.T_market)
                vol_mse = ((sigma_pred - self.IV_true) ** 2).mean().item()
        
        return {
            "total": total_loss.item(),
            "data": data_loss.item(),
            "smooth": smooth_loss.item(),
            "arb": arb_loss.item(),
            "vol_mse": vol_mse,
        }
    
    def train(
        self,
        n_epochs: int = 2000,
        log_every: int = 100,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs
            log_every: Logging frequency
            
        Returns:
            Training history
        """
        print(f"Training VolCalibrationPINN for {n_epochs} epochs...")
        print(f"  Market points: {len(self.K_market)}")
        print(f"  Lambda smooth: {self.lambda_smooth}")
        print(f"  Lambda arb: {self.lambda_arb}")
        print(f"  Device: {self.device}")
        print()
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            metrics = self.train_step()
            
            epoch_time = time.time() - start_time
            
            # Record history
            self.history["total"].append(metrics["total"])
            self.history["data"].append(metrics["data"])
            self.history["smooth"].append(metrics["smooth"])
            self.history["arb"].append(metrics["arb"])
            self.history["vol_mse"].append(metrics["vol_mse"])
            self.history["time"].append(epoch_time)
            
            # Update scheduler
            self.scheduler.step(metrics["total"])
            
            # Log
            if epoch % log_every == 0 or epoch == n_epochs - 1:
                elapsed = sum(self.history["time"])
                print(
                    f"Epoch {epoch:5d}: "
                    f"total={metrics['total']:.6f}, "
                    f"data={metrics['data']:.6f}, "
                    f"vol_mse={metrics['vol_mse']:.6f}, "
                    f"time={elapsed:.1f}s"
                )
        
        return self.history


def evaluate_calibration(
    model: VolCalibrationPINN,
    market_data: dict[str, torch.Tensor],
    device: str = "cpu",
) -> dict:
    """
    Evaluate calibration quality.
    
    Returns:
        Dictionary with price_mse, vol_mse, max_price_error, etc.
    """
    model.eval()
    
    K = market_data["K"].to(device)
    T = market_data["T"].to(device)
    C_market = market_data["C_market"].to(device)
    IV_true = market_data.get("IV_true")
    
    with torch.no_grad():
        C_model, sigma_model = model(K, T)
    
    # Price errors
    price_errors = (C_model - C_market).cpu().numpy()
    price_mse = float(np.mean(price_errors ** 2))
    price_mae = float(np.mean(np.abs(price_errors)))
    max_price_error = float(np.max(np.abs(price_errors)))
    
    # Relative price error
    rel_errors = np.abs(price_errors) / (C_market.cpu().numpy() + 1e-6) * 100
    mean_rel_error = float(np.mean(rel_errors))
    
    results = {
        "price_mse": price_mse,
        "price_mae": price_mae,
        "max_price_error": max_price_error,
        "mean_rel_error_pct": mean_rel_error,
    }
    
    # Vol errors (if ground truth available)
    if IV_true is not None:
        IV_true = IV_true.to(device)
        sigma_model_np = sigma_model.cpu().numpy()
        IV_true_np = IV_true.cpu().numpy()
        
        vol_errors = sigma_model_np - IV_true_np
        results["vol_mse"] = float(np.mean(vol_errors ** 2))
        results["vol_mae"] = float(np.mean(np.abs(vol_errors)))
        results["max_vol_error"] = float(np.max(np.abs(vol_errors)))
        results["mean_vol_error_pct"] = float(np.mean(np.abs(vol_errors) / IV_true_np) * 100)
    
    return results
