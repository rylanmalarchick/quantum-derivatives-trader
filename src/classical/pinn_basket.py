"""
High-dimensional PINN for multi-asset basket options.

Handles N-dimensional input (S₁, S₂, ..., Sₙ, t) for basket option pricing.
Uses Latin Hypercube Sampling for efficient high-dim collocation.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import time

from .networks import MLP
from ..pde.basket import BasketParams, basket_payoff, basket_pde_residual


class BasketPINN(nn.Module):
    """
    Physics-Informed Neural Network for basket option pricing.
    
    Input: (S₁, S₂, ..., Sₙ, t) - n asset prices and time
    Output: V - option value
    
    The network learns to satisfy the N-dimensional Black-Scholes PDE
    while matching the terminal payoff condition.
    """
    
    def __init__(
        self,
        n_assets: int = 5,
        hidden_dims: list[int] = [128, 128, 128, 128, 128, 128],
        S_max: np.ndarray = None,
        T_max: float = 1.0,
        activation: str = "tanh",
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.T_max = T_max
        
        # Default S_max if not provided
        if S_max is None:
            self.S_max = torch.tensor([200.0] * n_assets)
        else:
            self.S_max = torch.tensor(S_max, dtype=torch.float32)
        
        # Input: n_assets + 1 (for time)
        # Output: 1 (option value)
        self.network = MLP(
            in_dim=n_assets + 1,
            out_dim=1,
            hidden_dims=hidden_dims,
        )
        
        # Use swish activation for smoother gradients in high-dim
        if activation == "swish":
            self._replace_activations(nn.SiLU())
    
    def _replace_activations(self, new_activation: nn.Module):
        """Replace all Tanh activations with a new activation."""
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Tanh):
                parent = self.network
                parts = name.split('.')
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_activation)
    
    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            S: Asset prices tensor, shape (batch, n_assets)
            t: Time tensor, shape (batch,) or (batch, 1)
            
        Returns:
            Option value tensor, shape (batch,)
        """
        # Ensure correct shapes
        if S.dim() == 1:
            S = S.unsqueeze(0)
        t = t.view(-1)
        
        # Normalize inputs to [0, 1]
        S_max = self.S_max.to(S.device)
        S_norm = S / S_max
        t_norm = t / self.T_max
        
        # Stack inputs: (batch, n_assets + 1)
        x = torch.cat([S_norm, t_norm.unsqueeze(-1)], dim=-1)
        
        # Forward through network
        V = self.network(x).squeeze(-1)
        
        # Scale output by basket value for better learning dynamics
        # This is a physics-informed output scaling
        basket_scale = (S * 0.2).sum(dim=-1)  # Approximate ATM value scale
        V = V * basket_scale
        
        return V
    
    def predict_with_greeks(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict option value and Greeks using autodiff.
        
        Returns:
            Dictionary with V, deltas (n_assets), gammas (n_assets)
        """
        S = S.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        
        V = self.forward(S, t)
        
        # Deltas = ∂V/∂Sᵢ
        dV_dS = torch.autograd.grad(
            V.sum(), S, create_graph=True, retain_graph=True
        )[0]  # Shape: (batch, n_assets)
        
        # Gammas = ∂²V/∂Sᵢ² (diagonal only)
        gammas = []
        for i in range(self.n_assets):
            d2V_dSi2 = torch.autograd.grad(
                dV_dS[:, i].sum(), S, create_graph=True, retain_graph=True
            )[0][:, i]
            gammas.append(d2V_dSi2)
        gammas = torch.stack(gammas, dim=-1)
        
        # Theta = ∂V/∂t
        theta = torch.autograd.grad(
            V.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        return {
            "V": V.detach(),
            "deltas": dV_dS.detach(),
            "gammas": gammas.detach(),
            "theta": theta.detach(),
        }


class BasketPINNTrainer:
    """Trainer for basket PINN with physics-informed loss."""
    
    def __init__(
        self,
        model: BasketPINN,
        params: BasketParams,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.params = params
        self.device = device
        
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, 
        )
    
    def compute_pde_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss at interior points."""
        S = S.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        
        V = self.model(S, t)
        
        residual = basket_pde_residual(V, S, t, self.params)
        
        return (residual ** 2).mean()
    
    def compute_terminal_loss(self, S: torch.Tensor) -> torch.Tensor:
        """Compute terminal condition (payoff) loss."""
        t = torch.full((S.shape[0],), self.params.T, device=self.device)
        
        V_pred = self.model(S, t)
        V_true = basket_payoff(S, self.params)
        
        return ((V_pred - V_true) ** 2).mean()
    
    def train_step(
        self,
        S_int: torch.Tensor,
        t_int: torch.Tensor,
        S_term: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        S_int = S_int.to(self.device)
        t_int = t_int.to(self.device)
        S_term = S_term.to(self.device)
        
        # Compute losses
        loss_pde = self.compute_pde_loss(S_int, t_int)
        loss_ic = self.compute_terminal_loss(S_term)
        
        # Total loss
        loss_total = self.lambda_pde * loss_pde + self.lambda_ic * loss_ic
        
        # Backward and optimize
        loss_total.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            "total": loss_total.item(),
            "pde": loss_pde.item(),
            "ic": loss_ic.item(),
        }
    
    def train(
        self,
        n_epochs: int,
        n_interior: int = 10000,
        n_terminal: int = 5000,
        resample_every: int = 100,
        log_every: int = 50,
        callback: Optional[callable] = None,
    ) -> dict:
        """
        Full training loop with periodic resampling.
        
        Args:
            n_epochs: Number of training epochs
            n_interior: Number of interior collocation points
            n_terminal: Number of terminal collocation points
            resample_every: Resample collocation points every N epochs
            log_every: Log progress every N epochs
            callback: Optional callback(epoch, losses) function
            
        Returns:
            Training history dictionary
        """
        from ..pde.basket import generate_basket_collocation_lhs
        
        history = {"total": [], "pde": [], "ic": [], "time": []}
        
        print(f"Training BasketPINN for {n_epochs} epochs...")
        print(f"  Interior points: {n_interior}")
        print(f"  Terminal points: {n_terminal}")
        print(f"  Assets: {self.params.n_assets}")
        print(f"  Device: {self.device}")
        print()
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Resample collocation points periodically
            if epoch % resample_every == 0:
                data = generate_basket_collocation_lhs(
                    self.params,
                    n_interior=n_interior,
                    n_terminal=n_terminal,
                    seed=epoch,
                )
                S_int = data["S_int"]
                t_int = data["t_int"]
                S_term = data["S_term"]
            
            # Train step
            losses = self.train_step(S_int, t_int, S_term)
            
            # Record history
            for key in history:
                if key != "time":
                    history[key].append(losses[key])
            history["time"].append(time.time() - start_time)
            
            # Learning rate scheduling
            self.scheduler.step(losses["total"])
            
            # Logging
            if epoch % log_every == 0 or epoch == n_epochs - 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:5d}: total={losses['total']:.4f}, "
                      f"pde={losses['pde']:.4f}, ic={losses['ic']:.4f}, "
                      f"time={elapsed:.1f}s")
            
            # Callback
            if callback is not None:
                callback(epoch, losses)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        
        return history


def evaluate_basket_pinn(
    model: BasketPINN,
    params: BasketParams,
    n_test: int = 1000,
    device: str = "cpu",
) -> dict:
    """
    Evaluate PINN against Monte Carlo baseline.
    
    Args:
        model: Trained BasketPINN
        params: Basket parameters
        n_test: Number of test points
        device: Device for evaluation
        
    Returns:
        Dictionary with metrics
    """
    from ..pde.basket import monte_carlo_basket, generate_basket_collocation_lhs
    from scipy.stats import qmc
    
    model.eval()
    
    # Generate test points using LHS
    sampler = qmc.LatinHypercube(d=params.n_assets, seed=999)
    samples = sampler.random(n=n_test)
    
    S_test = np.zeros((n_test, params.n_assets))
    for i in range(params.n_assets):
        S_test[:, i] = params.S_min[i] + samples[:, i] * (params.S_max[i] - params.S_min[i])
    
    # PINN predictions at t=0
    S_tensor = torch.tensor(S_test, dtype=torch.float32, device=device)
    t_tensor = torch.zeros(n_test, device=device)
    
    with torch.no_grad():
        V_pinn = model(S_tensor, t_tensor).cpu().numpy()
    
    # Monte Carlo baseline (slower but accurate)
    V_mc = np.zeros(n_test)
    mc_std = np.zeros(n_test)
    
    print(f"Computing MC baseline for {n_test} test points...")
    for i in range(n_test):
        result = monte_carlo_basket(params, S_test[i], n_paths=50000, seed=i)
        V_mc[i] = result["price"]
        mc_std[i] = result["std_error"]
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_test} done")
    
    # Compute metrics
    errors = V_pinn - V_mc
    abs_errors = np.abs(errors)
    
    # Relative errors (avoid div by zero)
    rel_errors = np.abs(errors) / np.maximum(V_mc, 0.1) * 100
    
    metrics = {
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(abs_errors)),
        "max_error": float(np.max(abs_errors)),
        "mean_rel_error_pct": float(np.mean(rel_errors)),
        "median_rel_error_pct": float(np.median(rel_errors)),
        "pinn_prices": V_pinn,
        "mc_prices": V_mc,
        "mc_std_errors": mc_std,
        "test_spots": S_test,
    }
    
    return metrics
