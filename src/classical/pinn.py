"""
Classical Physics-Informed Neural Network for option pricing.

The loss function combines:
1. PDE residual loss (physics constraint)
2. Boundary condition loss
3. Terminal condition loss (payoff at maturity)
"""

import torch
import torch.nn as nn
from typing import Optional

from .networks import MLP, ResidualMLP
from ..pde.black_scholes import BSParams


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for option pricing.

    Input: (S, t) - spot price and time
    Output: V - option value

    The network learns to satisfy the Black-Scholes PDE while matching
    boundary and terminal conditions.
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64, 64, 64, 64],
        S_max: float = 200.0,
        T_max: float = 1.0,
        use_residual: bool = False,
    ):
        super().__init__()

        self.S_max = S_max
        self.T_max = T_max

        if use_residual:
            self.network = ResidualMLP(
                in_dim=2,
                out_dim=1,
                hidden_dim=hidden_dims[0] if hidden_dims else 64,
                n_blocks=len(hidden_dims),
            )
        else:
            self.network = MLP(
                in_dim=2,
                out_dim=1,
                hidden_dims=hidden_dims,
            )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            S: Spot price tensor, shape (batch,) or (batch, 1)
            t: Time tensor, shape (batch,) or (batch, 1)

        Returns:
            Option value tensor, shape (batch,)
        """
        # Ensure 1D
        S = S.view(-1)
        t = t.view(-1)

        # Normalize inputs to [0, 1]
        S_norm = S / self.S_max
        t_norm = t / self.T_max

        # Stack inputs
        x = torch.stack([S_norm, t_norm], dim=-1)

        # Forward through network
        V = self.network(x).squeeze(-1)

        return V

    def predict_with_greeks(
        self,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Predict option value and Greeks using autodiff.

        Returns:
            Dictionary with V, delta, gamma, theta
        """
        S = S.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        V = self.forward(S, t)

        # Delta = ∂V/∂S
        delta = torch.autograd.grad(
            V.sum(), S, create_graph=True, retain_graph=True
        )[0]

        # Gamma = ∂²V/∂S²
        gamma = torch.autograd.grad(
            delta.sum(), S, create_graph=True, retain_graph=True
        )[0]

        # Theta = ∂V/∂t
        theta = torch.autograd.grad(
            V.sum(), t, create_graph=True, retain_graph=True
        )[0]

        return {
            "V": V.detach(),
            "delta": delta.detach(),
            "gamma": gamma.detach(),
            "theta": theta.detach(),
        }


class PINNTrainer:
    """
    Trainer for PINN with configurable optimization.
    """

    def __init__(
        self,
        model: PINN,
        params: BSParams,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ):
        self.model = model
        self.params = params
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-6
        )

        self.history = {"total": [], "pde": [], "bc": [], "ic": []}

    def compute_loss(
        self,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss components."""
        from .losses import PINNLoss

        loss_fn = PINNLoss(
            self.params,
            self.lambda_pde,
            self.lambda_bc,
            self.lambda_ic,
        )

        return loss_fn(
            self.model,
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal,
        )

    def train_step(
        self,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step."""
        self.optimizer.zero_grad()

        losses = self.compute_loss(
            S_interior, t_interior,
            S_boundary, t_boundary,
            S_terminal,
        )

        losses["total"].backward()
        self.optimizer.step()

        # Record history
        for key in self.history:
            self.history[key].append(losses[key].item())

        return {k: v.item() for k, v in losses.items()}

    def train(
        self,
        n_epochs: int,
        n_interior: int = 1000,
        n_boundary: int = 200,
        n_terminal: int = 200,
        print_every: int = 100,
    ) -> dict[str, list[float]]:
        """
        Full training loop.

        Generates fresh collocation points each epoch (important for PINNs).
        """
        from ..data.collocation import generate_collocation_points

        for epoch in range(n_epochs):
            # Generate fresh points
            S_int, t_int, S_bc, t_bc, S_term = generate_collocation_points(
                n_interior=n_interior,
                n_boundary=n_boundary,
                n_terminal=n_terminal,
                S_max=self.model.S_max,
                T=self.params.T,
            )

            losses = self.train_step(S_int, t_int, S_bc, t_bc, S_term)

            # Learning rate scheduling
            self.scheduler.step(losses["total"])

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: total={losses['total']:.6f}, "
                    f"pde={losses['pde']:.6f}, bc={losses['bc']:.6f}, "
                    f"ic={losses['ic']:.6f}"
                )

        return self.history
