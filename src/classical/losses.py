"""
Loss functions for PINN training.

The total loss combines:
1. PDE residual loss - physics constraint
2. Boundary condition loss - spatial boundaries
3. Terminal condition loss - payoff at maturity (initial condition in backward time)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..pde.black_scholes import BSParams


class PINNLoss:
    """
    Combined loss for PINN training on Black-Scholes PDE.

    The loss function:
        L = λ_pde * L_PDE + λ_bc * L_BC + λ_ic * L_IC

    Where:
        L_PDE: Mean squared PDE residual at interior collocation points
        L_BC: Mean squared boundary condition violation
        L_IC: Mean squared terminal condition violation
    """

    def __init__(
        self,
        params: BSParams,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
        option_type: str = "call",
    ):
        self.params = params
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.option_type = option_type

    def pde_residual(
        self,
        model: nn.Module,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Black-Scholes PDE residual.

        The residual should be zero if the network satisfies the PDE:
            ∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
        """
        S = S.requires_grad_(True)
        t = t.requires_grad_(True)

        V = model(S, t)

        # First derivatives using autodiff
        dV_dt = torch.autograd.grad(
            V.sum(), t, create_graph=True, retain_graph=True
        )[0]
        dV_dS = torch.autograd.grad(
            V.sum(), S, create_graph=True, retain_graph=True
        )[0]

        # Second derivative
        d2V_dS2 = torch.autograd.grad(
            dV_dS.sum(), S, create_graph=True, retain_graph=True
        )[0]

        # PDE residual
        residual = (
            dV_dt
            + 0.5 * self.params.sigma**2 * S**2 * d2V_dS2
            + self.params.r * S * dV_dS
            - self.params.r * V
        )

        return residual

    def boundary_loss(
        self,
        model: nn.Module,
        S: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Boundary condition loss.

        For European call:
            V(0, t) = 0
            V(S_max, t) ≈ S - K*exp(-r*(T-t))  (for large S)
        """
        V = model(S, t)
        tau = self.params.T - t

        # S = 0 boundary
        mask_zero = S < 1e-6
        loss_zero = (V[mask_zero] ** 2).sum()

        # S = S_max boundary (asymptotic)
        mask_large = S > (model.S_max - 1e-6)
        if self.option_type == "call":
            target_large = S[mask_large] - self.params.K * torch.exp(
                -self.params.r * tau[mask_large]
            )
        else:
            target_large = torch.zeros_like(V[mask_large])

        loss_large = ((V[mask_large] - target_large) ** 2).sum()

        n_points = mask_zero.sum() + mask_large.sum()
        if n_points > 0:
            return (loss_zero + loss_large) / n_points
        return torch.tensor(0.0, device=S.device)

    def terminal_loss(
        self,
        model: nn.Module,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """
        Terminal condition loss (payoff at maturity).

        V(S, T) = max(S - K, 0) for call
        V(S, T) = max(K - S, 0) for put
        """
        t_T = torch.full_like(S, self.params.T)
        V = model(S, t_T)

        if self.option_type == "call":
            payoff = torch.relu(S - self.params.K)
        else:
            payoff = torch.relu(self.params.K - S)

        return ((V - payoff) ** 2).mean()

    def __call__(
        self,
        model: nn.Module,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            model: PINN model
            S_interior, t_interior: Interior collocation points
            S_boundary, t_boundary: Boundary collocation points
            S_terminal: Terminal (t=T) collocation points

        Returns:
            Dictionary with loss components and total
        """
        losses = {}

        # PDE residual loss
        residual = self.pde_residual(model, S_interior, t_interior)
        losses["pde"] = self.lambda_pde * (residual**2).mean()

        # Boundary condition loss
        losses["bc"] = self.lambda_bc * self.boundary_loss(
            model, S_boundary, t_boundary
        )

        # Terminal condition loss
        losses["ic"] = self.lambda_ic * self.terminal_loss(model, S_terminal)

        # Total loss
        losses["total"] = losses["pde"] + losses["bc"] + losses["ic"]

        return losses


class AdaptiveWeightedLoss(PINNLoss):
    """
    PINN loss with adaptive loss weighting.

    Automatically balances loss terms during training using gradient statistics.
    Reference: Wang et al., "When and why PINNs fail to train"
    """

    def __init__(
        self,
        params: BSParams,
        alpha: float = 0.9,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.alpha = alpha
        self.running_max_grad = {"pde": 1.0, "bc": 1.0, "ic": 1.0}

    def update_weights(self, model: nn.Module, losses: dict[str, torch.Tensor]):
        """Update loss weights based on gradient magnitudes."""
        grads = {}

        for key in ["pde", "bc", "ic"]:
            model.zero_grad()
            losses[key].backward(retain_graph=True)

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grads[key] = grad_norm**0.5

        # Update running max
        max_grad = max(grads.values())
        for key in grads:
            self.running_max_grad[key] = (
                self.alpha * self.running_max_grad[key]
                + (1 - self.alpha) * max_grad / (grads[key] + 1e-8)
            )

        # Update lambdas
        self.lambda_pde = self.running_max_grad["pde"]
        self.lambda_bc = self.running_max_grad["bc"]
        self.lambda_ic = self.running_max_grad["ic"]
