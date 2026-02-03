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


class MoneynessWeightedLoss(PINNLoss):
    """
    PINN loss with moneyness-based weighting.
    
    Addresses the issue where models perform poorly on OTM options 
    (near-zero values) by weighting samples inversely to option value.
    
    Key insight: MSE naturally weights ITM options more heavily because
    errors there are larger in absolute terms. This loss function 
    rebalances to give OTM options more weight.
    """
    
    def __init__(
        self,
        params: BSParams,
        K: float = 100.0,
        otm_weight: float = 10.0,
        atm_weight: float = 5.0,
        itm_weight: float = 1.0,
        use_relative_error: bool = False,
        **kwargs,
    ):
        """
        Args:
            params: Black-Scholes parameters
            K: Strike price for determining moneyness
            otm_weight: Weight multiplier for OTM options (S/K < 0.95)
            atm_weight: Weight multiplier for ATM options (0.95 <= S/K <= 1.05)
            itm_weight: Weight multiplier for ITM options (S/K > 1.05)
            use_relative_error: If True, use relative error for terminal loss
        """
        super().__init__(params, **kwargs)
        self.K = K
        self.otm_weight = otm_weight
        self.atm_weight = atm_weight
        self.itm_weight = itm_weight
        self.use_relative_error = use_relative_error
    
    def _compute_weights(self, S: torch.Tensor) -> torch.Tensor:
        """Compute sample weights based on moneyness."""
        moneyness = S / self.K
        weights = torch.ones_like(S)
        
        # OTM: S/K < 0.95
        otm_mask = moneyness < 0.95
        weights[otm_mask] = self.otm_weight
        
        # ATM: 0.95 <= S/K <= 1.05
        atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
        weights[atm_mask] = self.atm_weight
        
        # ITM: S/K > 1.05
        itm_mask = moneyness > 1.05
        weights[itm_mask] = self.itm_weight
        
        return weights
    
    def terminal_loss(
        self,
        model: torch.nn.Module,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted terminal condition loss.
        
        Optionally uses relative error to handle the wide range of option values.
        """
        t_T = torch.full_like(S, self.params.T)
        V = model(S, t_T)
        
        if self.option_type == "call":
            payoff = torch.relu(S - self.params.K)
        else:
            payoff = torch.relu(self.params.K - S)
        
        weights = self._compute_weights(S)
        
        if self.use_relative_error:
            # Relative error with small epsilon for stability
            epsilon = 0.1  # Min price threshold
            rel_error = (V - payoff) ** 2 / (payoff.abs() + epsilon) ** 2
            return (weights * rel_error).mean()
        else:
            # Weighted MSE
            return (weights * (V - payoff) ** 2).mean()
    
    def __call__(
        self,
        model: torch.nn.Module,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted loss components."""
        losses = {}
        
        # PDE residual loss (weighted by moneyness at interior points)
        residual = self.pde_residual(model, S_interior, t_interior)
        weights = self._compute_weights(S_interior)
        losses["pde"] = self.lambda_pde * (weights * residual ** 2).mean()
        
        # Boundary condition loss (unweighted - already focused on extremes)
        losses["bc"] = self.lambda_bc * self.boundary_loss(
            model, S_boundary, t_boundary
        )
        
        # Terminal condition loss (weighted)
        losses["ic"] = self.lambda_ic * self.terminal_loss(model, S_terminal)
        
        # Total loss
        losses["total"] = losses["pde"] + losses["bc"] + losses["ic"]
        
        return losses


class LogPriceLoss(PINNLoss):
    """
    PINN loss that works in log-price space.
    
    Key insight: Option prices span many orders of magnitude.
    Training on log(V + epsilon) makes gradients more uniform.
    
    This is especially useful for the terminal condition, where
    the payoff can range from 0 to 100+.
    """
    
    def __init__(
        self,
        params: BSParams,
        epsilon: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            params: Black-Scholes parameters
            epsilon: Small offset to handle log(0)
        """
        super().__init__(params, **kwargs)
        self.epsilon = epsilon
    
    def terminal_loss(
        self,
        model: torch.nn.Module,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """Terminal condition in log-price space."""
        t_T = torch.full_like(S, self.params.T)
        V = model(S, t_T)
        
        if self.option_type == "call":
            payoff = torch.relu(S - self.params.K)
        else:
            payoff = torch.relu(self.params.K - S)
        
        # Log transform with offset
        log_V = torch.log(V.abs() + self.epsilon)
        log_payoff = torch.log(payoff + self.epsilon)
        
        return ((log_V - log_payoff) ** 2).mean()
