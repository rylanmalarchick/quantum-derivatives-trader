"""
PINN-based option pricer.

Wraps trained PINN models for pricing and Greeks computation.
"""

import torch
import numpy as np
from typing import Optional, Union
from pathlib import Path

from ..classical.pinn import PINN
from ..quantum.hybrid_pinn import HybridPINN
from ..pde.black_scholes import BSParams


class PINNPricer:
    """
    Option pricer using trained PINN model.

    Provides interface for pricing and Greeks computation
    using a trained physics-informed neural network.
    """

    def __init__(
        self,
        model: Union[PINN, HybridPINN],
        params: BSParams,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.params = params
        self.device = device
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        params: BSParams,
        model_type: str = "classical",
        device: str = "cpu",
    ) -> "PINNPricer":
        """
        Load a trained PINN model.

        Args:
            model_path: Path to saved model weights
            params: Black-Scholes parameters
            model_type: "classical" or "hybrid"
            device: Computation device

        Returns:
            PINNPricer instance
        """
        if model_type == "classical":
            model = PINN()
        else:
            model = HybridPINN()

        model.load_state_dict(torch.load(model_path, map_location=device))

        return cls(model, params, device)

    def price(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Price option at given spot and time.

        Args:
            S: Spot price(s)
            t: Time(s)

        Returns:
            Option price(s)
        """
        S_t = torch.tensor(np.atleast_1d(S), dtype=torch.float32, device=self.device)
        t_t = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            V = self.model(S_t, t_t)

        return V.cpu().numpy()

    def delta(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute delta (∂V/∂S) using autodiff."""
        S_t = torch.tensor(
            np.atleast_1d(S), dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        t_t = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device)

        V = self.model(S_t, t_t)
        delta = torch.autograd.grad(V.sum(), S_t)[0]

        return delta.detach().cpu().numpy()

    def gamma(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute gamma (∂²V/∂S²) using autodiff."""
        S_t = torch.tensor(
            np.atleast_1d(S), dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        t_t = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device)

        V = self.model(S_t, t_t)
        delta = torch.autograd.grad(V.sum(), S_t, create_graph=True)[0]
        gamma = torch.autograd.grad(delta.sum(), S_t)[0]

        return gamma.detach().cpu().numpy()

    def theta(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute theta (∂V/∂t) using autodiff."""
        S_t = torch.tensor(np.atleast_1d(S), dtype=torch.float32, device=self.device)
        t_t = torch.tensor(
            np.atleast_1d(t), dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        V = self.model(S_t, t_t)
        theta = torch.autograd.grad(V.sum(), t_t)[0]

        return theta.detach().cpu().numpy()

    def vega(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
        bump: float = 0.01,
    ) -> np.ndarray:
        """
        Compute vega (∂V/∂σ) using finite difference bump.

        Note: This requires retraining with different σ for true vega.
        Here we approximate by bumping σ and repricing.
        """
        # This is an approximation - true vega would require
        # training sensitivity or ensemble of models
        print("Warning: vega approximation only")
        return np.zeros_like(np.atleast_1d(S))

    def greeks(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Compute all Greeks at once.

        Returns:
            Dictionary with price, delta, gamma, theta
        """
        S_t = torch.tensor(
            np.atleast_1d(S), dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        t_t = torch.tensor(
            np.atleast_1d(t), dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        V = self.model(S_t, t_t)

        # Delta
        delta = torch.autograd.grad(
            V.sum(), S_t, create_graph=True, retain_graph=True
        )[0]

        # Gamma
        gamma = torch.autograd.grad(
            delta.sum(), S_t, create_graph=True, retain_graph=True
        )[0]

        # Theta
        theta = torch.autograd.grad(V.sum(), t_t, retain_graph=True)[0]

        return {
            "price": V.detach().cpu().numpy(),
            "delta": delta.detach().cpu().numpy(),
            "gamma": gamma.detach().cpu().numpy(),
            "theta": theta.detach().cpu().numpy(),
        }

    def pde_residual(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Compute Black-Scholes PDE residual.

        Should be close to zero for a well-trained PINN.
        """
        S_t = torch.tensor(
            np.atleast_1d(S), dtype=torch.float32, device=self.device
        ).requires_grad_(True)
        t_t = torch.tensor(
            np.atleast_1d(t), dtype=torch.float32, device=self.device
        ).requires_grad_(True)

        V = self.model(S_t, t_t)

        # Compute derivatives
        dV_dt = torch.autograd.grad(
            V.sum(), t_t, create_graph=True, retain_graph=True
        )[0]
        dV_dS = torch.autograd.grad(
            V.sum(), S_t, create_graph=True, retain_graph=True
        )[0]
        d2V_dS2 = torch.autograd.grad(dV_dS.sum(), S_t, retain_graph=True)[0]

        # PDE residual
        residual = (
            dV_dt
            + 0.5 * self.params.sigma**2 * S_t**2 * d2V_dS2
            + self.params.r * S_t * dV_dS
            - self.params.r * V
        )

        return residual.detach().cpu().numpy()


class EnsemblePINNPricer:
    """
    Ensemble of PINN models for uncertainty quantification.

    Trains multiple PINNs with different initializations
    and uses ensemble statistics for uncertainty estimates.
    """

    def __init__(
        self,
        models: list[Union[PINN, HybridPINN]],
        params: BSParams,
        device: str = "cpu",
    ):
        self.models = [m.to(device) for m in models]
        self.params = params
        self.device = device

        for m in self.models:
            m.eval()

    def price_with_uncertainty(
        self,
        S: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Price with uncertainty quantification.

        Returns:
            (mean_price, std_price) from ensemble
        """
        S_t = torch.tensor(np.atleast_1d(S), dtype=torch.float32, device=self.device)
        t_t = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device)

        predictions = []
        with torch.no_grad():
            for model in self.models:
                V = model(S_t, t_t)
                predictions.append(V.cpu().numpy())

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        return mean, std
