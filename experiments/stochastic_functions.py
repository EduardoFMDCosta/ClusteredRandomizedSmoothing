from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.stochastic import StochasticFunction
from control_action import Control_action


class AdditiveThreeMode1DGaussianMixture(StochasticFunction):
    def __init__(
        self,
        locs: torch.Tensor,
        probs: torch.Tensor,
        stds: torch.Tensor,
    ) -> None:
        self.locs = locs
        self.probs = probs
        self.stds = stds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")

        leading_shape = x.shape[:-1]
        locs = self.locs.to(device=x.device, dtype=x.dtype)
        stds = self.stds.to(device=x.device, dtype=x.dtype)
        probs = self.probs.to(device=x.device)

        component_idx = torch.distributions.Categorical(probs=probs).sample(leading_shape)
        noise = locs[component_idx] + stds[component_idx] * torch.randn(leading_shape, dtype=x.dtype, device=x.device)
        y = x[..., 0] + noise

        return y.unsqueeze(-1)


class MultiplicativeThreeMode1DGaussianMixture(StochasticFunction):
    def __init__(
        self,
        locs: torch.Tensor,
        probs: torch.Tensor,
        stds: torch.Tensor,
    ) -> None:
        self.locs = locs
        self.probs = probs
        self.stds = stds

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")

        leading_shape = x.shape[:-1]
        locs = self.locs.to(device=x.device, dtype=x.dtype)
        stds = self.stds.to(device=x.device, dtype=x.dtype)
        probs = self.probs.to(device=x.device)

        component_idx = torch.distributions.Categorical(probs=probs).sample(leading_shape)
        noise = locs[component_idx] + stds[component_idx] * torch.randn(leading_shape, dtype=x.dtype, device=x.device)
        y = x[..., 0] * noise

        return y.unsqueeze(-1)


class TrajFlowMeszarosStochasticFunction(StochasticFunction):
    """Experiment-specific adapter exposing trajflow_meszaros as StochasticFunction."""

    def __init__(self, model=None) -> None:
        if model is None:
            from TrajFlowExperiment.trajflow_meszaros import trajflow_meszaros
            model = trajflow_meszaros()

        self.model = model
        self.dt = self.model.dt
        self.d_ctrl_max = torch.tensor([2.0, 0.05], device=self.model.device)  # max acceleration and curvature control

    def pos_to_ctrl(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.to(device=self.model.device)
        # assume x.shape = [..., N]
        # Get the batch dimensions
        batch_dims = x.shape[:-1]

        # Flatten the batch dimensions for processing
        x_flat = x.view(-1, x.shape[-1])

        # We need to assure that d can be reshaped to (-1, self.num_timesteps_in, 2)
        assert np.mod(x_flat.shape[-1], self.model.num_timesteps_in * 2) == 0, "The last dimension of x must be divisible by num_timesteps_in * 2"
        # get proper shape for the model input
        X = x_flat.view(len(x_flat), -1, self.model.num_timesteps_in, 2)
        X_tar = X[:, :1] # Agent to be predicted
        X_oth = X[:, 1:] # Other agent(s)
        X_0 = X_tar[:, :, 0, :]
        U, V_0, Phi_0 = Control_action.Inverse_Dynamical_Model(X_tar, dt=self.dt, device=x.device)
        # U.shape is (batch size, number agents, number time_steps - 1, 2)
        # V_0.shape is (batch size, number agents)
        # Phi_0.shape is (batch size, number agents)

        # Apply ctrl_max
        U = U / self.d_ctrl_max.view(1, 1, 1, -1)  # Normalize by max control

        # Transform U shape back to original batch dimensions
        u = U.view(*batch_dims, -1)
        x_0 = X_0.view(*batch_dims, -1)
        v_0 = V_0.view(*batch_dims, -1)
        phi_0 = Phi_0.view(*batch_dims, -1)
        x_oth = X_oth.view(*batch_dims, -1)
        return u, [x_0, v_0, phi_0, x_oth]

    def ctrl_to_pos(self, u: torch.Tensor, start: list[torch.Tensor]) -> torch.Tensor:
        x_0, v_0, phi_0, x_oth = start
        # u.shape is (..., M)
        batch_dims = u.shape[:-1]
        u_flat = u.view(-1, u.shape[-1])
        U = u_flat.view(len(u_flat), 1, self.model.num_timesteps_in - 1, 2)

        # Reverse normalization
        U = U * self.d_ctrl_max.view(1, 1, 1, -1)

        X_0 = x_0.view(-1, 1, 2)
        V_0 = v_0.view(-1, 1)
        Phi_0 = phi_0.view(-1, 1)
        X_oth = x_oth.view(len(u_flat), -1, self.model.num_timesteps_in, 2)

        X_tar = Control_action.Dynamical_Model(U, X_0, V_0, Phi_0, dt=self.dt, device=u.device)
        # Transform back to pass as model inputs
        X = torch.cat([X_tar, X_oth], dim=1)  # Shape: (batch size, number agents, number time steps, 2)
        x = X.view(*batch_dims, -1)
        return x
    

    def __call__(self, x: torch.Tensor, h: list[torch.Tensor] = []) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")
        
        # if h is not empty, assume that x are control inputs and we need to transform to positions
        if len(h) > 0:
            x = self.ctrl_to_pos(x, start=h)

        model_device = getattr(self.model, "device", x.device)
        y = self.model(x.to(model_device))
        return y.to(device=x.device, dtype=x.dtype)
