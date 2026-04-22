from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.stochastic import StochasticFunction


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")

        model_device = getattr(self.model, "device", x.device)
        y = self.model(x.to(model_device))
        return y.to(device=x.device, dtype=x.dtype)
