from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.stochastic import StochasticFunction


class ThreeMode1DGaussianMixture(StochasticFunction):
    def __init__(
        self,
        locs: torch.Tensor,
        probs: torch.Tensor,
        std: float,
    ) -> None:
        self.locs = locs
        self.probs = probs
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")

        leading_shape = x.shape[:-1]
        locs = self.locs.to(device=x.device, dtype=x.dtype)
        probs = self.probs.to(device=x.device)

        component_idx = torch.distributions.Categorical(probs=probs).sample(leading_shape)
        component_means = x[..., 0] + locs[component_idx]
        noise = self.std * torch.randn(leading_shape, dtype=x.dtype, device=x.device)
        y = component_means + noise

        return y.unsqueeze(-1)


class TrajFlowMeszarosStochasticFunction(StochasticFunction):
    """Experiment-specific adapter exposing trajflow_meszaros as StochasticFunction."""

    def __init__(self, model=None) -> None:
        if model is None:
            traj_dir = ROOT / "trajectory_prediction"
            if str(traj_dir) not in sys.path:
                sys.path.insert(0, str(traj_dir))

            module_path = traj_dir / "trajflow_meszaros.py"
            spec = importlib.util.spec_from_file_location("trajflow_meszaros", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module from {module_path}")

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "Failed to initialize TrajFlow adapter because an optional dependency is missing. "
                    "Install TrajFlow dependencies (for example torch-geometric) to use "
                    "TrajFlowMeszarosStochasticFunction."
                ) from exc
            model = module.trajflow_meszaros()

        self.model = model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., d)")

        model_device = getattr(self.model, "device", x.device)
        y = self.model(x.to(model_device))
        return y.to(device=x.device, dtype=x.dtype)
