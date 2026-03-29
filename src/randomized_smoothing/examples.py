from __future__ import annotations

import torch

from .stochastic import StochasticFunction


class ThreeMode1DGaussianMixture(StochasticFunction):
    """1D stochastic function with three separated Gaussian components.

    For each input x, this defines a 1D mixture with means
    x + offsets[k], fixed std, and mixing probabilities.
    """

    def __init__(
        self,
        offsets: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        std: float = 0.15,
    ) -> None:
        self.offsets = offsets if offsets is not None else torch.tensor([-4.0, 0.0, 4.0])
        self.probs = probs if probs is not None else torch.tensor([0.25, 0.5, 0.25])
        self.std = float(std)

        if self.offsets.ndim != 1:
            raise ValueError("offsets must be a 1D tensor")
        if self.probs.ndim != 1:
            raise ValueError("probs must be a 1D tensor")
        if self.offsets.shape[0] != self.probs.shape[0]:
            raise ValueError("offsets and probs must have the same length")

        probs_sum = self.probs.sum()
        if probs_sum <= 0:
            raise ValueError("probs must sum to a positive value")
        self.probs = self.probs / probs_sum

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape (m, d)")
        if x.shape[1] != 1:
            raise ValueError("This example expects 1D inputs, i.e. d=1")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        m = x.shape[0]
        num_components = self.offsets.shape[0]

        device = x.device
        dtype = x.dtype

        probs = self.probs.to(device=device, dtype=dtype)
        offsets = self.offsets.to(device=device, dtype=dtype)

        component_idx = torch.multinomial(
            probs.expand(m, num_components),
            num_samples=num_samples,
            replacement=True,
        )

        component_means = x[:, 0].unsqueeze(1) + offsets[component_idx]
        noise = self.std * torch.randn((m, num_samples), device=device, dtype=dtype)
        y = component_means + noise

        # q = 1, so we add a final singleton dimension.
        return y.unsqueeze(-1)
