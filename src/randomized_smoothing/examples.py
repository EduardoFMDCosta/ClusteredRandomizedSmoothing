import torch
from .stochastic import StochasticFunction


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

    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        m = x.shape[0]
        num_components = self.locs.shape[0]

        component_idx = torch.multinomial(
            self.probs.expand(m, num_components),
            num_samples=num_samples,
            replacement=True,
        )

        component_means = x[:, 0].unsqueeze(1) + self.locs[component_idx]
        noise = self.std * torch.randn((m, num_samples))
        y = component_means + noise

        # q = 1, so we add a final singleton dimension.
        return y.unsqueeze(-1)
