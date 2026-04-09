from abc import ABC, abstractmethod
import torch


class StochasticFunction(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Generate stochastic outputs for each point in x.

        Args:
            x: Input tensor with shape (..., d), where the leading dimensions are
                arbitrary batch dimensions.

        Returns:
            Tensor with shape (..., q).
        """
        raise NotImplementedError
    
    def sample_from_noisy_inputs(self, x: torch.Tensor, smoothing_distribution: torch.distributions.Distribution, num_samples: int) -> torch.Tensor:
        return self(
            x.unsqueeze(1) + smoothing_distribution.sample(size=(x.shape[0], num_samples))
        )


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
