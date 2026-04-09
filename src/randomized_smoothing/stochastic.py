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