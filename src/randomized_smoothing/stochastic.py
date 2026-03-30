from abc import ABC, abstractmethod
import torch


class StochasticFunction(ABC):
    @abstractmethod
    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Generate stochastic outputs for each point in x.

        Args:
            x: Input tensor with shape (m, d), representing m points in d-dimensional space.
            num_samples: Number of samples to generate per input.

        Returns:
            Tensor with shape (m, num_samples, q).
        """
        raise NotImplementedError
