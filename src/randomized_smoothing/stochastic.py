from abc import ABC, abstractmethod
import torch


class StochasticFunction(ABC):
    @abstractmethod
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Generate stochastic outputs for each point in x.

        Args:
            x: Input tensor with shape (..., d), where the leading dimensions are
                arbitrary batch dimensions.

        Returns:
            Tensor with shape (..., q).
        """
        raise NotImplementedError
