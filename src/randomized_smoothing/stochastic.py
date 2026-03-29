from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class StochasticFunction(ABC):
    """Base interface for stochastic functions.

    A stochastic function takes a batch of inputs of shape (m, d) and returns
    sampled outputs of shape (m, num_samples, q).
    """

    @abstractmethod
    def sample(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Generate stochastic outputs for each point in x.

        Args:
            x: Input tensor with shape (m, d).
            num_samples: Number of samples to generate per input.

        Returns:
            Tensor with shape (m, num_samples, q).
        """
        raise NotImplementedError
