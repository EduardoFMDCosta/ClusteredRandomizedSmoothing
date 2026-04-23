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
    
    def sample_from_noisy_inputs(
            self, 
            x: torch.Tensor, 
            smoothing_distribution: 
            torch.distributions.Distribution, 
            num_samples: int,
            h: list[torch.Tensor] = [],
        ) -> torch.Tensor:

        noise = smoothing_distribution.sample(size=(x.shape[0], num_samples))
        x_new = x.unsqueeze(1) + noise.to(x.device, dtype=x.dtype)  # shape (batch_size, num_samples, d)
        # Check if self() requires h as an argument
        if 'h' in self.__call__.__code__.co_varnames:
            H = []
            # Reshape helper varaibles in h with the same logic as x_new
            for h_i in h:
                h_i_new = h_i.unsqueeze(1).repeat_interleave(num_samples, dim=1)
                H.append(h_i_new)
            return self(x_new, h=H)
        else:
            return self(x_new)