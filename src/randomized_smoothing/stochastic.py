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
            smoothing_distribution: torch.distributions.Distribution,
            num_samples: int,
            h: list[torch.Tensor] = [],
            noise_sub_batch_size: int | None = None,
        ) -> torch.Tensor:

        if noise_sub_batch_size is not None and noise_sub_batch_size <= 0:
            raise ValueError("noise_sub_batch_size must be positive when provided")

        noise = smoothing_distribution.sample(size=(x.shape[0], num_samples))
        x_new = x.unsqueeze(1) + noise.to(x.device, dtype=x.dtype)  # shape (batch_size, num_samples, d)

        has_h = 'h' in self.__call__.__code__.co_varnames

        if noise_sub_batch_size is None or noise_sub_batch_size >= num_samples:
            if has_h:
                H = [h_i.unsqueeze(1).repeat_interleave(num_samples, dim=1) for h_i in h]
                return self(x_new, h=H)
            else:
                return self(x_new)

        # Sub-batch over the noise/samples dimension to limit peak memory.
        chunks = []
        for start in range(0, num_samples, noise_sub_batch_size):
            end = min(start + noise_sub_batch_size, num_samples)
            x_chunk = x_new[:, start:end, :]
            if has_h:
                chunk_size = end - start
                H = [h_i.unsqueeze(1).repeat_interleave(chunk_size, dim=1) for h_i in h]
                chunks.append(self(x_chunk, h=H))
            else:
                chunks.append(self(x_chunk))
        return torch.cat(chunks, dim=1)