from abc import ABC, abstractmethod
import math
import torch

from torch.distributions import MultivariateNormal


class Distribution(ABC):
    @abstractmethod
    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError


class MultivariateGaussian(Distribution):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.mean = mean
        self.cov = cov
        self._dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

    @property
    def sigma(self) -> float:
        return self.cov.diagonal().sqrt().max().item()

    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        return self._dist.sample(size)


class Dirac(Distribution):
    def __init__(self, loc: torch.Tensor):
        self.loc = loc

    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        if len(size) == 0:
            return self.loc.clone()

        expand_dims = (1,) * len(size)
        view_shape = (self.loc.shape[0], *expand_dims, *self.loc.shape[1:])
        target_shape = (self.loc.shape[0], *size, *self.loc.shape[1:])
        return self.loc.view(view_shape).expand(target_shape).clone()


class MixtureDirac(Distribution):
    def __init__(self, locs: torch.Tensor, probs: torch.Tensor):
        self.locs = locs
        self.probs = probs

    def sample(self, size: tuple[int, ...]) -> torch.Tensor:
        if len(size) == 0:
            nx, _, dimension = self.locs.shape
            indices = torch.multinomial(self.probs, num_samples=1, replacement=True).squeeze(1)
            gather_idx = indices.view(nx, 1, 1).expand(-1, 1, dimension)
            return self.locs.gather(dim=1, index=gather_idx).squeeze(1)

        nx, _, dimension = self.locs.shape
        flat_num_samples = math.prod(size)
        indices = torch.multinomial(self.probs, num_samples=flat_num_samples, replacement=True)
        indices = indices.view(nx, *size)

        batch_idx_shape = (nx, *([1] * len(size)))
        batch_idx = torch.arange(nx, device=self.locs.device).view(batch_idx_shape).expand(nx, *size)
        return self.locs[batch_idx, indices]