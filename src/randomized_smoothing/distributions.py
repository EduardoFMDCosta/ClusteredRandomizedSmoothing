from abc import ABC, abstractmethod
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

    def sample(self) -> torch.Tensor:
        return self.loc.clone()


class MixtureDirac(Distribution):
    def __init__(self, locs: torch.Tensor, probs: torch.Tensor):
        self.locs = locs
        self.probs = probs

    def sample(self) -> torch.Tensor:
        nx, _, dimension = self.locs.shape
        indices = torch.multinomial(self.probs, num_samples=1, replacement=True).squeeze(1)
        gather_idx = indices.view(nx, 1, 1).expand(-1, 1, dimension)
        return self.locs.gather(dim=1, index=gather_idx).squeeze(1)