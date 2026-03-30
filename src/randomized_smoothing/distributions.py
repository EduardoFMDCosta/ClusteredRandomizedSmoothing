import torch
from torch.distributions import MultivariateNormal


class MultivariateGaussian:
    def __init__(self, dimension: int, noise_var: float):
        mean = torch.zeros(dimension)
        cov = noise_var * torch.eye(dimension)

        self._dist = MultivariateNormal(loc=mean, covariance_matrix=cov)

    def sample(self, size: tuple[int, int]) -> torch.Tensor:
        return self._dist.sample(size)