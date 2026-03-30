from abc import ABC, abstractmethod
import torch

from randomized_smoothing.distributions import Dirac, Distribution, MultivariateGaussian
from randomized_smoothing.stochastic import StochasticFunction
from randomized_smoothing.utils import _validate_alpha
from .clusters import Cluster


class Smoothing(ABC):
    @abstractmethod
    def smooth(self) -> Distribution:
        raise NotImplementedError
    @abstractmethod
    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        outputs = list()
        for _ in range(num_simulations):
            distribution = self.smooth()
            outputs.append(distribution.sample())
        return torch.stack(outputs)


class AlphaTrimmingSmoothing(Smoothing):
    """Alpha-trimming mean smoother on stochastic outputs.

    See Rekavandi et al. (2024), Neurips 2024, Equation (8).
    """

    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            x: torch.Tensor, 
            N: int, 
            alpha: float,
            noise_var: float
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = N
        self.alpha = _validate_alpha(alpha)
        self.noise_distribution = MultivariateGaussian(dimension=x.shape[-1], noise_var=noise_var)

    def smooth(self) -> Distribution:
        noise = self.noise_distribution.sample(size=(self.x.shape[0], self.N))
        outputs = self.stochastic_function.sample(self.x.unsqueeze(1) + noise)

        tail = int(self.alpha * self.N)
        sorted_outputs, _ = torch.sort(outputs, dim=1)
        trimmed_core = sorted_outputs[:, tail : self.N - tail, :]
        return Dirac(loc=trimmed_core.mean(dim=1))



class ClusterizedAlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            x: torch.Tensor, 
            N: int, 
            alpha: float,
            noise_var: float,
            cluster: Cluster
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = N
        self.alpha = _validate_alpha(alpha)
        self.cluster = cluster
        self.noise_distribution = MultivariateGaussian(dimension=x.shape[-1], noise_var=noise_var)

    def smooth(self) -> Distribution:
        noise = self.noise_distribution.sample(size=(self.x.shape[0], self.N))
        outputs = self.stochastic_function.sample(self.x.unsqueeze(1) + noise)

        centroids = self.cluster.centroids
        weights = self.cluster.weights

        # TODO: 

