from dataclasses import dataclass
from typing import Sequence, Tuple
import torch

from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.stochastic import StochasticFunction
from randomized_smoothing.utils import _trim_core_per_dimension, _validate_alpha
from .clusters import Cluster


class AlphaTrimmingSmoothing:
    """Alpha-trimming mean smoother on stochastic outputs.

    See Rekavandi et al. (2024), Neurips 2024, Equation (8).
    """

    def __init__(self, 
                 stochastic_function: StochasticFunction, 
                 x: torch.Tensor, 
                 N: int, 
                 alpha: float,
                 noise_var: float):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = N
        self.alpha = _validate_alpha(alpha)
        self.noise_distribution = MultivariateGaussian(dimension=x.shape[-1], noise_var=noise_var)

    def smooth(self) -> torch.Tensor:

        outputs = self.stochastic_function.sample(self.x, num_samples=self.N)

        m, _, _ = outputs.shape
        smoothed = []
        for i in range(m):
            trimmed_core = _trim_core_per_dimension(outputs[i], alpha=self.alpha)
            smoothed.append(trimmed_core.mean(dim=0))
        return torch.stack(smoothed, dim=0)

    def sample(self, num_samples: int) -> torch.Tensor:
        pass



class ClusterizedAlphaTrimmingSmoothing:
    def __init__(self, stochastic_function: StochasticFunction, x: torch.Tensor, N: int, alpha: float, cluster: Cluster):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = N
        self.alpha = _validate_alpha(alpha)
        self.cluster = cluster

    def smooth(self) -> torch.Tensor:
        outputs = self.stochastic_function.sample(self.x, num_samples=self.N)

        centroids = self.cluster.centroids
        weights = self.cluster.weights

        # TODO: 

    def sample(self, num_samples: int) -> torch.Tensor:
        pass
