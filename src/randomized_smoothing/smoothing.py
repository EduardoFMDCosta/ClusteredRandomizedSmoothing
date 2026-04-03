from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch

from randomized_smoothing.dataclasses import Parameters, RobustnessCertificate
from randomized_smoothing.regions import HyperRectangle

from .clopper_pearson import ClopperPearsonLowerBound
from .distributions import Dirac, Distribution, MixtureDirac, MultivariateGaussian
from .stochastic import StochasticFunction
from .utils import (
    _alpha_trimming_lower_bound,
    _clusterized_alpha_trimming_lower_bound,
    _validate_alpha,
    _compute_probability_given_radius,
)
from .clusters import Cluster


@dataclass
class RobustSmoothedOutput:
    distribution: Distribution
    certified_probs: Optional[torch.Tensor]
    radii: Optional[torch.Tensor]
    

class Smoothing(ABC):
    @abstractmethod
    def smooth(self, x: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        first_draw = self.smooth().distribution.sample()
        draws = torch.empty(
            (first_draw.shape[0], num_simulations, *first_draw.shape[1:]),
            dtype=first_draw.dtype,
            device=first_draw.device,
        )
        draws[:, 0, ...] = first_draw
        for i in range(1, num_simulations):
            draws[:, i, ...] = self.smooth().distribution.sample()
        return draws


class AlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction,
            x: torch.Tensor,
            smoothing_distribution: Distribution,
            parameters: Parameters,
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.smoothing_distribution = smoothing_distribution
        
        if _validate_alpha(parameters.smoothing.alpha):
            self.parameters = parameters

        self.cluster = self._compute_cluster()
        self.regions = self.cluster.regions

    def _generate_noisy_samples_from_stochastic_function(self, num_samples: int) -> torch.Tensor:
        nx = self.x.shape[0]
        return self.stochastic_function.sample(
            self.x.unsqueeze(1) + self.smoothing_distribution.sample(size=(nx, num_samples))
        )  

    def _compute_cluster(self) -> Cluster:
        raw_outputs = self._generate_noisy_samples_from_stochastic_function(num_samples=self.parameters.confidence.num_samples)
        return Cluster(outputs=raw_outputs, parameters=self.parameters.clustering)

    def smooth(self) -> Distribution:
        outputs = self._generate_noisy_samples_from_stochastic_function(num_samples=self.parameters.smoothing.N)
        
        cluster = self.cluster
        centroids = cluster.centroids
        weights = cluster.weights

        nx, num_samples, _ = outputs.shape
        num_clusters = cluster.num_clusters

        diff = outputs.unsqueeze(2) - centroids.unsqueeze(1)
        squared_distances = (diff * diff).sum(dim=-1)
        assignments = squared_distances.argmin(dim=2)

        trimmed_mean = torch.zeros_like(centroids)
        positions = torch.arange(num_samples, device=outputs.device).view(1, num_samples, 1)
        inf = torch.tensor(torch.inf, device=outputs.device, dtype=outputs.dtype)
        zero = torch.zeros(1, device=outputs.device, dtype=outputs.dtype)

        for j in range(num_clusters):
            cluster_mask = assignments == j
            cluster_values = torch.where(cluster_mask.unsqueeze(-1), outputs, inf)
            sorted_cluster_values = cluster_values.sort(dim=1).values

            counts = cluster_mask.sum(dim=1)
            left = torch.floor(self.parameters.smoothing.alpha * counts.to(outputs.dtype)).to(torch.long)
            right = counts - left

            in_trimmed_core = (positions >= left.view(nx, 1, 1)) & (positions < right.view(nx, 1, 1))
            trimmed_sum = torch.where(in_trimmed_core, sorted_cluster_values, zero).sum(dim=1)
            trimmed_count = (right - left).to(outputs.dtype).view(nx, 1)

            cluster_trimmed_mean = trimmed_sum / trimmed_count.clamp(min=1)
            has_trimmed_points = trimmed_count.squeeze(-1) > 0
            trimmed_mean[:, j, :] = torch.where(
                has_trimmed_points.unsqueeze(-1),
                cluster_trimmed_mean,
                centroids[:, j, :],
            )

        distribution = MixtureDirac(locs=trimmed_mean, probs=weights)
        return RobustSmoothedOutput(
            distribution=distribution,
            certified_probs=None, 
            radii=None)


class GaussianAlphaTrimmingSmoothing(AlphaTrimmingSmoothing):
    def __init__(
            self,
            stochastic_function: StochasticFunction,
            x: torch.Tensor,
            sigma: float,
            parameters: Parameters,
        ):
        dimension = x.shape[-1]
        mean = torch.zeros(dimension, device=x.device, dtype=x.dtype)
        cov = (sigma ** 2) * torch.eye(dimension, device=x.device, dtype=x.dtype)
        smoothing_distribution = MultivariateGaussian(mean=mean, cov=cov)

        super().__init__(
            stochastic_function=stochastic_function,
            x=x,
            smoothing_distribution=smoothing_distribution,
            parameters=parameters,
        )

    
    def _clopper_pearson_lower_bound(self) -> torch.Tensor:
        return ClopperPearsonLowerBound(
            coverage=self.parameters.clustering.coverage,
            partition_probs=self.cluster.weights,
            parameters=self.parameters.confidence,
        ).lower_bound

    def robust_certification(self, radius: float,) -> RobustnessCertificate:
        if not isinstance(self.smoothing_distribution, MultivariateGaussian):
            raise ValueError("Robustness certification is only available for MultivariateGaussian")
        
        lb = self._clopper_pearson_lower_bound()
        p_for_radius = _compute_probability_given_radius(
            p=lb, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )

        certified_probs = _clusterized_alpha_trimming_lower_bound(p=p_for_radius, num_samples=self.parameters.smoothing.num_samples, alpha=self.parameters.smoothing.alpha, weights=self.cluster.weights)

        return RobustnessCertificate(
            certified_radius=radius,
            certified_probs=certified_probs
        )