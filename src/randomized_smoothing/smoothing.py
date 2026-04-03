from abc import ABC, abstractmethod
import torch

from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, RobustnessCertificate, SmoothingParameters

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



class Smoothing(ABC):
    @abstractmethod
    def smooth(self, x: torch.Tensor) -> Distribution:
        raise NotImplementedError
    
    # @abstractmethod
    # def robust_certification(self, region_coverage: float, p_partition: torch.Tensor, beta: float, num_samples: int, radius: float) -> RobustnessCertificate:
    #     raise NotImplementedError

    def monte_carlo(self, x: torch.Tensor, num_simulations: int) -> torch.Tensor:
        draws = [self.smooth(x=x).sample() for _ in range(num_simulations)]
        return torch.stack(draws, dim=1)


class AlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction,
            smoothing_distribution: Distribution, 
            smoothing_params: SmoothingParameters,
            clustering_params: ClusteringParameters,
            confidence_params: ConfidenceParameters,
        ):
        self.stochastic_function = stochastic_function
        self.smoothing_distribution = smoothing_distribution
        
        if _validate_alpha(smoothing_params.alpha):
            self.smoothing_parameters = smoothing_params
        self.clustering_params = clustering_params
        self.confidence_params = confidence_params
        
        self.cluster = self._construct_cluster()

    def get_regions(self):
        return self.cluster.regions

    def _generate_noisy_samples_from_stochastic_function(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        nx = x.shape[0]
        return self.stochastic_function.sample(
            x.unsqueeze(1) + self.smoothing_distribution.sample(size=(nx, num_samples))
        )  

    def _construct_cluster(self, x: torch.Tensor, num_samples: int) -> Cluster:
        raw_outputs = self._generate_noisy_samples_from_stochastic_function(x=x, num_samples=num_samples)
        return Cluster(outputs=raw_outputs, params=self.smoothing_parameters,)

    def smooth(self, x: torch.Tensor) -> Distribution:
        noise = self.smoothing_distribution.sample(size=(x.shape[0], self.N))
        outputs = self.stochastic_function.sample(x.unsqueeze(1) + noise)

        centroids = self.cluster.centroids
        weights = self.cluster.weights

        nx, num_samples, _ = outputs.shape
        num_clusters = self.cluster.num_clusters

        distances = torch.cdist(outputs, centroids)
        assignments = distances.argmin(dim=2)

        trimmed_mean = torch.zeros_like(centroids)
        positions = torch.arange(num_samples, device=outputs.device).view(1, num_samples, 1)
        inf = torch.tensor(torch.inf, device=outputs.device, dtype=outputs.dtype)
        zero = torch.zeros(1, device=outputs.device, dtype=outputs.dtype)

        for j in range(num_clusters):
            cluster_mask = assignments == j
            cluster_values = torch.where(cluster_mask.unsqueeze(-1), outputs, inf)
            sorted_cluster_values = cluster_values.sort(dim=1).values

            counts = cluster_mask.sum(dim=1)
            left = torch.floor(self.alpha * counts.to(outputs.dtype)).to(torch.long)
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

        return MixtureDirac(locs=trimmed_mean, probs=weights)
    
    def _clopper_pearson_lower_bound(self) -> torch.Tensor:
        return ClopperPearsonLowerBound(
            cluster=self.cluster,
            clustering_parameters=self.clustering_params,
            confidence_params=self.confidence_params,
        ).lower_bound

    def robust_certification(self, x: torch.Tensor, radius: float,) -> RobustnessCertificate:
        if not isinstance(self.smoothing_distribution, MultivariateGaussian):
            raise ValueError("Robustness certification is only available for MultivariateGaussian")
        
        lb = self._clopper_pearson_lower_bound()
        p_for_radius = _compute_probability_given_radius(
            p=lb, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )

        certified_probs = _clusterized_alpha_trimming_lower_bound(p=p_for_radius, num_samples=self.smoothing_parameters.num_samples, alpha=self.alpha, weights=self.cluster.weights)

        return RobustnessCertificate(
            certified_radius=radius,
            certified_probs=certified_probs
        )

