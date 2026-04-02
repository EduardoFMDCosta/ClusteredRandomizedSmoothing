from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from .clopper_pearson import ClopperPearsonLowerBound
from .distributions import Dirac, Distribution, MixtureDirac, MultivariateGaussian
from .stochastic import StochasticFunction
from .utils import (
    _alpha_trimming_lower_bound,
    _clusterized_alpha_trimming_lower_bound,
    _validate_alpha,
    _validate_cluster,
    _compute_probability_given_radius,
)
from .clusters import Cluster


@dataclass
class SmoothingParameters:
    N: int
    alpha: float
    smoothing_distribution: Distribution

@dataclass
class RobustnessCertificate:
    certified_radius: float
    certified_probs: torch.Tensor


class Smoothing(ABC):
    @abstractmethod
    def smooth(self) -> Distribution:
        raise NotImplementedError
    
    @abstractmethod
    def robust_certification(self, region_coverage: float, beta: float, num_samples: int, radius: float) -> RobustnessCertificate:
        raise NotImplementedError

    def monte_carlo(self, x: torch.Tensor, num_simulations: int) -> torch.Tensor:
        draws = [self.smooth(x=x).sample() for _ in range(num_simulations)]
        return torch.stack(draws, dim=1)


class AlphaTrimmingSmoothing(Smoothing):
    """Alpha-trimming mean smoother on stochastic outputs.

    See Rekavandi et al. (2024), Neurips 2024, Equation (8).
    """

    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            params: SmoothingParameters
        ):
        self.stochastic_function = stochastic_function
        self.N = params.N
        self.alpha = _validate_alpha(params.alpha)
        self.smoothing_distribution = params.smoothing_distribution

    def smooth(self, x: torch.Tensor) -> Distribution:
        nx = x.shape[0]
        noise = self.smoothing_distribution.sample(size=(nx, self.N))
        outputs = self.stochastic_function.sample(x.unsqueeze(1) + noise)

        tail = int(self.alpha * self.N)
        sorted_outputs, _ = torch.sort(outputs, dim=1)
        trimmed_core = sorted_outputs[:, tail : self.N - tail, :]
        mean_trimmed_core = trimmed_core.mean(dim=1)
        return Dirac(loc=mean_trimmed_core)
    
    def robust_certification(
            self, 
            region_coverage: float, 
            beta: float, 
            num_samples: int, 
            radius: float,) -> RobustnessCertificate:
        if not isinstance(self.smoothing_distribution, MultivariateGaussian):
            raise ValueError("Robustness certification is only available for MultivariateGaussian")
        
        p = torch.tensor([[region_coverage]])
        clopper_pearson = ClopperPearsonLowerBound(p=p, beta=beta, num_samples=num_samples)
        p_for_radius = _compute_probability_given_radius(
            p=clopper_pearson.lower_bound, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )
        
        certified_probs = _alpha_trimming_lower_bound(
            p=p_for_radius, 
            num_samples=num_samples, 
            alpha=self.alpha
        )

        return RobustnessCertificate(
            certified_radius=radius,
            certified_probs=certified_probs
        )



class ClusterizedAlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            params: SmoothingParameters,
            cluster: Cluster
        ):
        self.stochastic_function = stochastic_function
        self.N = params.N
        self.alpha = _validate_alpha(params.alpha)
        self.cluster = _validate_cluster(cluster)
        self.smoothing_distribution = params.smoothing_distribution

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
    

    def robust_certification(self, region_coverage: float, beta: float, num_samples: int, radius: float,) -> RobustnessCertificate:
        if not isinstance(self.smoothing_distribution, MultivariateGaussian):
            raise ValueError("Robustness certification is only available for MultivariateGaussian")
        
        # TODO: FIX
        
        clopper_pearson = ClopperPearsonLowerBound(p=p, beta=beta, num_samples=num_samples, cluster=self.cluster)
        p_for_radius = _compute_probability_given_radius(
            p=clopper_pearson.lower_bound, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )

        certified_probs = _clusterized_alpha_trimming_lower_bound(p=p_for_radius, num_samples=num_samples, alpha=self.alpha, weights=self.cluster.weights)

        return RobustnessCertificate(
            certified_radius=radius,
            certified_probs=certified_probs
        )

