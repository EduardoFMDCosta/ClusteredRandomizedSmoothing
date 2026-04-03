from abc import ABC, abstractmethod
import math
import torch

from randomized_smoothing.dataclasses import Parameters, RobustnessCertificate

from .clopper_pearson import ClopperPearsonLowerBound
from .distributions import Distribution, MixtureDirac, MultivariateGaussian
from .stochastic import StochasticFunction
from .utils import (
    _validate_alpha,
    _binomial_cumulative_probability,
)
from .clusters import Cluster
    

class Smoothing(ABC):
    @abstractmethod
    def smooth(self) -> Distribution:
        raise NotImplementedError

    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        first_draw = self.smooth().sample()
        draws = torch.empty(
            (first_draw.shape[0], num_simulations, *first_draw.shape[1:]),
            dtype=first_draw.dtype,
            device=first_draw.device,
        )
        draws[:, 0, ...] = first_draw
        for i in range(1, num_simulations):
            draws[:, i, ...] = self.smooth().sample()
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
        probs_partition = cluster.probs_partition

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

        return MixtureDirac(locs=trimmed_mean, probs=probs_partition)


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

    def _compute_probability_given_radius(p: torch.Tensor, radius: float, sigma: float) -> torch.Tensor:
        z = torch.special.ndtri(p)
        shifted = z - radius / sigma
        probs = 0.5 * (1.0 + torch.erf(shifted / math.sqrt(2.0)))

        return probs
    
    
    
    def _alpha_trimming_lower_bound(p: float, num_samples: int, alpha: float) -> float:
        alphaN = int(alpha * num_samples)
        return _binomial_cumulative_probability(from_idx=0, to_idx=alphaN, p=p)
    

    def _compute_per_region_factor(p: float, p_clusters: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
        device = p_clusters.device
        dtype = torch.float64

        p_region = torch.tensor(float(p), dtype=dtype, device=device)
        one = torch.tensor(1.0, dtype=dtype, device=device)

        p_v = p_clusters.to(dtype=dtype)

        log_p_region = torch.log(p_region)
        log_one_minus_p_region = torch.log(one - p_region)
        log_p_v = torch.log(p_v)
        log_one_minus_p_v = torch.log(one - p_v)

        result = torch.zeros_like(p_v)

        for ell in range(1, num_samples + 1):
            ell_t = torch.tensor(float(ell), dtype=dtype, device=device)

            inner_sum = torch.tensor(0.0, dtype=dtype, device=device)
            max_j = int(alpha * ell)
            for j in range(0, max_j + 1):
                j_t = torch.tensor(float(j), dtype=dtype, device=device)

                log_choose_ell_j = (
                    torch.lgamma(ell_t + 1.0)
                    - torch.lgamma(j_t + 1.0)
                    - torch.lgamma(ell_t - j_t + 1.0)
                )
                log_term_j = (
                    log_choose_ell_j
                    + j_t * log_one_minus_p_region
                    + (ell_t - j_t) * log_p_region
                )
                inner_sum = inner_sum + torch.exp(log_term_j)

            log_choose_n_ell = (
                torch.lgamma(torch.tensor(float(num_samples), dtype=dtype, device=device) + 1.0)
                - torch.lgamma(ell_t + 1.0)
                - torch.lgamma(torch.tensor(float(num_samples - ell), dtype=dtype, device=device) + 1.0)
            )

            cluster_term = torch.exp(
                log_choose_n_ell
                + ell_t * log_one_minus_p_v
                + (num_samples - ell_t) * log_p_v
            )
            result = result + inner_sum * cluster_term
        return result


    def bound(p: float, p_clusters: torch.Tensor, num_samples: int, alpha: float, probs_partition: torch.Tensor) -> torch.Tensor:
        result = _compute_per_region_factor(p=p, p_clusters=p_clusters, num_samples=num_samples, alpha=alpha)
        return (probs_partition * result).to(p_clusters.dtype)


    def robust_certification(self, radius: float,) -> RobustnessCertificate:
        cp = ClopperPearsonLowerBound(
            coverage=self.parameters.clustering.coverage,
            partition_probs=self.cluster.probs_partition,
            parameters=self.parameters.confidence,
        )

        lb = cp.lower_bound
        p_for_radius = _compute_probability_given_radius(
            p=lb, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )



        certified_probs = self.bound(
            p=p_for_radius, 
            num_samples=self.parameters.smoothing.num_samples, 
            alpha=self.parameters.smoothing.alpha, 
            probs_partition=self.cluster.probs_partition
        )

        return RobustnessCertificate(
            radius=radius,
            probs=certified_probs
        )