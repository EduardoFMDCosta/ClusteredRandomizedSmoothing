from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import torch

from .clopper_pearson import ClopperPearsonLowerBound
from .distributions import Dirac, Distribution, MixtureDirac
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
    smoothing_distribution: Distribution
    N: int
    alpha: float
    cluster: Cluster = None

@dataclass
class RobustnessCertificate:
    certified_radius: float
    certified_probs: torch.Tensor


class Smoothing(ABC):
    @abstractmethod
    def smooth(self) -> Distribution:
        raise NotImplementedError
    
    @abstractmethod
    def robust_certification(self, p: float, radius: float) -> RobustnessCertificate:
        raise NotImplementedError

    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        draws = [self.smooth().sample() for _ in range(num_simulations)]
        return torch.stack(draws, dim=1)


class AlphaTrimmingSmoothing(Smoothing):
    """Alpha-trimming mean smoother on stochastic outputs.

    See Rekavandi et al. (2024), Neurips 2024, Equation (8).
    """

    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            x: torch.Tensor, 
            params: SmoothingParameters
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = params.N
        self.alpha = _validate_alpha(params.alpha)
        self.smoothing_distribution = params.smoothing_distribution

    def smooth(self) -> Distribution:
        noise = self.smoothing_distribution.sample(size=(self.x.shape[0], self.N))
        outputs = self.stochastic_function.sample(self.x.unsqueeze(1) + noise)

        tail = int(self.alpha * self.N)
        sorted_outputs, _ = torch.sort(outputs, dim=1)
        trimmed_core = sorted_outputs[:, tail : self.N - tail, :]
        return Dirac(loc=trimmed_core.mean(dim=1))
    
    def robust_certification(self, p: float, beta: float, num_samples: int, radius: float,) -> RobustnessCertificate:
        clopper_pearson = ClopperPearsonLowerBound(p=p, beta=beta, num_samples=num_samples)
        p_for_radius = _compute_probability_given_radius(
            p=clopper_pearson.lower_bound, 
            radius=radius, 
            sigma=self.smoothing_distribution.sigma
        )
        
        certified_probs = _alpha_trimming_lower_bound(p=p_for_radius, num_samples=num_samples, alpha=self.alpha)

        return RobustnessCertificate(
            certified_radius=radius,
            certified_probs=certified_probs
        )



class ClusterizedAlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction, 
            x: torch.Tensor, 
            params: SmoothingParameters
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.N = params.N
        self.alpha = _validate_alpha(params.alpha)
        self.cluster = _validate_cluster(params.cluster)
        self.smoothing_distribution = params.smoothing_distribution

    def smooth(self) -> Distribution:
        noise = self.smoothing_distribution.sample(size=(self.x.shape[0], self.N))
        outputs = self.stochastic_function.sample(self.x.unsqueeze(1) + noise)

        centroids = self.cluster.centroids
        weights = self.cluster.weights
        if outputs.ndim != 3 or centroids.ndim != 3 or weights.ndim != 2:
            raise ValueError("Expected outputs (m, N, q), centroids (m, k, q), and weights (m, k)")

        m, n, q = outputs.shape
        m_c, k, q_c = centroids.shape
        m_w, k_w = weights.shape
        if m != m_c or m != m_w or q != q_c or k != k_w:
            raise ValueError("Incompatible shapes between outputs, cluster centroids, and cluster weights")

        distances = torch.cdist(outputs, centroids)
        assignments = distances.argmin(dim=2)  # (m, N)

        membership = torch.nn.functional.one_hot(assignments, num_classes=k).to(dtype=torch.bool)
        membership_mkn = membership.permute(0, 2, 1)  # (m, k, N)

        values = outputs.unsqueeze(1).expand(-1, k, -1, -1)  # (m, k, N, q)
        inf = torch.tensor(torch.inf, device=outputs.device, dtype=outputs.dtype)
        masked_values = torch.where(membership_mkn.unsqueeze(-1), values, inf)
        sorted_values = masked_values.sort(dim=2).values

        counts = membership_mkn.sum(dim=2)  # (m, k)
        left = torch.floor(self.alpha * counts.to(outputs.dtype)).to(torch.long)
        right = counts - left

        positions = torch.arange(n, device=outputs.device).view(1, 1, n, 1)
        in_trimmed_core = (positions >= left.unsqueeze(-1).unsqueeze(-1)) & (
            positions < right.unsqueeze(-1).unsqueeze(-1)
        )

        trimmed_sum = torch.where(in_trimmed_core, sorted_values, torch.zeros(1, device=outputs.device, dtype=outputs.dtype)).sum(dim=2)
        trimmed_count = (right - left).to(outputs.dtype)
        trimmed_mean = trimmed_sum / trimmed_count.clamp(min=1).unsqueeze(-1)
        trimmed_mean = torch.where(trimmed_count.unsqueeze(-1) > 0, trimmed_mean, centroids)

        return MixtureDirac(locs=trimmed_mean, probs=weights)
    

    def robust_certification(self, p: float, beta: float, num_samples: int, radius: float,) -> RobustnessCertificate:
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

