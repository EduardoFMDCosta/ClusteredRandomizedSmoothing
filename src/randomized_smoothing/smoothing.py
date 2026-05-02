from abc import ABC, abstractmethod
import math
import torch

from randomized_smoothing.dataclasses import Parameters, RobustnessCertificate
from randomized_smoothing.optimization import compute_total_prob_lower_bound, robustness_bound

from .clopper_pearson import ClopperPearsonInterval
from .distributions import Distribution, MixtureDirac, MultivariateGaussian
from .stochastic import StochasticFunction
from .clusters import Cluster
    

class Smoothing(ABC):
    @abstractmethod
    def smooth(self) -> Distribution:
        raise NotImplementedError

    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        first_draw = self.smooth().sample(size=())
        draws = torch.empty(
            (first_draw.shape[0], num_simulations, *first_draw.shape[1:]),
            dtype=first_draw.dtype,
            device=first_draw.device,
        )
        draws[:, 0, ...] = first_draw
        for i in range(1, num_simulations):
            draws[:, i, ...] = self.smooth().sample(size=())
        return draws


class AlphaTrimmingSmoothing(Smoothing):
    def __init__(
            self, 
            stochastic_function: StochasticFunction,
            x: torch.Tensor,
            smoothing_distribution: Distribution,
            parameters: Parameters,
            h: list[torch.Tensor] = [],
            cluster_anchor: torch.Tensor | None = None,
        ):
        self.stochastic_function = stochastic_function
        self.x = x
        self.h = h
        self.smoothing_distribution = smoothing_distribution
        self.parameters = parameters
        self.cluster_anchor = cluster_anchor

        self.cluster = self._compute_cluster()
        self.regions = self.cluster.regions 

    def _compute_cluster(self) -> Cluster:
        raw_outputs = self.stochastic_function.sample_from_noisy_inputs(
            x=self.x,
            smoothing_distribution=self.smoothing_distribution,
            num_samples=self.parameters.confidence.num_samples,
            h=self.h,
            noise_sub_batch_size=self.parameters.runtime.noise_sub_batch_size,
        )
        return Cluster(
            outputs=raw_outputs,
            parameters=self.parameters.clustering,
            behavior_anchor=self.cluster_anchor,
        )

    def smooth(self) -> Distribution:
        outputs = self.stochastic_function.sample_from_noisy_inputs(
            x=self.x,
            smoothing_distribution=self.smoothing_distribution,
            num_samples=self.parameters.smoothing.N,
            h=self.h,
            noise_sub_batch_size=self.parameters.runtime.noise_sub_batch_size,
        )
        
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

    def monte_carlo(self, num_simulations: int) -> torch.Tensor:
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        nx = self.x.shape[0]
        num_samples = self.parameters.smoothing.N
        total_samples = num_samples * num_simulations

        outputs = self.stochastic_function.sample_from_noisy_inputs(
            x=self.x,
            smoothing_distribution=self.smoothing_distribution,
            num_samples=total_samples,
            h=self.h,
            noise_sub_batch_size=self.parameters.runtime.noise_sub_batch_size,
        )

        output_dim = outputs.shape[-1]
        outputs = outputs.view(nx, num_simulations, num_samples, output_dim)

        cluster = self.cluster
        centroids = cluster.centroids
        probs_partition = cluster.probs_partition
        num_clusters = cluster.num_clusters

        diff = outputs.unsqueeze(3) - centroids.unsqueeze(1).unsqueeze(1)
        squared_distances = (diff * diff).sum(dim=-1)
        assignments = squared_distances.argmin(dim=3)

        trimmed_mean = torch.zeros(
            (nx, num_simulations, num_clusters, output_dim),
            dtype=outputs.dtype,
            device=outputs.device,
        )
        positions = torch.arange(num_samples, device=outputs.device).view(1, 1, num_samples, 1)
        inf = torch.tensor(torch.inf, device=outputs.device, dtype=outputs.dtype)
        zero = torch.zeros(1, device=outputs.device, dtype=outputs.dtype)

        for j in range(num_clusters):
            cluster_mask = assignments == j
            cluster_values = torch.where(cluster_mask.unsqueeze(-1), outputs, inf)
            sorted_cluster_values = cluster_values.sort(dim=2).values

            counts = cluster_mask.sum(dim=2)
            left = torch.floor(self.parameters.smoothing.alpha * counts.to(outputs.dtype)).to(torch.long)
            right = counts - left

            in_trimmed_core = (positions >= left.view(nx, num_simulations, 1, 1)) & (
                positions < right.view(nx, num_simulations, 1, 1)
            )
            trimmed_sum = torch.where(in_trimmed_core, sorted_cluster_values, zero).sum(dim=2)
            trimmed_count = (right - left).to(outputs.dtype).view(nx, num_simulations, 1)

            cluster_trimmed_mean = trimmed_sum / trimmed_count.clamp(min=1)
            has_trimmed_points = trimmed_count.squeeze(-1) > 0
            trimmed_mean[:, :, j, :] = torch.where(
                has_trimmed_points.unsqueeze(-1),
                cluster_trimmed_mean,
                centroids[:, j, :].unsqueeze(1),
            )

        sampled_cluster = torch.multinomial(probs_partition, num_samples=num_simulations, replacement=True)
        gather_idx = sampled_cluster.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, output_dim)

        return trimmed_mean.gather(dim=2, index=gather_idx).squeeze(2)


class GaussianAlphaTrimmingSmoothing(AlphaTrimmingSmoothing):
    def __init__(
            self,
            stochastic_function: StochasticFunction,
            x: torch.Tensor,
            sigma: float,
            parameters: Parameters,
            h: list[torch.Tensor] = [],
            cluster_anchor: torch.Tensor | None = None,
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
            h=h,
            cluster_anchor=cluster_anchor,
        )

    @staticmethod
    def _compute_probability_given_radius(p: torch.Tensor, radius: float | torch.Tensor, sigma: float, is_lb: bool) -> torch.Tensor:
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device=p.device, dtype=p.dtype)
        
        z = torch.special.ndtri(p)
        shifted = z - radius / sigma if is_lb else z + radius / sigma
        probs = 0.5 * (1.0 + torch.erf(shifted / math.sqrt(2.0)))

        return probs
    

    def _compute_bound(
        self,
        probs_region: torch.Tensor,
        probs_partition_lb: torch.Tensor,
        probs_partition_ub: torch.Tensor,
        num_samples: int,
        alpha: float,
    ) -> torch.Tensor:
        return robustness_bound(
            probs_region=probs_region,
            probs_partition_lb=probs_partition_lb,
            probs_partition_ub=probs_partition_ub,
            num_samples=num_samples,
            alpha=alpha,
        )


    def _compute_total_bound(
        self,
        probs_region: torch.Tensor,
        probs_partition_lb: torch.Tensor,
        probs_partition_ub: torch.Tensor,
        num_samples: int,
        alpha: float,
    ) -> torch.Tensor:
        probs_region_np = probs_region.detach().cpu().to(torch.float64).numpy()
        probs_partition_lb_np = probs_partition_lb.detach().cpu().to(torch.float64).numpy()
        probs_partition_ub_np = probs_partition_ub.detach().cpu().to(torch.float64).numpy()

        total_bounds: list[float] = []
        for p_R, p_lower, p_upper in zip(probs_region_np, probs_partition_lb_np, probs_partition_ub_np):
            lower_bound, _, _ = compute_total_prob_lower_bound(
                M=probs_region.shape[1],
                N=num_samples,
                alpha=alpha,
                p_R=p_R,
                p_V_lower=p_lower,
                p_V_upper=p_upper,
            )
            total_bounds.append(lower_bound)

        return torch.tensor(total_bounds, device=probs_region.device, dtype=probs_region.dtype)


    def robust_certification(self, radius: torch.Tensor,) -> RobustnessCertificate:
        # Compute Clopper-Pearson lower bound
        cp = ClopperPearsonInterval(
            probs_region=self.cluster.probs_region,
            probs_partition=self.cluster.probs_partition,
            parameters=self.parameters.confidence,
        )
        probs_region = cp.region.lb
        probs_partition_lb = cp.partition.lb
        probs_partition_ub = cp.partition.ub

        # Adjust probabilities for the given radius
        probs_region = self._compute_probability_given_radius(
            p=probs_region,
            radius=radius, 
            sigma=self.smoothing_distribution.sigma,
            is_lb=True,
        )

        probs_partition_lb = self._compute_probability_given_radius(
            p=probs_partition_lb,
            radius=radius,
            sigma=self.smoothing_distribution.sigma,
            is_lb=True,
        )

        probs_partition_ub = self._compute_probability_given_radius(
            p=probs_partition_ub,
            radius=radius,
            sigma=self.smoothing_distribution.sigma,
            is_lb=False,
        )

        # Certify
        certified_probs_per_cluster = self._compute_bound(
            probs_region=probs_region,
            probs_partition_lb=probs_partition_lb,
            probs_partition_ub=probs_partition_ub,
            num_samples=self.parameters.smoothing.N,
            alpha=self.parameters.smoothing.alpha,
        )

        total_probs = self._compute_total_bound(
            probs_region=probs_region,
            probs_partition_lb=probs_partition_lb,
            probs_partition_ub=probs_partition_ub,
            num_samples=self.parameters.smoothing.N,
            alpha=self.parameters.smoothing.alpha,
        )

        return RobustnessCertificate(
            radius=radius,
            probs=certified_probs_per_cluster,
            total_probs=total_probs,
        )