import torch
from dataclasses import dataclass

from randomized_smoothing.dataclasses import ConfidenceParameters
from .utils import _binomial_cumulative_probability


@dataclass
class _Interval:
    lb: torch.Tensor
    ub: torch.Tensor


@dataclass
class _ClopperPearsonIntervals:
    region: _Interval
    partition: _Interval


class ClopperPearsonInterval:
    def __init__(
        self,
        probs_region: torch.Tensor,
        probs_partition: torch.Tensor,
        parameters: ConfidenceParameters,
    ):
        if probs_region.ndim != 2:
            raise ValueError("probs_region must have shape (nx, num_clusters)")

        if probs_partition.ndim != 2:
            raise ValueError("probs_partition must have shape (nx, num_clusters)")

        if probs_region.shape != probs_partition.shape:
            raise ValueError("probs_region and probs_partition must have matching shape")

        n = int(parameters.num_samples)
        if n <= 0:
            raise ValueError("parameters.num_samples must be positive")

        nx, num_clusters = probs_partition.shape
        if num_clusters <= 0:
            raise ValueError("probs_partition must contain at least one cluster")

        self.num_clusters = int(num_clusters)
        self._confidence_beta = float(parameters.beta)
        per_tail_beta = self._confidence_beta

        region_lb, region_ub = self._two_sided_interval_tensor(
            probs=probs_region,
            num_samples=n,
            tail_beta=per_tail_beta,
        )
        partition_lb, partition_ub = self._two_sided_interval_tensor(
            probs=probs_partition,
            num_samples=n,
            tail_beta=per_tail_beta,
        )

        self.intervals = _ClopperPearsonIntervals(
            region=_Interval(lb=region_lb, ub=region_ub),
            partition=_Interval(lb=partition_lb, ub=partition_ub),
        )

    @property
    def region(self) -> _Interval:
        return self.intervals.region

    @property
    def partition(self) -> _Interval:
        return self.intervals.partition

    @property
    def beta(self) -> float:
        return 2.0 * float(self.num_clusters) * self._confidence_beta

    def _two_sided_interval_tensor(
        self,
        probs: torch.Tensor,
        num_samples: int,
        tail_beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nx, num_clusters = probs.shape
        counts = torch.round(probs.clamp(0.0, 1.0) * num_samples).to(dtype=torch.long)

        lb = torch.empty_like(probs)
        ub = torch.empty_like(probs)
        for i in range(nx):
            for j in range(num_clusters):
                count = int(counts[i, j].item())
                lb[i, j] = self._lower_bound_from_count(
                    count=count,
                    num_samples=num_samples,
                    beta=tail_beta,
                )
                ub[i, j] = self._upper_bound_from_count(
                    count=count,
                    num_samples=num_samples,
                    beta=tail_beta,
                )

        return lb, ub

    @staticmethod
    def _lower_bound_from_count(count: int, num_samples: int, beta: float) -> float:
        # One-sided CP lower bound: solve P[X >= count | p] = beta for p.
        if count <= 0:
            return 0.0

        if count >= num_samples:
            return float(beta ** (1.0 / float(num_samples)))

        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            tail_prob = _binomial_cumulative_probability(
                from_k=count,
                to_k=num_samples,
                p=1.0 - mid,
                num_samples=num_samples,
            )
            if tail_prob >= beta:
                hi = mid
            else:
                lo = mid

        return hi

    @staticmethod
    def _upper_bound_from_count(count: int, num_samples: int, beta: float) -> float:
        # One-sided CP upper bound: solve P[X <= count | p] = beta for p.
        if count >= num_samples:
            return 1.0

        if count <= 0:
            return float(1.0 - beta ** (1.0 / float(num_samples)))

        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            cdf = _binomial_cumulative_probability(
                from_k=0,
                to_k=count,
                p=1.0 - mid,
                num_samples=num_samples,
            )
            if cdf >= beta:
                lo = mid
            else:
                hi = mid

        return lo