import torch
from dataclasses import dataclass

from randomized_smoothing.dataclasses import ConfidenceParameters
from .optimization import _binomial_cumulative_probability


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
        n = int(parameters.num_samples)
        nx, num_clusters = probs_partition.shape

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

        # If clustering produces exactly one partition set, that set must be the
        # entire space, so its probability is 1.
        if self.num_clusters == 1:
            partition_lb = torch.ones_like(probs_partition)
            partition_ub = torch.ones_like(probs_partition)

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
        counts = torch.round(probs.clamp(0.0, 1.0) * num_samples).to(dtype=torch.long)

        flat_counts = counts.reshape(-1)
        flat_lb = torch.empty_like(flat_counts, dtype=probs.dtype)
        flat_ub = torch.empty_like(flat_counts, dtype=probs.dtype)

        for idx, count_tensor in enumerate(flat_counts):
            count = int(count_tensor.item())
            flat_lb[idx] = self._lower_bound_from_count(
                count=count,
                num_samples=num_samples,
                beta=tail_beta,
            )
            flat_ub[idx] = self._upper_bound_from_count(
                count=count,
                num_samples=num_samples,
                beta=tail_beta,
            )

        return flat_lb.view_as(probs), flat_ub.view_as(probs)

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
                p=mid,
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
                p=mid,
                num_samples=num_samples,
            )
            if cdf >= beta:
                lo = mid
            else:
                hi = mid

        return lo