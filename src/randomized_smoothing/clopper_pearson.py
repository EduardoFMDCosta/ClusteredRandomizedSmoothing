import torch
from dataclasses import dataclass

from randomized_smoothing.dataclasses import ConfidenceParameters
from .utils import _binomial_cumulative_probability


@dataclass
class _LowerBound:
    coverage: torch.Tensor
    partition_probs: torch.Tensor | None


class ClopperPearsonLowerBound:
    def __init__(
        self,
        coverage: float,
        partition_probs: torch.Tensor,
        parameters: ConfidenceParameters,
    ):
        if partition_probs.ndim != 2:
            raise ValueError("partition_probs must have shape (nx, num_clusters)")

        n = int(parameters.num_samples)
        if n <= 0:
            raise ValueError("parameters.num_samples must be positive")

        nx, num_clusters = partition_probs.shape
        if num_clusters <= 0:
            raise ValueError("partition_probs must contain at least one cluster")

        if num_clusters == 1:
            delta = float(parameters.beta)
            partition_lb = None
        else:
            delta = float(parameters.beta) / (2.0 * float(num_clusters))
            counts_partition = torch.round(partition_probs.clamp(0.0, 1.0) * n).to(dtype=torch.long)
            partition_lb = torch.empty_like(partition_probs)
            for i in range(nx):
                for j in range(num_clusters):
                    partition_lb[i, j] = self._lower_bound_from_count(
                        count=int(counts_partition[i, j].item()),
                        num_samples=n,
                        delta=delta,
                    )

        count_coverage = int(round(float(coverage) * n))
        count_coverage = max(0, min(n, count_coverage))
        coverage_lb_scalar = self._lower_bound_from_count(
            count=count_coverage,
            num_samples=n,
            delta=delta,
        )
        coverage_lb = torch.full(
            (nx, 1),
            fill_value=coverage_lb_scalar,
            dtype=partition_probs.dtype,
            device=partition_probs.device,
        )

        self.lower_bound = _LowerBound(
            coverage=coverage_lb,
            partition_probs=partition_lb,
        )

    @staticmethod
    def _lower_bound_from_count(count: int, num_samples: int, delta: float) -> float:
        # One-sided CP lower bound: solve P[X >= count | p] = delta for p.
        if count <= 0:
            return 0.0

        if count >= num_samples:
            return float(delta ** (1.0 / float(num_samples)))

        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            tail_prob = _binomial_cumulative_probability(
                from_idx=count,
                to_idx=num_samples,
                p=1.0 - mid,
            )
            if tail_prob >= delta:
                hi = mid
            else:
                lo = mid

        return hi