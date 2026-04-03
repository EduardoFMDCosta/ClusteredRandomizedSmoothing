import torch

from randomized_smoothing.dataclasses import ConfidenceParameters
from .utils import _binomial_cumulative_probability


class ClopperPearsonLowerBound:
    def __init__(
        self,
        coverage: float,
        partition_probs: torch.Tensor,
        parameters: ConfidenceParameters,
    ):
        self.beta = parameters.beta
        self.num_samples = parameters.num_samples
        
        self.coverage = coverage
        self.partition_probs = partition_probs
        self.num_clusters = partition_probs.shape[-1]

        self.p = self._construct_p()
        self.lower_bound = self._lower_bound_clopper_pearson()

    def _construct_p(self) -> torch.Tensor:
        if self.num_clusters == 1:
            return torch.tensor([[self.coverage]], dtype=torch.float64, device=self.partition_probs.device)
        else:
            nx = self.partition_probs.shape[0]
            region_coverage_repeated = torch.full((nx, self.num_clusters), self.coverage, dtype=self.partition_probs.dtype, device=self.partition_probs.device)
            return torch.cat([region_coverage_repeated, self.partition_probs], dim=1)

    def _lower_bound_clopper_pearson(self) -> torch.Tensor:
        lb = torch.zeros_like(self.p)
        nx = self.p.shape[0]
        num_events = self.p.shape[1]

        adjusted_beta = self.beta / num_events

        for row in range(nx):
            for col in range(num_events):
                prob = self.p[row, col].item()
                z = int(self.num_samples * prob)

                lo, hi = 0.0, 1.0
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    tail_mid = _binomial_cumulative_probability(from_idx=z, to_idx=self.num_samples, p=mid)
                    if tail_mid >= adjusted_beta:
                        hi = mid
                    else:
                        lo = mid

                lb[row, col] = 0.5 * (lo + hi)

        return lb