from typing import Optional
import torch

from .utils import _binomial_cumulative_probability

class ClopperPearsonLowerBound:
    def __init__(
        self,
        p: torch.Tensor,
        beta: float,
        num_samples: int,
    ):
        self.beta = beta
        self.num_samples = num_samples
        self.p = p
        self.lower_bound = self._lower_bound_clopper_pearson()

    def _lower_bound_clopper_pearson(self) -> torch.Tensor:
        lb = torch.zeros_like(self.p)
        nx = self.p.shape[0]
        num_clusters = self.p.shape[1]

        adjusted_beta = self.beta / num_clusters

        for row in range(nx):
            for col in range(num_clusters):
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