from typing import Optional
import torch

from .utils import _binomial_tail_probability
from .clusters import Cluster

class ClopperPearsonLowerBound:
    def __init__(
        self,
        p: float,
        beta: float,
        num_samples: int,
        cluster: Optional[Cluster] = None
    ):
        self.beta = beta
        self.num_samples = num_samples
        self.cluster = cluster

        self.p = self._construct_p(p=p)
        self.lower_bound = self._lower_bound_clopper_pearson()

    def _construct_p(self, p: float) -> torch.Tensor:
        if self.cluster is None:
            return torch.tensor([[p]], dtype=torch.float32)
        else:
            weights = self.cluster.weights.squeeze(-1)
            p_block = torch.full_like(weights, p)
            return torch.cat([p_block, weights], dim=1)

    def _lower_bound_clopper_pearson(self) -> torch.Tensor:
        
        lb = torch.zeros_like(self.p)
        nx = self.p.shape[0]
        num_clusters = self.p.shape[1]

        adjusted_beta = self.beta / num_clusters

        for row in range(nx):
            for col in range(num_clusters):
                p = self.p[row, col].item()
                z = int(self.num_samples * p)

                lo, hi = 0.0, 1.0
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    tail_mid = _binomial_tail_probability(z=z, num_samples=self.num_samples, p=mid)
                    if tail_mid >= adjusted_beta:
                        hi = mid
                    else:
                        lo = mid

                lb[row, col] = 0.5 * (lo + hi)

        return lb