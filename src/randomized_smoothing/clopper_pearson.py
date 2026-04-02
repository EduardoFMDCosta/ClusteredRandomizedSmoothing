from typing import Optional
import torch

from .utils import clopper_pearson_tail_probability
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
        m_samples = int(self.num_samples)
        out = torch.zeros_like(self.p, dtype=torch.float64)

        beta_per = self.beta / self.p.shape[1]

        for row_idx in range(self.p.shape[0]):
            for col_idx in range(self.p.shape[1]):
                p_i = float(self.p[row_idx, col_idx].item())
                z_i = int(m_samples * p_i)

                if z_i == 0 or beta_per == 0.0:
                    out[row_idx, col_idx] = 0.0
                    continue
                if beta_per == 1.0:
                    out[row_idx, col_idx] = 1.0
                    continue

                lo = 0.0
                hi = 1.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    tail_mid = clopper_pearson_tail_probability(z=z_i, m=m_samples, p=mid)
                    if tail_mid >= beta_per:
                        hi = mid
                    else:
                        lo = mid

                out[row_idx, col_idx] = 0.5 * (lo + hi)

        return out.to(dtype=self.p.dtype)