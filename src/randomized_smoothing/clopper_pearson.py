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
        self.beta = float(beta)
        self.num_samples = num_samples
        self.cluster = cluster

        self.p = self._construct_p(p=p)
        self.lower_bound = self._lower_bound_clopper_pearson()

    def _construct_p(self, p: float) -> torch.Tensor:
        if self.cluster is None:
            return torch.tensor([p], dtype=torch.float32)
        else:
            weights = self.cluster.weights.reshape(-1)
            m = weights.numel()
            p_block = torch.full((m,), p, dtype=weights.dtype, device=weights.device)
            out = torch.cat([p_block, weights], dim=0)
            return out

    def _lower_bound_clopper_pearson(self) -> torch.Tensor:
        m_prob = self.p.shape[0]
        if m_prob == 0:
            raise ValueError("p must contain at least one probability")

        beta_per = self.beta / float(m_prob)
        m_samples = int(self.num_samples)

        out = torch.zeros(m_prob, dtype=torch.float64, device=self.p.device)

        for i in range(m_prob):
            p_i = float(self.p[i].item())
            z_i = int(m_samples * p_i)

            if z_i == 0 or beta_per == 0.0:
                out[i] = 0.0
                continue
            if beta_per == 1.0:
                out[i] = 1.0
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

            out[i] = 0.5 * (lo + hi)

        return out.to(dtype=self.p.dtype)