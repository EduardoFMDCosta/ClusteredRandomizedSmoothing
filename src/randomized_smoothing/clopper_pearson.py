from dataclasses import dataclass
import torch

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, SmoothingParameters

from .utils import _binomial_cumulative_probability


class ClopperPearsonLowerBound:
    def __init__(
        self,
        cluster: Cluster,
        clustering_parameters: ClusteringParameters,
        confidence_params: ConfidenceParameters,
    ):
        self.beta = confidence_params.beta
        self.num_samples = confidence_params.num_samples

        self.region_coverage = clustering_parameters.region_coverage
        self.p_partition = cluster.weights
        self.num_clusters = cluster.num_clusters

        self.p = self._construct_p()
        self.lower_bound = self._lower_bound_clopper_pearson()

    def _construct_p(self) -> torch.Tensor:
        if self.num_clusters == 1:
            return torch.tensor([[self.region_coverage]], dtype=torch.float64, device=self.p_partition.device)
        else:
            nx = self.p_partition.shape[0]
            region_coverage_repeated = torch.full((nx, self.num_clusters), self.region_coverage, dtype=self.p_partition.dtype, device=self.p_partition.device)
            return torch.cat([region_coverage_repeated, self.p_partition], dim=1)

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