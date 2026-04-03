from dataclasses import dataclass
import torch

from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import HyperRectangle
from randomized_smoothing.smoothing import SmoothingParameters
from randomized_smoothing.utils import _run_kmeans


class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        params: SmoothingParameters,
        clustering_params: ClusteringParameters,
    ):
        self.params = params
        self.clustering_params = clustering_params
        self.num_clusters = clustering_params.num_clusters

        self.centroids, self.assignments = self._compute_centroids(outputs)
        self.weights = self._compute_weights()

        self.regions = self._get_regions(outputs)
    
    def _compute_centroids(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nx, num_samples, output_dim = outputs.shape
        k = self.num_clusters

        centroids = torch.empty((nx, k, output_dim), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((nx, num_samples), dtype=torch.long, device=outputs.device)
        for i in range(nx):
            c_i, a_i, _ = _run_kmeans(
                outputs[i],
                num_clusters=k,
                max_iter=self.clustering_params.max_iter,
                tol=self.clustering_params.tol,
            )
            centroids[i] = c_i
            assignments[i] = a_i

        return centroids, assignments

    def _compute_weights(self) -> torch.Tensor:
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=self.params.num_clusters)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)
    
    def _get_regions(self, outputs: torch.Tensor):
        return HyperRectangle.from_samples(samples=outputs, centroids=self.centroids, region_coverage=self.params.region_coverage)