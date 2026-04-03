import torch

from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import HyperRectangle
from randomized_smoothing.utils import _run_kmeans


class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        parameters: ClusteringParameters,
        coverage: float
    ):
        self.parameters = parameters
        self.num_clusters = parameters.num_clusters

        self.centroids, self.assignments = self._compute_centroids(outputs=outputs)
        self.weights = self._compute_weights()

        self.regions = self._get_regions(outputs=outputs, coverage=coverage)
    
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
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=self.num_clusters)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)
    
    def _get_regions(self, outputs: torch.Tensor, coverage: float):
        return HyperRectangle.from_samples(samples=outputs, centroids=self.centroids, coverage=coverage)