from dataclasses import dataclass
import torch

from randomized_smoothing.utils import _run_kmeans

@dataclass
class ClusterParameters:
    num_clusters: int
    max_iter: int = 1000
    tol: float = 1e-5

class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        params: ClusterParameters,
    ):
        self.params = params
        self.num_clusters = self.params.num_clusters
        self.centroids, self.assignments = self._compute_centroids(outputs)
        self.weights = self._compute_weights()
    
    def _compute_centroids(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nx, num_samples, output_dim = outputs.shape
        k = self.params.num_clusters

        centroids = torch.empty((nx, k, output_dim), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((nx, num_samples), dtype=torch.long, device=outputs.device)
        for i in range(nx):
            c_i, a_i, _ = _run_kmeans(
                outputs[i],
                num_clusters=k,
                max_iter=self.params.max_iter,
                tol=self.params.tol,
            )
            centroids[i] = c_i
            assignments[i] = a_i

        return centroids, assignments

    def _compute_weights(self) -> torch.Tensor:
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=self.num_clusters)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)