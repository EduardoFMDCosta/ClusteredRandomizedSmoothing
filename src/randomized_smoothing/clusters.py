from dataclasses import dataclass
import torch

from randomized_smoothing.utils import _run_kmeans, _select_num_clusters_elbow

@dataclass
class ClusterParameters:
    num_clusters: int | None = None
    num_samples: int = 1000
    max_num_clusters: int = 10
    max_iter: int = 1000
    tol: float = 1e-5

class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        params: ClusterParameters = ClusterParameters(),
    ):
        self.params = params
        self.centroids, self.assignments = self._compute_centroids(outputs)
        self.weights = self._compute_weights()
    
    def _compute_centroids(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m, n, q = outputs.shape

        if self.params.num_clusters is not None:
            if self.params.num_clusters <= 0:
                raise ValueError("num_clusters must be positive when provided")
            k = min(self.params.num_clusters, n)
        else:
            flat_outputs = outputs.reshape(m * n, q)
            sample_size = min(self.params.num_samples, flat_outputs.shape[0])
            generator = torch.Generator(device=flat_outputs.device)
            generator.manual_seed(0)

            if sample_size < flat_outputs.shape[0]:
                idx = torch.randperm(flat_outputs.shape[0], generator=generator, device=flat_outputs.device)[:sample_size]
                elbow_data = flat_outputs[idx]
            else:
                elbow_data = flat_outputs

            k = _select_num_clusters_elbow(
                elbow_data,
                max_k=min(self.params.max_num_clusters, n),
                max_iter=self.params.max_iter,
                tol=self.params.tol,
            )

        centroids = torch.empty((m, k, q), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((m, n), dtype=torch.long, device=outputs.device)

        for i in range(m):
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
        m, k, _ = self.centroids.shape

        one_hot_assignments = torch.nn.functional.one_hot(self.assignments, num_classes=k)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)