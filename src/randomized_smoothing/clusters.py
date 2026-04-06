import torch

from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import HyperRectangle
from randomized_smoothing.utils import _run_kmeans


class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        parameters: ClusteringParameters,
    ):
        self.parameters = parameters
        self.centroids, self.assignments = self._compute_centroids(outputs=outputs)
        self.probs_partition = self._compute_probs_partition()

        self.regions = self._get_regions(outputs=outputs)
        self.probs_region = self._compute_probs_regions(outputs=outputs)


    @property
    def num_clusters(self):
        return self.parameters.num_clusters
    
    def _compute_centroids(self, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nx, num_samples, output_dim = outputs.shape
        k = self.parameters.num_clusters

        centroids = torch.empty((nx, k, output_dim), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((nx, num_samples), dtype=torch.long, device=outputs.device)
        for i in range(nx):
            c_i, a_i, _ = _run_kmeans(
                outputs[i],
                num_clusters=k,
                max_iter=self.parameters.max_iter,
                tol=self.parameters.tol,
            )
            centroids[i] = c_i
            assignments[i] = a_i

        return centroids, assignments

    def _compute_probs_partition(self) -> torch.Tensor:
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=self.parameters.num_clusters)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)
    
    def _get_regions(self, outputs: torch.Tensor) -> HyperRectangle:
        return HyperRectangle.from_samples(samples=outputs, centroids=self.centroids, coverage=self.parameters.coverage)
    
    def _compute_probs_regions(self, outputs: torch.Tensor) -> torch.Tensor:
        # Compute how many samples fall into each region
        # Returns a tensor (nx, num_clusters) where each entry is the probability of a sample falling into the corresponding region
        lower = self.regions.lower
        upper = self.regions.upper

        in_region = (outputs.unsqueeze(2) >= lower.unsqueeze(1)) & (outputs.unsqueeze(2) <= upper.unsqueeze(1))
        in_region = in_region.all(dim=-1)
        return in_region.to(dtype=outputs.dtype).mean(dim=1)