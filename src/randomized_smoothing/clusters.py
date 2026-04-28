import torch

from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import (
    AxisAlignedHyperRectangle,
    EuclideanBall,
    HyperRectangle,
    TimeStepHyperrectangles,
)
from randomized_smoothing.utils import _select_num_clusters


_SET_CLASSES = {
    "HyperRectangle": HyperRectangle,
    "AxisAlignedHyperRectangle": AxisAlignedHyperRectangle,
    "EuclideanBall": EuclideanBall,
    "TimeStepHyperrectangles": TimeStepHyperrectangles,
}


class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        parameters: ClusteringParameters,
    ):
        self.parameters = parameters
        self.centroids, self.assignments, self.k_per_row = self._compute_centroids(outputs=outputs)
        self._k_max: int = int(self.k_per_row.max().item())
        self.probs_partition = self._compute_probs_partition()

        self.regions = self._get_regions(outputs=outputs)
        self.probs_region = self._compute_probs_regions(outputs=outputs)


    @property
    def num_clusters(self):
        return self._k_max
    
    def _compute_centroids(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run per-row DBSCAN clustering and pad rows to shared ``k_max``.

        Phantom centroid slots (indices k_i .. k_max-1 for row i) are filled
        with a copy of that row's cluster-0 centroid.  Because `torch.argmin`
        resolves ties by returning the *first* matching index, any sample that
        is equidistant to the real cluster-0 and a phantom centroid will always
        be assigned to index 0 — so phantom clusters accumulate zero probability
        mass without any additional masking in downstream code.
        """
        nx, num_samples, output_dim = outputs.shape

        # ---- per-row elbow selection ----------------------------------------
        k_list: list[int] = []
        centroids_list: list[torch.Tensor] = []
        assignments_list: list[torch.Tensor] = []
        for i in range(nx):
            c_i, a_i, k_i = _select_num_clusters(
                data=outputs[i],
                max_num_clusters=self.parameters.max_num_clusters,
                dbscan_eps=self.parameters.dbscan_eps,
                dbscan_min_samples=self.parameters.dbscan_min_samples,
            )
            k_list.append(k_i)
            centroids_list.append(c_i)
            assignments_list.append(a_i)

        # ---- pad to k_max ---------------------------------------------------
        k_max = max(k_list)
        k_per_row = torch.tensor(k_list, dtype=torch.long, device=outputs.device)

        centroids = torch.empty((nx, k_max, output_dim), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((nx, num_samples), dtype=torch.long, device=outputs.device)
        for i in range(nx):
            k_i = k_list[i]
            centroids[i, :k_i, :] = centroids_list[i]
            # Phantom slots: copy cluster-0 centroid so argmin always resolves
            # to index 0 (the real cluster) when distances tie.
            if k_i < k_max:
                centroids[i, k_i:, :] = centroids_list[i][0].unsqueeze(0).expand(k_max - k_i, -1)
            assignments[i] = assignments_list[i]

        return centroids, assignments, k_per_row

    def _compute_probs_partition(self) -> torch.Tensor:
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=self.num_clusters)
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)
    
    def _get_regions(self, outputs: torch.Tensor):
        region_cls = _SET_CLASSES.get(self.parameters.set_type)
        if region_cls is None:
            raise ValueError(f"Unsupported set_type: {self.parameters.set_type}")

        return region_cls.from_samples(
            samples=outputs,
            centroids=self.centroids,
            assignments=self.assignments,
            coverage=self.parameters.coverage,
        )
    
    def _compute_probs_regions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute probability of samples falling in region j, conditioned on being assigned to cluster j.
        
        This ensures by construction that probs_region[i,j] <= probs_partition[i,j],
        since a sample can only be in region j if it's assigned to cluster j.
        """
        num_clusters = self.num_clusters

        # Compute which samples are in which regions (shape: nx, num_samples, num_clusters)
        in_region = self.regions.contains(outputs)

        # Get one-hot assignments (shape: nx, num_samples, num_clusters)
        one_hot_assignments = torch.nn.functional.one_hot(self.assignments.long(), num_classes=num_clusters)
        one_hot_assignments = one_hot_assignments.to(dtype=outputs.dtype)

        # For each cluster j, count samples that are both:
        # 1. assigned to cluster j
        # 2. fall within region j
        in_region_and_assigned = in_region.to(dtype=outputs.dtype) * one_hot_assignments

        # Probability: (samples in region j AND assigned to j) / total samples
        return in_region_and_assigned.mean(dim=1)