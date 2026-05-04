import torch

from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import AxisAlignedHyperRectangle
from randomized_smoothing.utils import (
    _select_num_clusters,
    _select_num_clusters_hdbscan,
)


def _compute_centroids_from_assignments(
    data: torch.Tensor,
    assignments: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    centroids = torch.empty((num_clusters, data.shape[1]), dtype=data.dtype, device=data.device)
    for j in range(num_clusters):
        points = data[assignments == j]
        centroids[j] = points.mean(dim=0)
    return centroids


class Cluster:
    def __init__(
        self,
        outputs: torch.Tensor,
        parameters: ClusteringParameters,
        behavior_anchor: torch.Tensor | None = None,
    ):
        self.parameters = parameters
        self.behavior_anchor = behavior_anchor
        self.centroids, self.assignments, self.k_per_row = self._compute_centroids(outputs=outputs)
        self._k_max: int = int(self.k_per_row.max().item())
        self.regions = self._get_regions(outputs=outputs)
        self.partition_assignments = self._compute_partition_assignments(outputs=outputs)
        self.probs_partition = self._compute_probs_partition()
        self.probs_region = self._compute_probs_regions(outputs=outputs)


    @property
    def num_clusters(self):
        return self._k_max
    
    def _compute_centroids(
        self, outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run per-row clustering and pad rows to shared ``k_max``.

        Phantom centroid slots (indices k_i .. k_max-1 for row i) are filled
        with a copy of that row's cluster-0 centroid.  Because `torch.argmin`
        resolves ties by returning the *first* matching index, any sample that
        is equidistant to the real cluster-0 and a phantom centroid will always
        be assigned to index 0 — so phantom clusters accumulate zero probability
        mass without any additional masking in downstream code.
        """
        nx, num_samples, output_dim = outputs.shape

        if self.parameters.strategy == "behavior_split":
            return self._compute_behavior_split_centroids(outputs)

        # ---- per-row elbow selection ----------------------------------------
        k_list: list[int] = []
        centroids_list: list[torch.Tensor] = []
        assignments_list: list[torch.Tensor] = []
        for i in range(nx):
            if self.parameters.strategy == "hdbscan":
                c_i, a_i, k_i = _select_num_clusters_hdbscan(
                    data=outputs[i],
                    max_num_clusters=self.parameters.max_num_clusters,
                    hdbscan_min_cluster_size=self.parameters.hdbscan_min_cluster_size,
                    hdbscan_min_samples=self.parameters.hdbscan_min_samples,
                    hdbscan_cluster_selection_epsilon=self.parameters.hdbscan_cluster_selection_epsilon,
                    hdbscan_cluster_selection_method=self.parameters.hdbscan_cluster_selection_method,
                )
            else:
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

    def _compute_behavior_split_centroids(
        self,
        outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nx, num_samples, output_dim = outputs.shape
        if output_dim < 2:
            raise ValueError("behavior_split requires output_dim >= 2")
        if self.parameters.max_num_clusters < 2:
            assignments = torch.zeros((nx, num_samples), dtype=torch.long, device=outputs.device)
            centroids = outputs.mean(dim=1, keepdim=True)
            k_per_row = torch.ones((nx,), dtype=torch.long, device=outputs.device)
            return centroids, assignments, k_per_row
        if self.behavior_anchor is None:
            raise ValueError("behavior_split requires behavior_anchor with shape (nx, 2)")
        if self.behavior_anchor.shape != (nx, 2):
            raise ValueError(
                "behavior_anchor must have shape (nx, 2); "
                f"got {tuple(self.behavior_anchor.shape)}"
            )

        endpoints = outputs[..., -2:]
        anchor = self.behavior_anchor.to(device=outputs.device, dtype=outputs.dtype)

        centroids = torch.empty((nx, 2, output_dim), dtype=outputs.dtype, device=outputs.device)
        assignments = torch.empty((nx, num_samples), dtype=torch.long, device=outputs.device)
        k_per_row = torch.empty((nx,), dtype=torch.long, device=outputs.device)

        for i in range(nx):
            pts_full = outputs[i]
            pts_end = endpoints[i]
            init = anchor[i]
            delta = pts_end - init.unsqueeze(0)

            stay_mask = torch.linalg.norm(delta, dim=1) <= self.parameters.behavior_stay_radius
            turn_mask = (
                (delta[:, 1] <= -self.parameters.behavior_min_down)
                & (delta[:, 0] <= -self.parameters.behavior_min_left)
                & (~stay_mask)
            )

            if stay_mask.any():
                stay_center = pts_end[stay_mask].mean(dim=0)
            else:
                stay_center = init

            if turn_mask.any():
                turn_center = pts_end[turn_mask].mean(dim=0)
            else:
                non_stay = pts_end[~stay_mask]
                if non_stay.numel() > 0:
                    non_stay_delta = non_stay - init.unsqueeze(0)
                    turn_center = non_stay[torch.argmin(non_stay_delta[:, 1])]
                else:
                    turn_center = init + torch.tensor(
                        [-self.parameters.behavior_min_left, -self.parameters.behavior_min_down],
                        device=pts_end.device,
                        dtype=pts_end.dtype,
                    )

            endpoint_centroids = torch.stack([stay_center, turn_center], dim=0)
            d2 = ((pts_end.unsqueeze(1) - endpoint_centroids.unsqueeze(0)) ** 2).sum(dim=-1)
            row_assign = d2.argmin(dim=1)

            # Ensure both clusters are represented when requested.
            if self.parameters.behavior_force_two_clusters:
                count0 = int((row_assign == 0).sum().item())
                count1 = int((row_assign == 1).sum().item())
                if count0 == 0:
                    row_assign[torch.argmax(delta[:, 1])] = 0
                if count1 == 0:
                    row_assign[torch.argmin(delta[:, 1])] = 1

            unique = torch.unique(row_assign)
            if unique.numel() == 1 and not self.parameters.behavior_force_two_clusters:
                k_i = 1
                row_assign = torch.zeros_like(row_assign)
                row_centroids = pts_full.mean(dim=0, keepdim=True)
                centroids[i, 0] = row_centroids[0]
                centroids[i, 1] = row_centroids[0]
            else:
                k_i = 2
                row_centroids = _compute_centroids_from_assignments(
                    data=pts_full,
                    assignments=row_assign,
                    num_clusters=2,
                )
                centroids[i, 0] = row_centroids[0]
                centroids[i, 1] = row_centroids[1]

            assignments[i] = row_assign
            k_per_row[i] = k_i

        return centroids, assignments, k_per_row

    def _compute_probs_partition(self) -> torch.Tensor:
        one_hot_assignments = torch.nn.functional.one_hot(
            self.partition_assignments.long(),
            num_classes=self.num_clusters,
        )
        return one_hot_assignments.to(dtype=self.centroids.dtype).mean(dim=1)

    def _compute_partition_assignments(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.regions.distance(outputs).argmin(dim=2)
    
    def _get_regions(self, outputs: torch.Tensor):
        return AxisAlignedHyperRectangle.from_samples(
            samples=outputs,
            centroids=self.centroids,
            assignments=self.assignments,
            coverage=self.parameters.coverage,
        )
    
    def _compute_probs_regions(self, outputs: torch.Tensor) -> torch.Tensor:
        in_region = self.regions.contains(outputs)
        return in_region.to(dtype=outputs.dtype).mean(dim=1)