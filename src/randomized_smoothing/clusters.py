from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Cluster:
    """Cluster summary used by clusterized smoothing.

    Attributes:
        centroids: Tensor of shape (num_centroids, q).
        weights: Tensor of shape (num_centroids,). Should sum to 1.
    """

    centroids: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.centroids.ndim != 2:
            raise ValueError("centroids must have shape (num_centroids, q)")
        if self.weights.ndim != 1:
            raise ValueError("weights must have shape (num_centroids,)")
        if self.centroids.shape[0] != self.weights.shape[0]:
            raise ValueError("weights length must match number of centroids")

        w_sum = self.weights.sum()
        if w_sum <= 0:
            raise ValueError("weights must sum to a positive value")

        self.weights = self.weights / w_sum

    @property
    def num_centroids(self) -> int:
        return int(self.centroids.shape[0])
