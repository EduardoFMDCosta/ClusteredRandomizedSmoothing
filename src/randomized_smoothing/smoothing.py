from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .clusters import Cluster


def _validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must satisfy 0 <= alpha < 0.5")


def _trim_core_per_dimension(samples: torch.Tensor, alpha: float) -> torch.Tensor:
    """Coordinate-wise symmetric alpha-trimming.

    For each dimension independently, sort values and remove alpha from the
    lower tail and alpha from the upper tail.

    Args:
        samples: Tensor of shape (n, q).
        alpha: Trimming level in [0, 0.5).

    Returns:
        Tensor of shape (n_trim, q), where each column contains its trimmed,
        sorted values.
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n, q)")

    n = samples.shape[0]
    if n == 0:
        raise ValueError("cannot trim an empty sample set")

    tail = int(alpha * n)
    sorted_vals, _ = torch.sort(samples, dim=0)
    return sorted_vals[tail : n - tail, :]


def _sample_from_trimmed_core(trimmed_core: torch.Tensor, num_draws: int) -> torch.Tensor:
    """Sample from coordinate-wise trimmed empirical distribution.

    For q>1, each coordinate is sampled independently from its trimmed marginal.
    """
    if num_draws <= 0:
        raise ValueError("num_draws must be > 0")

    n_trim, q = trimmed_core.shape
    if n_trim == 0:
        raise ValueError("trimmed core must be non-empty")

    idx = torch.randint(0, n_trim, (num_draws, q), device=trimmed_core.device)
    return trimmed_core.gather(dim=0, index=idx)


class AlphaTrimmingSmoothing:
    """Global alpha-trimming smoother on stochastic outputs.

    Given outputs with shape (m, num_samples, q), computes an alpha-trimmed
    average per input and returns shape (m, q).
    """

    def __init__(self, alpha: float):
        _validate_alpha(alpha)
        self.alpha = alpha

    def smooth(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 3:
            raise ValueError("outputs must have shape (m, num_samples, q)")

        m, _, _ = outputs.shape
        smoothed = []
        for i in range(m):
            trimmed_core = _trim_core_per_dimension(outputs[i], alpha=self.alpha)
            smoothed.append(trimmed_core.mean(dim=0))
        return torch.stack(smoothed, dim=0)

    def sample_from_trimmed(self, outputs: torch.Tensor, num_draws: int) -> torch.Tensor:
        """Sample points from alpha-trimmed empirical sets.

        Args:
            outputs: Tensor with shape (m, num_samples, q).
            num_draws: Number of draws per input from the trimmed set.

        Returns:
            Tensor with shape (m, num_draws, q).
        """
        if outputs.ndim != 3:
            raise ValueError("outputs must have shape (m, num_samples, q)")
        if num_draws <= 0:
            raise ValueError("num_draws must be > 0")

        m, _, q = outputs.shape
        sampled = torch.empty((m, num_draws, q), dtype=outputs.dtype, device=outputs.device)

        for i in range(m):
            trimmed_core = _trim_core_per_dimension(outputs[i], alpha=self.alpha)
            sampled[i] = _sample_from_trimmed_core(trimmed_core, num_draws=num_draws)

        return sampled


@dataclass
class ClusterizedSmoothingResult:
    cluster_means: torch.Tensor
    cluster_weights: torch.Tensor
    cluster_centroids: torch.Tensor
    sampled: torch.Tensor


class ClusterizedAlphaTrimmingSmoothing:
    """Cluster-aware alpha trimming and weighted re-sampling.

    For each input i:
    1. Assign each sample to the nearest centroid.
    2. Inside each cluster, trim far-away samples by alpha and compute mean.
    3. Use cluster weights to sample from the weighted empirical distribution
       over cluster means.
    """

    def __init__(self, alpha: float):
        _validate_alpha(alpha)
        self.alpha = alpha

    def _cluster_trim_means(
        self,
        samples_i: torch.Tensor,
        cluster: Cluster,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_centroids = cluster.num_centroids
        q = samples_i.shape[1]

        distances = torch.cdist(samples_i, cluster.centroids)
        assignments = distances.argmin(dim=1)

        means = torch.zeros((num_centroids, q), dtype=samples_i.dtype, device=samples_i.device)
        valid = torch.zeros(num_centroids, dtype=torch.bool, device=samples_i.device)

        for k in range(num_centroids):
            mask = assignments == k
            if not mask.any():
                continue
            cluster_points = samples_i[mask]
            trimmed_core = _trim_core_per_dimension(cluster_points, alpha=self.alpha)
            means[k] = trimmed_core.mean(dim=0)
            valid[k] = True

        # Backfill empty clusters with their centroids to keep a stable support.
        means[~valid] = cluster.centroids[~valid]
        return means, valid

    def smooth_and_sample(
        self,
        outputs: torch.Tensor,
        cluster: Cluster | Sequence[Cluster],
        num_draws: int,
    ) -> ClusterizedSmoothingResult:
        if outputs.ndim != 3:
            raise ValueError("outputs must have shape (m, num_samples, q)")
        if num_draws <= 0:
            raise ValueError("num_draws must be > 0")

        m = outputs.shape[0]
        if isinstance(cluster, Cluster):
            clusters = [cluster for _ in range(m)]
        else:
            clusters = list(cluster)
            if len(clusters) != m:
                raise ValueError("when passing multiple clusters, length must equal m")

        num_centroids = clusters[0].num_centroids
        q = outputs.shape[2]

        for i, cluster_i in enumerate(clusters):
            if cluster_i.num_centroids != num_centroids:
                raise ValueError("all clusters must have the same number of centroids")
            if cluster_i.centroids.shape[1] != q:
                raise ValueError(f"cluster at index {i} has centroid dimension incompatible with q")

        cluster_means = torch.empty((m, num_centroids, q), dtype=outputs.dtype, device=outputs.device)
        cluster_weights = torch.empty((m, num_centroids), dtype=outputs.dtype, device=outputs.device)
        cluster_centroids = torch.empty((m, num_centroids, q), dtype=outputs.dtype, device=outputs.device)
        sampled = torch.empty((m, num_draws, q), dtype=outputs.dtype, device=outputs.device)

        for i in range(m):
            cluster_i = clusters[i]
            means_i, valid_i = self._cluster_trim_means(outputs[i], cluster_i)
            weights_i = cluster_i.weights.to(device=outputs.device, dtype=outputs.dtype)

            # Renormalize over clusters actually represented in the data.
            if valid_i.any():
                effective_weights = weights_i * valid_i.to(weights_i.dtype)
                effective_weights = effective_weights / effective_weights.sum()
            else:
                effective_weights = weights_i

            draw_idx = torch.multinomial(effective_weights, num_samples=num_draws, replacement=True)
            sampled[i] = means_i[draw_idx]
            cluster_means[i] = means_i
            cluster_weights[i] = weights_i
            cluster_centroids[i] = cluster_i.centroids.to(device=outputs.device, dtype=outputs.dtype)

        return ClusterizedSmoothingResult(
            cluster_means=cluster_means,
            cluster_weights=cluster_weights,
            cluster_centroids=cluster_centroids,
            sampled=sampled,
        )

    def sample_from_trimmed(
        self,
        outputs: torch.Tensor,
        cluster: Cluster | Sequence[Cluster],
        num_draws: int,
    ) -> torch.Tensor:
        """Sample from per-cluster trimmed empirical sets.

        This is analogous to global trimming Monte Carlo:
        - compute trimmed sets on fresh outputs,
        - sample from those trimmed sets.
        The only difference is trimming is done separately per cluster.
        """
        if outputs.ndim != 3:
            raise ValueError("outputs must have shape (m, num_samples, q)")
        if num_draws <= 0:
            raise ValueError("num_draws must be > 0")

        m, _, q = outputs.shape
        if isinstance(cluster, Cluster):
            clusters = [cluster for _ in range(m)]
        else:
            clusters = list(cluster)
            if len(clusters) != m:
                raise ValueError("when passing multiple clusters, length must equal m")

        num_centroids = clusters[0].num_centroids
        sampled = torch.empty((m, num_draws, q), dtype=outputs.dtype, device=outputs.device)

        for i in range(m):
            cluster_i = clusters[i]
            if cluster_i.num_centroids != num_centroids:
                raise ValueError("all clusters must have the same number of centroids")
            if cluster_i.centroids.shape[1] != q:
                raise ValueError(f"cluster at index {i} has centroid dimension incompatible with q")

            samples_i = outputs[i]
            distances = torch.cdist(samples_i, cluster_i.centroids)
            assignments = distances.argmin(dim=1)

            trimmed_points_by_cluster: list[torch.Tensor] = []
            valid = torch.zeros(num_centroids, dtype=torch.bool, device=outputs.device)
            for k in range(num_centroids):
                mask = assignments == k
                if not mask.any():
                    trimmed_points_by_cluster.append(cluster_i.centroids[k : k + 1])
                    continue
                cluster_points = samples_i[mask]
                trimmed_core = _trim_core_per_dimension(cluster_points, alpha=self.alpha)
                trimmed_points_by_cluster.append(trimmed_core)
                valid[k] = True

            weights_i = cluster_i.weights.to(device=outputs.device, dtype=outputs.dtype)
            if valid.any():
                effective_weights = weights_i * valid.to(weights_i.dtype)
                effective_weights = effective_weights / effective_weights.sum()
            else:
                effective_weights = weights_i

            draw_cluster_idx = torch.multinomial(effective_weights, num_samples=num_draws, replacement=True)

            draws_i = torch.empty((num_draws, q), dtype=outputs.dtype, device=outputs.device)
            for t in range(num_draws):
                k = int(draw_cluster_idx[t].item())
                candidates = trimmed_points_by_cluster[k]
                draws_i[t] = _sample_from_trimmed_core(candidates, num_draws=1).squeeze(0)

            sampled[i] = draws_i

        return sampled
