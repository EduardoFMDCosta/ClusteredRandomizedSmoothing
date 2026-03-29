from __future__ import annotations

from typing import Tuple

import torch

from .clusters import Cluster


def _kmeans_plus_plus_init(data: torch.Tensor, num_clusters: int) -> torch.Tensor:
    n, q = data.shape
    if num_clusters > n:
        raise ValueError("num_clusters cannot exceed number of data points")

    generator = torch.Generator(device=data.device)
    generator.manual_seed(0)

    centroids = torch.empty((num_clusters, q), dtype=data.dtype, device=data.device)
    first_idx = torch.randint(0, n, (1,), generator=generator, device=data.device)
    centroids[0] = data[first_idx]

    min_sq_dist = torch.cdist(data, centroids[0:1]).squeeze(1).pow(2)

    for k in range(1, num_clusters):
        probs = min_sq_dist / min_sq_dist.sum()
        new_idx = torch.multinomial(probs, num_samples=1, generator=generator)
        centroids[k] = data[new_idx]
        new_sq_dist = torch.cdist(data, centroids[k : k + 1]).squeeze(1).pow(2)
        min_sq_dist = torch.minimum(min_sq_dist, new_sq_dist)

    return centroids


def run_kmeans(
    data: torch.Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run K-means clustering on data with shape (n, q)."""
    if data.ndim != 2:
        raise ValueError("data must have shape (n, q)")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0")

    centroids = _kmeans_plus_plus_init(data, num_clusters)
    assignments = torch.zeros(data.shape[0], dtype=torch.long, device=data.device)

    for _ in range(max_iter):
        distances = torch.cdist(data, centroids)
        new_assignments = distances.argmin(dim=1)

        new_centroids = centroids.clone()
        for k in range(num_clusters):
            mask = new_assignments == k
            if mask.any():
                new_centroids[k] = data[mask].mean(dim=0)

        shift = torch.norm(new_centroids - centroids, dim=1).max()
        centroids = new_centroids
        assignments = new_assignments
        if shift.item() < tol:
            break

    return centroids, assignments


def fit_kmeans_cluster(
    observed_samples: torch.Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Cluster:
    """Estimate a Cluster object from observed samples.

    Args:
        observed_samples: Tensor of shape (n, q).
        num_clusters: Number of clusters.
    """
    centroids, assignments = run_kmeans(
        data=observed_samples,
        num_clusters=num_clusters,
        max_iter=max_iter,
        tol=tol,
    )
    counts = torch.bincount(assignments, minlength=num_clusters).to(observed_samples.dtype)
    weights = counts / counts.sum()
    return Cluster(centroids=centroids, weights=weights)


def fit_kmeans_clusters_per_input(
    observed_samples: torch.Tensor,
    num_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> list[Cluster]:
    """Estimate one Cluster object per input from shape (m, n, q).

    Args:
        observed_samples: Tensor of shape (m, n, q).
        num_clusters: Number of clusters for each input.
    """
    if observed_samples.ndim != 3:
        raise ValueError("observed_samples must have shape (m, n, q)")

    m = observed_samples.shape[0]
    clusters: list[Cluster] = []
    for i in range(m):
        clusters.append(
            fit_kmeans_cluster(
                observed_samples=observed_samples[i],
                num_clusters=num_clusters,
                max_iter=max_iter,
                tol=tol,
            )
        )
    return clusters
