import numpy as np
import torch
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture


# -----------------------------------------------------------------------------
# Clustering utilities
# -----------------------------------------------------------------------------
def _compute_centroids_from_assignments(
    data: torch.Tensor,
    assignments: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    centroids = torch.empty((num_clusters, data.shape[1]), dtype=data.dtype, device=data.device)
    for j in range(num_clusters):
        cluster_points = data[assignments == j]
        centroids[j] = cluster_points.mean(dim=0)
    return centroids


def _run_dbscan(
    data: torch.Tensor,
    eps: float,
    min_samples: int,
) -> torch.Tensor:
    """Return DBSCAN labels for one row of samples via :class:`sklearn.cluster.DBSCAN`.

    Labels are contiguous cluster ids ``0..k-1`` and ``-1`` for noise, matching
    sklearn's convention.
    """
    n = data.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=data.device)

    data_np = data.detach().cpu().numpy()
    sk_labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(data_np)
    return torch.as_tensor(sk_labels, dtype=torch.long, device=data.device)


def _merge_to_max_clusters(
    data: torch.Tensor,
    assignments: torch.Tensor,
    max_num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_clusters = int(assignments.max().item()) + 1
    if num_clusters <= max_num_clusters:
        centroids = _compute_centroids_from_assignments(data, assignments, num_clusters)
        return centroids, assignments, num_clusters

    counts = torch.bincount(assignments, minlength=num_clusters)
    keep = torch.topk(counts, k=max_num_clusters, largest=True, sorted=False).indices
    keep = keep.sort().values

    # Map kept old cluster ids → new contiguous ids 0..max_num_clusters-1.
    new_id = torch.full((num_clusters,), -1, dtype=torch.long, device=data.device)
    new_id[keep] = torch.arange(max_num_clusters, dtype=torch.long, device=data.device)

    keep_mask = torch.zeros(num_clusters, dtype=torch.bool, device=data.device)
    keep_mask[keep] = True
    in_kept = keep_mask[assignments]

    reduced_assignments = torch.empty_like(assignments)
    reduced_assignments[in_kept] = new_id[assignments[in_kept]]

    if (~in_kept).any():
        kept_pts = data[in_kept]
        kept_new_ids = reduced_assignments[in_kept]
        # Single-link reassignment: each dropped-cluster point joins the
        # cluster of its nearest kept-cluster sample.
        distances = torch.cdist(data[~in_kept], kept_pts, p=2)
        nearest_idx = distances.argmin(dim=1)
        reduced_assignments[~in_kept] = kept_new_ids[nearest_idx]

    reduced_centroids = _compute_centroids_from_assignments(
        data,
        reduced_assignments,
        max_num_clusters,
    )
    return reduced_centroids, reduced_assignments, max_num_clusters


def _select_num_clusters(
    data: torch.Tensor,
    max_num_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cluster one row with DBSCAN and cap clusters to ``max_num_clusters``.

    Returns
    -------
    centroids   : (k_selected, d)
    assignments : (n,)
    k_selected  : int
    """
    n = data.shape[0]
    if n == 0:
        raise ValueError("data must contain at least one sample")

    if max_num_clusters == 1:
        assignments = torch.zeros(n, dtype=torch.long, device=data.device)
        centroids = data.mean(dim=0, keepdim=True)
        return centroids, assignments, 1

    labels = _run_dbscan(data=data, eps=dbscan_eps, min_samples=dbscan_min_samples)
    return _finalize_density_labels(
        data=data,
        labels=labels,
        max_num_clusters=max_num_clusters,
    )


def _run_hdbscan(
    data: torch.Tensor,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    cluster_selection_method: str,
) -> torch.Tensor:
    """Return HDBSCAN labels for one row of samples via :class:`sklearn.cluster.HDBSCAN`.

    Labels match sklearn's convention: contiguous cluster ids ``0..k-1`` and
    ``-1`` for noise.
    """
    n = data.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=data.device)

    data_np = data.detach().cpu().numpy()
    sk_labels = HDBSCAN(
        allow_single_cluster=True,
        min_cluster_size=int(min_cluster_size),
        alpha=np.sqrt(2),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=str(cluster_selection_method),
        copy=True,
    ).fit_predict(data_np)
    return torch.as_tensor(sk_labels, dtype=torch.long, device=data.device)


def _select_num_clusters_hdbscan(
    data: torch.Tensor,
    max_num_clusters: int,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
    hdbscan_cluster_selection_epsilon: float,
    hdbscan_cluster_selection_method: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cluster one row with HDBSCAN and cap clusters to ``max_num_clusters``."""
    n = data.shape[0]
    if n == 0:
        raise ValueError("data must contain at least one sample")

    if max_num_clusters == 1:
        assignments = torch.zeros(n, dtype=torch.long, device=data.device)
        centroids = data.mean(dim=0, keepdim=True)
        return centroids, assignments, 1

    labels = _run_hdbscan(
        data=data,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        cluster_selection_method=hdbscan_cluster_selection_method,
    )
    return _finalize_density_labels(
        data=data,
        labels=labels,
        max_num_clusters=max_num_clusters,
    )


def _finalize_density_labels(
    data: torch.Tensor,
    labels: torch.Tensor,
    max_num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Shared post-processing for density-based clustering.

    Reassigns each noise point to the cluster of its nearest non-noise sample
    (single-linkage attribution), then caps the number of clusters at
    ``max_num_clusters`` via :func:`_merge_to_max_clusters`. Falls back to a
    single cluster if every point was labelled noise.
    """
    n = data.shape[0]
    noise_mask = labels == -1

    if (~noise_mask).sum().item() == 0:
        assignments = torch.zeros(n, dtype=torch.long, device=data.device)
        centroids = data.mean(dim=0, keepdim=True)
        return centroids, assignments, 1

    assignments = labels.clone()

    if noise_mask.any():
        core_pts = data[~noise_mask]
        core_labels = labels[~noise_mask]
        # Single-link reassignment: nearest non-noise sample (rather than
        # nearest centroid) decides which cluster a noise point joins.
        distances = torch.cdist(data[noise_mask], core_pts, p=2)
        nearest_core_idx = distances.argmin(dim=1)
        assignments[noise_mask] = core_labels[nearest_core_idx]

    return _merge_to_max_clusters(
        data=data,
        assignments=assignments,
        max_num_clusters=max_num_clusters,
    )
