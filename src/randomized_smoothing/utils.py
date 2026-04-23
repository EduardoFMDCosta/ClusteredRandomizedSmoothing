import torch


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


def _assign_to_nearest_centroid(data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    diff = data.unsqueeze(1) - centroids.unsqueeze(0)
    squared_distances = (diff * diff).sum(dim=-1)
    return squared_distances.argmin(dim=1)


def _run_dbscan(
    data: torch.Tensor,
    eps: float,
    min_samples: int,
) -> torch.Tensor:
    """Return DBSCAN labels for one row of samples.

    Labels are contiguous cluster ids ``0..k-1`` and ``-1`` for noise.
    """
    n = data.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=data.device)

    eps2 = float(eps) * float(eps)
    diff = data.unsqueeze(1) - data.unsqueeze(0)
    squared_distances = (diff * diff).sum(dim=-1)
    neighbor_mask = squared_distances <= eps2

    unassigned = -2
    noise = -1
    labels = torch.full((n,), unassigned, dtype=torch.long, device=data.device)
    cluster_id = 0

    for i in range(n):
        if labels[i].item() != unassigned:
            continue

        neighbors = torch.where(neighbor_mask[i])[0]
        if int(neighbors.numel()) < min_samples:
            labels[i] = noise
            continue

        labels[i] = cluster_id
        seeds = neighbors.tolist()
        in_seeds = torch.zeros(n, dtype=torch.bool, device=data.device)
        in_seeds[neighbors] = True

        idx = 0
        while idx < len(seeds):
            q = seeds[idx]
            idx += 1

            if labels[q].item() == noise:
                labels[q] = cluster_id
            if labels[q].item() != unassigned:
                continue

            labels[q] = cluster_id
            q_neighbors = torch.where(neighbor_mask[q])[0]
            if int(q_neighbors.numel()) >= min_samples:
                candidates = q_neighbors[~in_seeds[q_neighbors]]
                if candidates.numel() > 0:
                    in_seeds[candidates] = True
                    seeds.extend(candidates.tolist())

        cluster_id += 1

    labels[labels == unassigned] = noise
    return labels


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

    reduced_centroids = _compute_centroids_from_assignments(data, assignments, num_clusters)[keep]
    reduced_assignments = _assign_to_nearest_centroid(data, reduced_centroids)
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
    noise_mask = labels == -1

    if (~noise_mask).sum().item() == 0:
        assignments = torch.zeros(n, dtype=torch.long, device=data.device)
        centroids = data.mean(dim=0, keepdim=True)
        return centroids, assignments, 1

    core_labels = labels[~noise_mask]
    num_clusters = int(core_labels.max().item()) + 1
    assignments = labels.clone()

    centroids = _compute_centroids_from_assignments(
        data=data[~noise_mask],
        assignments=core_labels,
        num_clusters=num_clusters,
    )

    if noise_mask.any():
        assignments[noise_mask] = _assign_to_nearest_centroid(data[noise_mask], centroids)

    return _merge_to_max_clusters(
        data=data,
        assignments=assignments,
        max_num_clusters=max_num_clusters,
    )
