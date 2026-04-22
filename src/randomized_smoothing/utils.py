import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Clustering utilities
# -----------------------------------------------------------------------------
def _run_kmeans(
        data: torch.Tensor,
        num_clusters: int,
        max_iter: int,
        tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n = data.shape[0]

    generator = torch.Generator(device=data.device)
    generator.manual_seed(0)
    init_idx = torch.randperm(n, generator=generator, device=data.device)[:num_clusters]
    centroids = data[init_idx].clone()

    assignments = torch.zeros(n, dtype=torch.long, device=data.device)
    for _ in range(max_iter):
        diff = data.unsqueeze(1) - centroids.unsqueeze(0)
        squared_distances = (diff * diff).sum(dim=-1)
        new_assignments = squared_distances.argmin(dim=1)

        new_centroids = centroids.clone()
        one_hot = F.one_hot(new_assignments, num_classes=num_clusters).to(dtype=data.dtype)
        counts = one_hot.sum(dim=0)
        non_empty = counts > 0
        if non_empty.any():
            sums = one_hot.transpose(0, 1) @ data
            new_centroids[non_empty] = sums[non_empty] / counts[non_empty].unsqueeze(1)

        shift = torch.norm(new_centroids - centroids, dim=1).max()
        centroids = new_centroids
        assignments = new_assignments
        if shift.item() < tol:
            break

    final_diff = data.unsqueeze(1) - centroids.unsqueeze(0)
    final_squared_distances = (final_diff * final_diff).sum(dim=-1)
    sq_min_dist = final_squared_distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
    inertia = sq_min_dist.sum()
    return centroids, assignments, inertia


def _select_num_clusters(
        data: torch.Tensor,
        max_num_clusters: int,
        max_iter: int,
        tol: float,
        elbow_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select the optimal number of clusters via the inertia elbow method.

    Runs K-means for k = 1, 2, ... up to min(max_num_clusters, n) and stops as
    soon as the relative inertia improvement of adding one more cluster falls
    below ``elbow_threshold * W_1``, where W_1 is the inertia for k = 1.

    Returns
    -------
    centroids   : (k_selected, d)
    assignments : (n,)
    k_selected  : int
    """
    n = data.shape[0]
    k_max = min(max_num_clusters, n)

    prev_c, prev_a, W_prev = _run_kmeans(data, 1, max_iter, tol)
    if k_max == 1:
        return prev_c, prev_a, 1

    W1 = W_prev.item()

    for k in range(2, k_max + 1):
        c_k, a_k, W_k = _run_kmeans(data, k, max_iter, tol)
        gain = (W_prev.item() - W_k.item()) / W1 if W1 > 0.0 else 0.0
        if gain < elbow_threshold:
            # Adding cluster k does not improve enough; keep k-1 clusters.
            return prev_c, prev_a, k - 1
        prev_c, prev_a = c_k, a_k
        W_prev = W_k

    return prev_c, prev_a, k_max
