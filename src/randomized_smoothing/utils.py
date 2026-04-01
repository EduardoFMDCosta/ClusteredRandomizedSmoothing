import torch

def _validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 0.5):
        raise ValueError("Trimming parameter alpha must satisfy 0 <= alpha < 0.5")
    else:
        return alpha
    
def _validate_cluster(cluster) -> None:
    if cluster is None:
        raise ValueError("Cluster must be provided for ClusterizedAlphaTrimmingSmoothing")
    else:
        return cluster
    
def _run_kmeans(
        data: torch.Tensor,
        num_clusters: int,
        max_iter: int,
        tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n = data.shape[0]

    generator = torch.Generator()
    generator.manual_seed(0)
    init_idx = torch.randperm(n, generator=generator)[:num_clusters]
    centroids = data[init_idx].clone()

    assignments = torch.zeros(n, dtype=torch.long)
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

    final_distances = torch.cdist(data, centroids)
    sq_min_dist = final_distances.gather(1, assignments.unsqueeze(1)).squeeze(1).pow(2)
    inertia = sq_min_dist.sum()
    return centroids, assignments, inertia


def _select_num_clusters_elbow(
        data: torch.Tensor,
        max_k: int,
        max_iter: int,
        tol: float,
) -> int:
    n = data.shape[0]
    k_max = min(max_k, n)
    if k_max <= 1:
        return 1

    inertias: list[torch.Tensor] = []
    for k in range(1, k_max + 1):
        _, _, inertia_k = _run_kmeans(data, k, max_iter=max_iter, tol=tol)
        inertias.append(inertia_k)

    curve = torch.stack(inertias)
    x = torch.arange(1, k_max + 1, device=data.device, dtype=curve.dtype)
    x0 = x[0]
    y0 = curve[0]
    x1 = x[-1]
    y1 = curve[-1]

    denominator = torch.sqrt((y1 - y0).pow(2) + (x1 - x0).pow(2))
    if denominator.item() == 0.0:
        return 1

    distances = torch.abs((y1 - y0) * x - (x1 - x0) * curve + x1 * y0 - y1 * x0) / denominator
    return int(torch.argmax(distances).item()) + 1


def compute_probability_for_robust_radius(p: torch.Tensor, radius: float, sigma: float) -> float:
    z = torch.special.ndtri(p)
    shifted = z - radius / sigma
    value = 0.5 * (1.0 + torch.erf(shifted / torch.sqrt(torch.tensor(2.0))))

    if torch.any(value < 0):
        raise RuntimeError(f"Desired radius cannot be robustified. Please select a smaller radius or increase sigma.")

    return value.min().item()

