import math
import torch


# -----------------------------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------------------------
def _validate_alpha(alpha: float):
    if not (0.0 <= alpha < 0.5):
        raise ValueError("Trimming parameter alpha must satisfy 0 <= alpha < 0.5")
    else:
        return alpha
    
def _validate_cluster(cluster):
    if cluster is None:
        raise ValueError("Cluster must be provided for ClusterizedAlphaTrimmingSmoothing")
    else:
        return cluster


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


# -----------------------------------------------------------------------------
# Robustness certification utilities
# -----------------------------------------------------------------------------
def _compute_probability_given_radius(p: torch.Tensor, radius: float, sigma: float) -> float:
    p_tensor = torch.as_tensor(p)
    z = torch.special.ndtri(p_tensor)
    shifted = z - radius / sigma
    probs = 0.5 * (1.0 + torch.erf(shifted / math.sqrt(2.0)))

    if torch.any(probs < 0):
        raise RuntimeError(f"Desired radius cannot be robustified. Please select a smaller radius or increase sigma.")

    if p_tensor.ndim == 2 and p_tensor.shape[1] > 1:
        if p_tensor.shape[1] % 2 != 0:
            raise ValueError("Clustered probabilities must have shape (m, 2c)")

        num_regions = p_tensor.shape[1] // 2
        return torch.cat([probs[:, :num_regions], p_tensor[:, num_regions:]], dim=1)

    return probs.min().item()


def _binomial(k: torch.Tensor, N: int, p: float) -> torch.Tensor:
    k = k.to(dtype=torch.get_default_dtype())
    N = torch.tensor(float(N), dtype=k.dtype, device=k.device)
    p = torch.tensor(p, dtype=k.dtype, device=k.device)
    one = torch.tensor(1.0, dtype=k.dtype, device=k.device)

    log_combination = (
        torch.lgamma(N + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(N - k + 1.0)
    )
    log_prob = log_combination + k * torch.log(one - p) + (N - k) * torch.log(p)
    return torch.exp(log_prob)


def _binomial_cumulative_probability(from_idx: int, to_idx: int, p: float) -> float:
    k = torch.arange(from_idx, to_idx + 1)
    return _binomial(k=k, N=to_idx, p=p).sum().item()


def _alpha_trimming_lower_bound(p: float, num_samples: int, alpha: float) -> float:
    alphaN = int(alpha * num_samples)
    return _binomial_cumulative_probability(from_idx=0, to_idx=alphaN, p=p)


def _clusterized_alpha_trimming_lower_bound(p: torch.Tensor, num_samples: int, alpha: float, weights: torch.Tensor) -> torch.Tensor:
    pass
