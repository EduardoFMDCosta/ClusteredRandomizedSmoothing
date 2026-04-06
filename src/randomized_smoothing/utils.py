import math
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------------------------
def _validate_alpha(alpha: float):
    if not (0.0 <= alpha < 0.5):
        raise ValueError("Trimming parameter alpha must satisfy 0 <= alpha < 0.5")
    else:
        return True


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


# -----------------------------------------------------------------------------
# Robustness certification utilities
# -----------------------------------------------------------------------------
def log_combination(N: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(N + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(N - k + 1.0)

def _binomial(k: torch.Tensor, N: int, p: float) -> torch.Tensor:
    k = k.to(dtype=torch.float64)
    N = torch.tensor(float(N), dtype=k.dtype, device=k.device)
    p = torch.tensor(p, dtype=k.dtype, device=k.device)
    one = torch.tensor(1.0, dtype=k.dtype, device=k.device)

    log_comb = log_combination(N, k)
    log_prob = log_comb + k * torch.log(p) + (N - k) * torch.log(one - p)
    return torch.exp(log_prob)


def _binomial_cumulative_probability(from_k: int, to_k: int, p: float, num_samples: int) -> float:
    k = torch.arange(from_k, to_k + 1)
    return _binomial(k=k, N=num_samples, p=p).sum().item()


def robustness_bound(
    probs_region: torch.Tensor,
    probs_partition: torch.Tensor,
    num_samples: int,
    alpha: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if probs_region.ndim != 2:
        raise ValueError("probs_region must have shape (nx, num_clusters)")

    nx, num_clusters = probs_region.shape
    if weights.ndim != 2 or weights.shape[0] != nx:
        raise ValueError("weights must have shape (nx, num_clusters)")

    if probs_partition.ndim != 2 or probs_partition.shape != weights.shape:
        raise ValueError("probs_partition must have the same shape as weights")

    if weights.shape[1] != num_clusters:
        raise ValueError("probs_region and weights must have the same shape")

    output_dtype = probs_region.dtype
    device = probs_region.device
    dtype = torch.float64
    eps = 1e-12

    probs_region = probs_region.to(device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    probs_partition = probs_partition.to(device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    weights = weights.to(device=device, dtype=dtype)

    l = torch.arange(1, num_samples + 1, device=device, dtype=dtype)
    j = torch.arange(0, num_samples + 1, device=device, dtype=dtype)
    j_max = torch.floor(alpha * l).to(dtype=torch.long)

    l_row = l.view(1, num_samples, 1)
    j_row = j.view(1, 1, num_samples + 1)
    valid_j = j_row <= j_max.view(1, num_samples, 1)

    log_comb_lj = (
        torch.lgamma(l_row + 1.0)
        - torch.lgamma(j_row + 1.0)
        - torch.lgamma(l_row - j_row + 1.0)
    )

    c = probs_region.unsqueeze(-1).unsqueeze(-1)
    log_term_j = (
        log_comb_lj
        + j_row * torch.log(1.0 - c)
        + (l_row - j_row) * torch.log(c)
    )
    term_j = torch.where(valid_j, torch.exp(log_term_j), torch.zeros_like(log_term_j))
    inner_j = term_j.sum(dim=-1)

    log_comb_nl = (
        torch.lgamma(torch.tensor(float(num_samples), device=device, dtype=dtype) + 1.0)
        - torch.lgamma(l + 1.0)
        - torch.lgamma(torch.tensor(float(num_samples), device=device, dtype=dtype) - l + 1.0)
    )

    p = probs_partition.unsqueeze(-1)
    log_term_l = (
        log_comb_nl.view(1, 1, num_samples)
        + l.view(1, 1, num_samples) * torch.log(1.0 - p)
        + (num_samples - l).view(1, 1, num_samples) * torch.log(p)
    )
    term_l = torch.exp(log_term_l)

    per_cluster = (inner_j * term_l).sum(dim=2)
    return (weights * per_cluster).to(dtype=output_dtype)