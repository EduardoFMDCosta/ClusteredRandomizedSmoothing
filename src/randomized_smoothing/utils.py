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
    k = k.to(dtype=torch.get_default_dtype())
    N = torch.tensor(float(N), dtype=k.dtype, device=k.device)
    p = torch.tensor(p, dtype=k.dtype, device=k.device)
    one = torch.tensor(1.0, dtype=k.dtype, device=k.device)

    log_combination = log_combination(N, k)
    log_prob = log_combination + k * torch.log(one - p) + (N - k) * torch.log(p)
    return torch.exp(log_prob)


def _binomial_cumulative_probability(from_idx: int, to_idx: int, p: float) -> float:
    k = torch.arange(from_idx, to_idx + 1)
    return _binomial(k=k, N=to_idx, p=p).sum().item()