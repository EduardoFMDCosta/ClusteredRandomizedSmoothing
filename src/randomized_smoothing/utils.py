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
def _compute_probability_given_radius(p: torch.Tensor, radius: float, sigma: float) -> torch.Tensor:
    z = torch.special.ndtri(p)
    shifted = z - radius / sigma
    probs = 0.5 * (1.0 + torch.erf(shifted / math.sqrt(2.0)))

    return probs


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


def _alpha_trimming_lower_bound(p: float, num_samples: int, alpha: float) -> float:
    alphaN = int(alpha * num_samples)
    return _binomial_cumulative_probability(from_idx=0, to_idx=alphaN, p=p)


def _compute_per_region_factor(p: float, p_clusters: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    device = p_clusters.device
    dtype = torch.float64

    p_region = torch.tensor(float(p), dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)

    p_v = p_clusters.to(dtype=dtype)

    log_p_region = torch.log(p_region)
    log_one_minus_p_region = torch.log(one - p_region)
    log_p_v = torch.log(p_v)
    log_one_minus_p_v = torch.log(one - p_v)

    result = torch.zeros_like(p_v)

    for ell in range(1, num_samples + 1):
        ell_t = torch.tensor(float(ell), dtype=dtype, device=device)

        inner_sum = torch.tensor(0.0, dtype=dtype, device=device)
        max_j = int(alpha * ell)
        for j in range(0, max_j + 1):
            j_t = torch.tensor(float(j), dtype=dtype, device=device)

            log_choose_ell_j = (
                torch.lgamma(ell_t + 1.0)
                - torch.lgamma(j_t + 1.0)
                - torch.lgamma(ell_t - j_t + 1.0)
            )
            log_term_j = (
                log_choose_ell_j
                + j_t * log_one_minus_p_region
                + (ell_t - j_t) * log_p_region
            )
            inner_sum = inner_sum + torch.exp(log_term_j)

        log_choose_n_ell = (
            torch.lgamma(torch.tensor(float(num_samples), dtype=dtype, device=device) + 1.0)
            - torch.lgamma(ell_t + 1.0)
            - torch.lgamma(torch.tensor(float(num_samples - ell), dtype=dtype, device=device) + 1.0)
        )

        cluster_term = torch.exp(
            log_choose_n_ell
            + ell_t * log_one_minus_p_v
            + (num_samples - ell_t) * log_p_v
        )
        result = result + inner_sum * cluster_term
    return result


def _clusterized_alpha_trimming_lower_bound(p: float, p_clusters: torch.Tensor, num_samples: int, alpha: float, weights: torch.Tensor) -> torch.Tensor:
    result = _compute_per_region_factor(p=p, p_clusters=p_clusters, num_samples=num_samples, alpha=alpha)
    return (weights * result).to(p_clusters.dtype)