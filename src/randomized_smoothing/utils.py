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
    log_combination = (
        torch.lgamma(N + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(N - k + 1.0)
    )
    log_prob = log_combination + k * torch.log(1 - p) + (N - k) * torch.log(p)
    return torch.exp(log_prob)


def _binomial_tail_probability(z: int, N: int, p: float) -> float:
    k = torch.arange(z, N + 1)
    return _binomial(k=k, N=N, p=p).sum().item()







def _bound_rekavandi_et_al_2024(p: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    j_max = int(math.floor(alpha * num_samples))
    k = torch.arange(0, j_max + 1, dtype=torch.float64)
    success = torch.as_tensor(1.0 - p, dtype=torch.float64)
    certified_probs = _binom_pmf_terms(k=k, n=num_samples, success_prob=success).sum()

    return certified_probs

def _bound_ours_per_region(p: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    if p.ndim == 1:
        m = p.shape[0] // 2
        total_per_m = torch.zeros(m, dtype=p.dtype, device=p.device)

        p_regions = p[:m]
        p_partition = p[m:]

        for ell in range(1, num_samples + 1):
            j_max = int(math.floor(alpha * ell))
            j = torch.arange(0, j_max + 1, device=p.device)
            inner = _binom_pmf_terms(k=j.unsqueeze(1), n=ell, success_prob=(1.0 - p_regions).unsqueeze(0)).sum(dim=0)

            outer = _binom_pmf_terms(
                k=torch.tensor(float(ell), device=p.device, dtype=p.dtype),
                n=num_samples,
                success_prob=1.0 - p_partition,
            )
            total_per_m = total_per_m + inner * outer

        return total_per_m

    if p.ndim != 2 or p.shape[1] % 2 != 0:
        raise ValueError("Clustered probabilities must have shape (m, 2c)")

    m, two_c = p.shape
    num_regions = two_c // 2
    total_per_m = torch.zeros((m, num_regions), dtype=p.dtype, device=p.device)

    p_regions = p[:, :num_regions]
    p_partition = p[:, num_regions:]

    for ell in range(1, num_samples + 1):
        j_max = int(math.floor(alpha * ell))
        j = torch.arange(0, j_max + 1, device=p.device)
        inner = _binom_pmf_terms(
            k=j.view(-1, 1, 1),
            n=ell,
            success_prob=(1.0 - p_regions).unsqueeze(0),
        ).sum(dim=0)

        outer = _binom_pmf_terms(
            k=torch.tensor(float(ell), device=p.device, dtype=p.dtype),
            n=num_samples,
            success_prob=1.0 - p_partition,
        )
        total_per_m = total_per_m + inner * outer
    return total_per_m

def _bound_ours(p: torch.Tensor, num_samples: int, alpha: float, weights: torch.Tensor) -> torch.Tensor:
    total_per_m = _bound_ours_per_region(p=p, num_samples=num_samples, alpha=alpha)
    certified_probs = (weights * total_per_m)

    return certified_probs
