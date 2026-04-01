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


# -----------------------------------------------------------------------------
# Robustness certification utilities
# -----------------------------------------------------------------------------
def _compute_probability_given_radius(p: torch.Tensor, radius: float, sigma: float) -> float:
    z = torch.special.ndtri(p)
    shifted = z - radius / sigma
    probs = 0.5 * (1.0 + torch.erf(shifted / math.sqrt(2.0)))

    if torch.any(probs < 0):
        raise RuntimeError(f"Desired radius cannot be robustified. Please select a smaller radius or increase sigma.")

    return probs.min().item()


def _binom_pmf_terms(k: torch.Tensor, n: int, success_prob: torch.Tensor) -> torch.Tensor:
    """Compute Binomial(n, success_prob) PMF at integer tensor k.

    Args:
        k: Integer tensor of event counts.
        n: Number of trials.
        success_prob: Success probability tensor broadcastable with k.
    """
    n_tensor = torch.tensor(float(n), device=success_prob.device, dtype=success_prob.dtype)
    log_choose = (
        torch.lgamma(n_tensor + 1.0)
        - torch.lgamma(k.to(success_prob.dtype) + 1.0)
        - torch.lgamma(n_tensor - k.to(success_prob.dtype) + 1.0)
    )

    eps = torch.finfo(success_prob.dtype).tiny
    p = success_prob.clamp(min=eps, max=1.0 - eps)
    log_prob = log_choose + k * torch.log(p) + (n - k) * torch.log1p(-p)
    return torch.exp(log_prob)



def clopper_pearson_tail_probability(*, z: int, m: int, p: float) -> float:
    """Compute the binomial tail used in the Clopper-Pearson construction.

    Computes:
        sum_{j=z}^{m} C(m, j) p^j (1-p)^(m-j)

    Args:
        z: Success threshold Z.
        m: Number of Bernoulli trials M.
        p: Probability value in [0, 1].

    Returns:
        Tail probability as a Python float.
    """
    k = torch.arange(z, m + 1, dtype=torch.float64)
    success = torch.tensor(p, dtype=torch.float64)
    return float(_binom_pmf_terms(k=k, n=m, success_prob=success).sum().item())


def lower_bound_clopper_pearson(*, p: float, beta: float, num_samples: int) -> float:
    """Compute the one-sided Clopper-Pearson lower confidence bound.

    Args:
        p: Empirical success rate in [0, 1].
        beta: Confidence parameter in [0, 1].
        num_samples: Number of Bernoulli samples used to estimate p.

    Returns:
        Lower confidence bound as a float.
    """
    z = int(num_samples * p)
    if z == 0 or beta == 0.0:
        return 0.0
    if beta == 1.0:
        return 1.0

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tail_mid = clopper_pearson_tail_probability(z=z, m=num_samples, p=mid)
        if tail_mid >= beta:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)

def _bound_rekavandi_et_al_2024(p: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    j_max = int(math.floor(alpha * num_samples))
    k = torch.arange(0, j_max + 1, dtype=torch.float64)
    success = torch.as_tensor(1.0 - p, dtype=torch.float64)
    certified_probs = _binom_pmf_terms(k=k, n=num_samples, success_prob=success).sum()

    return certified_probs

def _bound_ours(p: torch.Tensor, num_samples: int, alpha: float, weights: torch.Tensor) -> torch.Tensor:
    m = weights.shape[0]
    total_per_m = torch.zeros(m)

    p_regions = p[:m]
    p_partition = p[m:]

    for ell in range(1, num_samples + 1):
        j_max = int(math.floor(alpha * ell))
        j = torch.arange(0, j_max + 1)
        inner = _binom_pmf_terms(k=j.unsqueeze(1), n=ell, success_prob=(1.0 - p_regions).unsqueeze(0)).sum(dim=0)

        outer = _binom_pmf_terms(
            k=torch.tensor(float(ell)),
            n=num_samples,
            success_prob=1.0 - p_partition,
        )
        total_per_m = total_per_m + inner * outer

    certified_probs = (weights * total_per_m)

    return certified_probs
