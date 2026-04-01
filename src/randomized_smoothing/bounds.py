from __future__ import annotations

import math
import torch


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


def robust_certification(
    p: float,
    alpha: float,
    n: int,
) -> float:
    j_max = int(math.floor(alpha * n))
    k = torch.arange(0, j_max + 1, dtype=torch.float64)
    success = torch.tensor(1.0 - p, dtype=torch.float64)
    return _binom_pmf_terms(k=k, n=n, success_prob=success).sum().item()


def clusterized_robust_certification(
    weights: torch.Tensor,
    p_regions: torch.Tensor,
    p_partition: torch.Tensor,
    alpha: float,
    n: int,
) -> torch.Tensor:
    m = weights.shape[0]
    total_per_m = torch.zeros(m)

    for ell in range(1, n + 1):
        j_max = int(math.floor(alpha * ell))
        j = torch.arange(0, j_max + 1)
        inner = _binom_pmf_terms(k=j.unsqueeze(1), n=ell, success_prob=(1.0 - p_regions).unsqueeze(0)).sum(dim=0)

        outer = _binom_pmf_terms(
            k=torch.tensor(float(ell)),
            n=n,
            success_prob=1.0 - p_partition,
        )
        total_per_m = total_per_m + inner * outer

    return torch.sum(weights * total_per_m).item()


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


def lower_bound_clopper_pearson(*, beta: float, p: float, num_samples: int) -> float:
    """Compute the Clopper-Pearson lower bound p_check from the lemma equation.

    Given ``M = num_samples`` and ``Z = int(M * p)``, finds ``p_check in [0, 1]``
    such that:

        beta = sum_{j=Z}^{M} C(M, j) p_check^j (1-p_check)^(M-j)

    using a monotone bisection solve over p_check.

    Args:
        beta: Confidence parameter in [0, 1].
        p: Empirical success fraction in [0, 1].
        num_samples: Number of samples M (> 0).

    Returns:
        The lower-bound value p_check.
    """
    m = int(num_samples)
    z = int(m * p)

    if z == 0:
        if beta != 1.0:
            raise ValueError("No solution for z=0 unless beta=1")
        return 0.0

    if beta == 0.0:
        return 0.0
    if beta == 1.0:
        return 1.0

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tail_mid = clopper_pearson_tail_probability(z=z, m=m, p=mid)
        if tail_mid >= beta:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)