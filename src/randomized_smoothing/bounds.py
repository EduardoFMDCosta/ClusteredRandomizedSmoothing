from __future__ import annotations

import math

import torch


def _validate_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must satisfy 0 <= alpha < 1")


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


def alpha_trimming_bound(
    p: float,
    *,
    alpha: float,
    n: int,
    p_check: float | None = None,
) -> float:
    """Compute the global alpha-trimming probability bound.

    Computes:
        sum_{j=0}^{floor(alpha n)} C(n, j) (1-p)^j p^(n-j)

    Args:
        p: Base probability in [0, 1].
        alpha: Trimming threshold fraction.
        n: Number of trials N.
        p_check: Optional checked probability value. When provided, this validates p < p_check.

    Returns:
        Bound value as a Python float.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    _validate_probability("p", p)
    _validate_alpha(alpha)
    if p_check is not None:
        _validate_probability("p_check", p_check)
        if not (p < p_check):
            raise ValueError("expected p < p_check")

    j_max = int(math.floor(alpha * n))
    k = torch.arange(0, j_max + 1, dtype=torch.float64)
    success = torch.tensor(1.0 - p, dtype=torch.float64)
    return float(_binom_pmf_terms(k=k, n=n, success_prob=success).sum().item())


def clusterized_alpha_trimming_bound(
    pi: torch.Tensor,
    p_r: torch.Tensor,
    p_v: torch.Tensor,
    *,
    alpha: float,
    n: int,
    validate_pi_sum: bool = True,
) -> torch.Tensor:
    """Compute the clusterized alpha-trimming bound.

    Computes:
        sum_m pi_m sum_{ell=1}^N [
            sum_{j=0}^{floor(alpha ell)} C(ell, j) (1-p_Rm)^j p_Rm^(ell-j)
        ] C(N, ell) (1-p_Vm)^ell p_Vm^(N-ell)

    Args:
        pi: Tensor of shape (M,) with cluster weights.
        p_r: Tensor of shape (M,) with p_{R_m}.
        p_v: Tensor of shape (M,) with p_{V_m}.
        alpha: Trimming threshold fraction.
        n: Number of trials N.
        validate_pi_sum: If True, checks that pi sums to 1.

    Returns:
        Scalar tensor with the weighted bound value.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    _validate_alpha(alpha)

    if pi.ndim != 1 or p_r.ndim != 1 or p_v.ndim != 1:
        raise ValueError("pi, p_r and p_v must all be 1D tensors")
    if not (pi.shape == p_r.shape == p_v.shape):
        raise ValueError("pi, p_r and p_v must have the same shape")

    dtype = torch.promote_types(torch.promote_types(pi.dtype, p_r.dtype), p_v.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float64
    device = pi.device

    pi_t = pi.to(device=device, dtype=dtype)
    p_r_t = p_r.to(device=device, dtype=dtype)
    p_v_t = p_v.to(device=device, dtype=dtype)

    if ((pi_t < 0) | (pi_t > 1)).any():
        raise ValueError("pi must contain probabilities in [0, 1]")
    if ((p_r_t < 0) | (p_r_t > 1)).any():
        raise ValueError("p_r must contain probabilities in [0, 1]")
    if ((p_v_t < 0) | (p_v_t > 1)).any():
        raise ValueError("p_v must contain probabilities in [0, 1]")

    if validate_pi_sum:
        if not torch.isclose(pi_t.sum(), torch.tensor(1.0, device=device, dtype=dtype), atol=1e-6, rtol=0.0):
            raise ValueError("pi must sum to 1")

    m = pi_t.shape[0]
    total_per_m = torch.zeros(m, device=device, dtype=dtype)

    for ell in range(1, n + 1):
        j_max = int(math.floor(alpha * ell))
        j = torch.arange(0, j_max + 1, device=device, dtype=dtype)
        inner = _binom_pmf_terms(k=j.unsqueeze(1), n=ell, success_prob=(1.0 - p_r_t).unsqueeze(0)).sum(dim=0)

        outer = _binom_pmf_terms(
            k=torch.tensor(float(ell), device=device, dtype=dtype),
            n=n,
            success_prob=1.0 - p_v_t,
        )
        total_per_m = total_per_m + inner * outer

    return torch.sum(pi_t * total_per_m)
