from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
import torch


def log_combination(N: torch.Tensor | int | float, k: torch.Tensor | int | float) -> torch.Tensor:
    if isinstance(N, torch.Tensor):
        device = N.device
    elif isinstance(k, torch.Tensor):
        device = k.device
    else:
        device = None

    N_t = torch.as_tensor(N, dtype=torch.float64, device=device)
    k_t = torch.as_tensor(k, dtype=torch.float64, device=device)
    return torch.lgamma(N_t + 1.0) - torch.lgamma(k_t + 1.0) - torch.lgamma(N_t - k_t + 1.0)


def _binomial(
    k: torch.Tensor | int | float,
    N: torch.Tensor | int | float,
    p: torch.Tensor | float,
) -> torch.Tensor:
    p_t = torch.as_tensor(p, dtype=torch.float64)
    k_t = torch.as_tensor(k, dtype=torch.float64, device=p_t.device)
    N_t = torch.as_tensor(N, dtype=torch.float64, device=p_t.device)

    if k_t.ndim == 1 and p_t.ndim > 0:
        p_t = p_t.unsqueeze(-1)
        if N_t.ndim > 0:
            N_t = N_t.unsqueeze(-1)

    one = torch.tensor(1.0, dtype=p_t.dtype, device=p_t.device)
    log_comb = log_combination(N_t, k_t)
    log_prob = log_comb + k_t * torch.log(p_t) + (N_t - k_t) * torch.log(one - p_t)
    return torch.exp(log_prob)


def _binomial_cumulative_probability(
    from_k: int,
    to_k: int,
    p: torch.Tensor | float,
    num_samples: int,
) -> float | torch.Tensor:
    p_t = torch.as_tensor(p, dtype=torch.float64)
    k = torch.arange(from_k, to_k + 1, device=p_t.device, dtype=torch.float64)
    cumulative = _binomial(k=k, N=num_samples, p=p_t).sum(dim=-1)
    if cumulative.ndim == 0:
        return float(cumulative.item())
    return cumulative


def robustness_bound(
    probs_region: torch.Tensor,
    probs_partition_lb: torch.Tensor,
    probs_partition_ub: torch.Tensor,
    num_samples: int,
    alpha: float,
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if probs_region.ndim != 2:
        raise ValueError("probs_region must have shape (nx, num_clusters)")

    output_dtype = probs_region.dtype
    device = probs_region.device
    dtype = torch.float64
    eps = 1e-12

    probs_region = probs_region.to(device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    probs_partition_lb = probs_partition_lb.to(device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    probs_partition_ub = probs_partition_ub.to(device=device, dtype=dtype).clamp(eps, 1.0 - eps)
    ratio = (probs_region / probs_partition_ub).clamp(eps, 1.0 - eps)

    s = torch.arange(1, num_samples + 1, device=device, dtype=dtype)
    j = torch.arange(0, num_samples + 1, device=device, dtype=dtype)
    two_alpha_floor = 2 * torch.floor(alpha * s)
    j_min = (s - two_alpha_floor).to(dtype=torch.long).clamp(min=0)

    s_row = s.view(1, 1, num_samples, 1)
    j_row = j.view(1, 1, 1, num_samples + 1)
    valid_j = (j_row >= j_min.view(1, 1, num_samples, 1)) & (j_row <= s_row)

    log_comb_sj = log_combination(s_row, j_row)

    r = ratio.unsqueeze(-1).unsqueeze(-1)
    log_term_j = (
        log_comb_sj
        + j_row * torch.log(r)
        + (s_row - j_row) * torch.log(1.0 - r)
    )
    term_j = torch.where(valid_j, torch.exp(log_term_j), torch.zeros_like(log_term_j))
    inner_j = term_j.sum(dim=-1)

    term_s_lb = _binomial(k=s, N=float(num_samples), p=probs_partition_lb)
    term_s_ub = _binomial(k=s, N=float(num_samples), p=probs_partition_ub)
    term_s = torch.minimum(term_s_lb, term_s_ub)

    factor_s = (s / float(num_samples)).view(1, 1, num_samples)
    per_cluster = (factor_s * inner_j * term_s).sum(dim=2)
    return per_cluster.to(dtype=output_dtype)


def _validate_vector(name: str, values: np.ndarray | list[float], M: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (M,):
        raise ValueError(f"{name} must have shape ({M},)")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return arr


def eval_f_m(p: float, p_R_m: float, N: int, alpha: float) -> float:
    if N <= 0:
        raise ValueError("N must be positive")
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must satisfy 0 <= alpha < 0.5")

    p = float(np.clip(p, 0.0, 1.0))
    p_R_m = float(np.clip(p_R_m, 0.0, 1.0))

    if p <= 1e-12:
        return 0.0

    ratio = float(np.clip(p_R_m / p, 0.0, 1.0))
    s = np.arange(1, N + 1, dtype=np.int64)
    j = np.arange(0, N + 1, dtype=np.int64)

    comb_Ns = np.asarray([math.comb(N, int(s_i)) for s_i in s], dtype=np.float64)
    outer = (s.astype(np.float64) / float(N)) * comb_Ns * np.power(p, s) * np.power(1.0 - p, N - s)

    inner = np.zeros_like(outer)
    for idx, s_i in enumerate(s):
        j_min = max(int(s_i - 2 * math.floor(alpha * s_i)), 0)
        valid = j[j_min : s_i + 1]
        comb_sj = np.asarray([math.comb(int(s_i), int(j_i)) for j_i in valid], dtype=np.float64)
        inner[idx] = np.sum(comb_sj * np.power(ratio, valid) * np.power(1.0 - ratio, s_i - valid))

    return float(np.sum(outer * inner))


def compute_total_prob_lower_bound(
    M: int,
    N: int,
    alpha: float,
    p_R: np.ndarray | list[float],
    p_V_lower: np.ndarray | list[float],
    p_V_upper: np.ndarray | list[float],
    K: int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    if M <= 0:
        raise ValueError("M must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if K <= 0:
        raise ValueError("K must be positive")

    p_R = _validate_vector("p_R", p_R, M)
    p_V_lower = _validate_vector("p_V_lower", p_V_lower, M)
    p_V_upper = _validate_vector("p_V_upper", p_V_upper, M)

    if np.any(p_V_lower > p_V_upper):
        raise ValueError("p_V_lower must be componentwise <= p_V_upper")

    effective_lower = np.maximum(p_V_lower, p_R)
    if effective_lower.sum() > 1.0 + 1e-9 or p_V_upper.sum() < 1.0 - 1e-9:
        raise ValueError("partition bounds are infeasible with simplex constraints")
    if np.any(effective_lower > p_V_upper + 1e-12):
        raise ValueError("Each p_V upper bound must be >= max(p_V lower, p_R)")

    anchor_points = np.vstack(
        [np.linspace(effective_lower[m], p_V_upper[m], K, dtype=np.float64) for m in range(M)]
    )
    f_values = np.vstack(
        [
            np.asarray([eval_f_m(p=anchor_points[m, k], p_R_m=p_R[m], N=N, alpha=alpha) for k in range(K)], dtype=np.float64)
            for m in range(M)
        ]
    )

    lambdas = cp.Variable((M, K), nonneg=True)
    p_m = cp.Variable(M)

    constraints = [
        cp.sum(lambdas, axis=1) == 1,
        cp.sum(cp.multiply(lambdas, anchor_points), axis=1) == p_m,
        p_m >= effective_lower,
        p_m <= p_V_upper,
        cp.sum(p_m) == 1,
    ]

    problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(lambdas, f_values))), constraints)
    status = None
    for solver in (cp.CLARABEL, cp.SCS):
        try:
            problem.solve(solver=solver, warm_start=True)
        except cp.SolverError:
            continue
        status = problem.status
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break

    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"total probability LP ended with status {problem.status}")

    lower_bound = float(problem.value)
    lambda_opt = np.asarray(lambdas.value, dtype=np.float64)

    # Short sanity check: lower bound should be below any feasible evaluation.
    rng = np.random.default_rng(0)
    residual = 1.0 - float(effective_lower.sum())
    if residual <= 1e-12:
        random_p = effective_lower
    else:
        for _ in range(256):
            candidate = effective_lower + residual * rng.dirichlet(np.ones(M))
            if np.all(candidate <= p_V_upper + 1e-10):
                random_p = candidate
                break
        else:
            capacities = np.maximum(p_V_upper - effective_lower, 0.0)
            random_p = effective_lower + residual * capacities / capacities.sum()

    sampled_value = float(sum(eval_f_m(random_p[m], p_R[m], N, alpha) for m in range(M)))
    assert lower_bound <= sampled_value + 1e-8

    return lower_bound, lambda_opt, anchor_points