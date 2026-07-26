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

    # All arithmetic happens in log-space and the result is collapsed with
    # ``logsumexp`` to keep the bound representable when individual binomial
    # tail terms underflow (e.g. for large N or extreme p / ratio).
    log_r = torch.log(ratio)
    log_1mr = torch.log1p(-ratio)
    log_p_lb = torch.log(probs_partition_lb)
    log_1mp_lb = torch.log1p(-probs_partition_lb)
    log_p_ub = torch.log(probs_partition_ub)
    log_1mp_ub = torch.log1p(-probs_partition_ub)

    N = num_samples
    s = torch.arange(1, N + 1, device=device, dtype=dtype)
    j = torch.arange(0, N + 1, device=device, dtype=dtype)
    alpha_floor = torch.floor(alpha * s)
    j_min = (s - alpha_floor).to(dtype=torch.long).clamp(min=0)

    s_row = s.view(1, 1, N, 1)
    j_row = j.view(1, 1, 1, N + 1)
    valid_j = (j_row >= j_min.view(1, 1, N, 1)) & (j_row <= s_row)

    # log C(s, j) + j * log(r) + (s - j) * log(1 - r)
    log_comb_sj = log_combination(s_row, j_row)
    log_term_j = (
        log_comb_sj
        + j_row * log_r.unsqueeze(-1).unsqueeze(-1)
        + (s_row - j_row) * log_1mr.unsqueeze(-1).unsqueeze(-1)
    )
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    log_term_j = torch.where(valid_j, log_term_j, neg_inf)
    log_inner_j = torch.logsumexp(log_term_j, dim=-1)  # (nx, k, N)

    # log C(N, s) + s * log(p) + (N - s) * log(1 - p), broadcast over (nx, k, N).
    log_comb_Ns = log_combination(float(N), s).view(1, 1, N)
    s_b = s.view(1, 1, N)
    log_term_s_lb = (
        log_comb_Ns
        + s_b * log_p_lb.unsqueeze(-1)
        + (N - s_b) * log_1mp_lb.unsqueeze(-1)
    )
    log_term_s_ub = (
        log_comb_Ns
        + s_b * log_p_ub.unsqueeze(-1)
        + (N - s_b) * log_1mp_ub.unsqueeze(-1)
    )
    # min of two positives in linear space is min of their logs.
    log_term_s = torch.minimum(log_term_s_lb, log_term_s_ub)

    log_factor_s = torch.log(s / float(N)).view(1, 1, N)
    log_per_s = log_factor_s + log_inner_j + log_term_s

    log_per_cluster = torch.logsumexp(log_per_s, dim=-1)  # (nx, k)
    per_cluster = torch.exp(log_per_cluster)
    return per_cluster.to(dtype=output_dtype)


def _validate_vector(name: str, values: np.ndarray | list[float], M: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (M,):
        raise ValueError(f"{name} must have shape ({M},)")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return arr


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return -math.inf
    m = float(np.max(values))
    if not math.isfinite(m):
        return m
    return m + float(np.log(np.exp(values - m).sum()))


def _scaled_log(log_x: float, k: np.ndarray) -> np.ndarray:
    # k * log_x with the convention 0 * (-inf) = 0 (so x^0 = 1 when x = 0).
    k64 = k.astype(np.float64)
    if math.isfinite(log_x):
        return k64 * log_x
    out = np.full_like(k64, log_x)
    out[k64 == 0.0] = 0.0
    return out


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

    # Log-space arithmetic: C(N, s) overflows float64 for N >~ 1029, so this
    # function must not materialize binomial coefficients directly.
    log_p = math.log(p) if p > 0.0 else -math.inf
    log_1mp = math.log1p(-p) if p < 1.0 else -math.inf
    log_r = math.log(ratio) if ratio > 0.0 else -math.inf
    log_1mr = math.log1p(-ratio) if ratio < 1.0 else -math.inf

    log_factorial = np.fromiter(
        (math.lgamma(i + 1.0) for i in range(N + 1)),
        dtype=np.float64,
        count=N + 1,
    )

    s_arr = np.arange(1, N + 1, dtype=np.int64)
    log_comb_Ns = log_factorial[N] - log_factorial[s_arr] - log_factorial[N - s_arr]
    log_outer = (
        np.log(s_arr.astype(np.float64) / float(N))
        + log_comb_Ns
        + _scaled_log(log_p, s_arr)
        + _scaled_log(log_1mp, N - s_arr)
    )

    log_per_s = np.full(N, -math.inf, dtype=np.float64)
    for idx, s_i in enumerate(s_arr):
        s_int = int(s_i)
        j_min = max(s_int - math.floor(alpha * s_int), 0)
        j_arr = np.arange(j_min, s_int + 1, dtype=np.int64)
        log_comb_sj = log_factorial[s_int] - log_factorial[j_arr] - log_factorial[s_int - j_arr]
        log_terms = (
            log_comb_sj
            + _scaled_log(log_r, j_arr)
            + _scaled_log(log_1mr, s_int - j_arr)
        )
        log_per_s[idx] = log_outer[idx] + _logsumexp(log_terms)

    total_log = _logsumexp(log_per_s)
    if not math.isfinite(total_log):
        return 0.0
    return float(math.exp(total_log))


def _estimate_lipschitz_constants(
    N: int,
    alpha: float,
    p_R: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    num_grid_points: int,
) -> np.ndarray:
    """Estimate per-cluster Lipschitz constants by max finite-difference slope."""
    M = p_R.shape[0]
    if num_grid_points < 2:
        raise ValueError("num_grid_points must be at least 2")

    lipschitz = np.zeros(M, dtype=np.float64)
    for m in range(M):
        lo = float(interval_lower[m])
        hi = float(interval_upper[m])
        if hi <= lo + 1e-15:
            continue

        grid = np.linspace(lo, hi, num_grid_points, dtype=np.float64)
        values = np.asarray(
            [eval_f_m(p=float(p), p_R_m=float(p_R[m]), N=N, alpha=alpha) for p in grid],
            dtype=np.float64,
        )
        slopes = np.abs(np.diff(values) / np.diff(grid))
        lipschitz[m] = float(np.max(slopes)) if slopes.size > 0 else 0.0

    return lipschitz


def compute_total_prob_lower_bound(
    M: int,
    N: int,
    alpha: float,
    p_R: np.ndarray | list[float],
    p_V_lower: np.ndarray | list[float],
    p_V_upper: np.ndarray | list[float],
    num_anchor_points: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    if M <= 0:
        raise ValueError("M must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if num_anchor_points <= 0:
        raise ValueError("num_anchor_points must be positive")

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
        [np.linspace(effective_lower[m], p_V_upper[m], num_anchor_points, dtype=np.float64) for m in range(M)]
    )
    f_values = np.vstack(
        [
            np.asarray(
                [eval_f_m(p=anchor_points[m, k], p_R_m=p_R[m], N=N, alpha=alpha) for k in range(num_anchor_points)],
                dtype=np.float64,
            )
            for m in range(M)
        ]
    )

    lambdas = cp.Variable((M, num_anchor_points), nonneg=True)
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

    lower_bound_lp = float(problem.value)
    lambda_opt = np.asarray(lambdas.value, dtype=np.float64)

    # Correction for piecewise-linear LP discretization: subtract an estimated
    # per-cluster Lipschitz envelope term sum_m L_m h_{m,K}.
    if num_anchor_points > 1:
        h_mk = np.max(np.diff(anchor_points, axis=1), axis=1)
    else:
        h_mk = np.zeros(M, dtype=np.float64)

    # Use the same feasible intervals as the LP anchors for each cluster.
    num_grid_points = max(33, min(257, 4 * num_anchor_points + 1))
    lipschitz_estimates = _estimate_lipschitz_constants(
        N=N,
        alpha=alpha,
        p_R=p_R,
        interval_lower=effective_lower,
        interval_upper=p_V_upper,
        num_grid_points=num_grid_points,
    )
    lipschitz_correction_term = float(np.sum(lipschitz_estimates * h_mk))

    lower_bound = max(0.0, lower_bound_lp - lipschitz_correction_term)

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