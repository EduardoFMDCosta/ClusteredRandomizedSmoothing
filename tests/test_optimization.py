from __future__ import annotations

import numpy as np
import torch

from randomized_smoothing.optimization import (
    compute_total_prob_lower_bound,
    eval_f_m,
    robustness_bound,
)


def test_eval_f_m_matches_single_cluster_robustness_bound() -> None:
    N = 10
    alpha = 0.2
    p_v = 0.6
    p_r = 0.3

    expected = eval_f_m(p=p_v, p_R_m=p_r, N=N, alpha=alpha)

    bound = robustness_bound(
        probs_region=torch.tensor([[p_r]], dtype=torch.float64),
        probs_partition_lb=torch.tensor([[p_v]], dtype=torch.float64),
        probs_partition_ub=torch.tensor([[p_v]], dtype=torch.float64),
        num_samples=N,
        alpha=alpha,
    )
    observed = float(bound.squeeze().item())

    assert np.isclose(observed, expected, atol=1e-10, rtol=1e-8)


def test_compute_total_prob_lower_bound_is_valid() -> None:
    M = 3
    N = 12
    alpha = 0.1
    num_anchor_points = 5

    p_R = np.array([0.05, 0.12, 0.09], dtype=np.float64)
    p_V_lower = np.array([0.10, 0.20, 0.25], dtype=np.float64)
    p_V_upper = np.array([0.65, 0.75, 0.70], dtype=np.float64)

    lower_bound, lambdas, anchor_points = compute_total_prob_lower_bound(
        M=M,
        N=N,
        alpha=alpha,
        p_R=p_R,
        p_V_lower=p_V_lower,
        p_V_upper=p_V_upper,
        num_anchor_points=num_anchor_points,
    )

    assert lambdas.shape == (M, num_anchor_points)
    assert anchor_points.shape == (M, num_anchor_points)
    assert np.all(lambdas >= -1e-9)
    assert np.allclose(lambdas.sum(axis=1), 1.0, atol=1e-7)

    p_from_lambdas = np.sum(lambdas * anchor_points, axis=1)
    effective_lower = np.maximum(p_R, p_V_lower)
    assert np.all(p_from_lambdas >= effective_lower - 1e-8)
    assert np.all(p_from_lambdas <= p_V_upper + 1e-8)
    assert np.isclose(p_from_lambdas.sum(), 1.0, atol=1e-7)

    rng = np.random.default_rng(123)
    residual = 1.0 - effective_lower.sum()
    candidate = effective_lower + residual * rng.dirichlet(np.ones(M))
    if np.any(candidate > p_V_upper):
        capacities = np.maximum(p_V_upper - effective_lower, 0.0)
        candidate = effective_lower + residual * capacities / capacities.sum()

    sampled_value = float(sum(eval_f_m(candidate[m], p_R[m], N, alpha) for m in range(M)))
    assert lower_bound <= sampled_value + 1e-8


def test_total_lp_bound_is_valid_probability() -> None:
    N = 12
    alpha = 0.1

    p_R = np.array([0.05, 0.12, 0.09], dtype=np.float64)
    p_V_lower = np.array([0.10, 0.20, 0.25], dtype=np.float64)
    p_V_upper = np.array([0.65, 0.75, 0.70], dtype=np.float64)

    total_bound, _, _ = compute_total_prob_lower_bound(
        M=3,
        N=N,
        alpha=alpha,
        p_R=p_R,
        p_V_lower=p_V_lower,
        p_V_upper=p_V_upper,
        num_anchor_points=5,
    )

    assert 0.0 <= total_bound <= 1.0