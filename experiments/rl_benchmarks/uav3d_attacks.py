"""Task-directed adversarial attack primitives for the UAV3D controller.

The threat model perturbs the (normalized) **state** fed to the policy within
an L2 ball ``||delta|| <= radius`` — the same input space the existing
state-space ``GaussianAlphaTrimmingSmoothing`` defense (see
``collect_uav3d_rollouts.py``) smooths over. Rather than maximizing an
abstract output-deviation metric, the attack searches the ball for the
perturbation that most hurts the *mission*: reduces goal progress or steers
into an obstacle, scored via a side-effect-free one-step analytic lookahead
through UAV3D's own (deterministic, linear) dynamics and reward structure.

``UAV3D.step()`` is stateful and consumes the ``np.random`` stream, so it
cannot be called from inside a candidate-scoring loop without corrupting the
real rollout. ``_analytic_one_step_task_cost`` instead reads only pure,
side-effect-free env state (``env.A``/``env.B``/``env.obstacles``/geometry
predicates) to score a candidate action without touching ``env._state``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

try:
    import ot
except ImportError as exc:  # pragma: no cover - import guard for optional runtime dependency
    raise ImportError(
        "POT is required for uav3d_attacks. Install with 'pip install pot'."
    ) from exc

from dynamics import UAV3D
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from train_rl import VariationalPolicy


def _random_ball_points(
    num_points: int,
    dimension: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Uniformly sample points in the closed L2 ball ``||delta|| <= radius``.

    Duplicated verbatim from ``experiments/paper/trajflow_adversarial.py`` to
    avoid a cross-directory import into that paper-figure-reproduction script.
    """
    if radius == 0.0:
        return torch.zeros((1, dimension), device=device, dtype=dtype)
    if num_points <= 1:
        return torch.zeros((1, dimension), device=device, dtype=dtype)

    # Include the clean point (delta=0) and sample the remaining candidates
    # uniformly in the closed ball to search over ||delta|| <= radius.
    vec = torch.randn((num_points - 1, dimension), device=device, dtype=dtype)
    vec = vec / torch.linalg.norm(vec, dim=1, keepdim=True).clamp(min=1e-12)
    scales = torch.rand((num_points - 1, 1), device=device, dtype=dtype)
    scales = radius * scales.pow(1.0 / float(dimension))
    sampled = scales * vec
    origin = torch.zeros((1, dimension), device=device, dtype=dtype)
    return torch.cat((origin, sampled), dim=0)


def _min_obstacle_surface_distance(pos: np.ndarray, obstacles) -> float:
    """L2 distance from ``pos`` to the nearest obstacle AABB (0 if inside)."""
    best = float("inf")
    for x_min, x_max, y_min, y_max, z_min, z_max in obstacles:
        dx = max(x_min - pos[0], 0.0, pos[0] - x_max)
        dy = max(y_min - pos[1], 0.0, pos[1] - y_max)
        dz = max(z_min - pos[2], 0.0, pos[2] - z_max)
        dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        if dist < best:
            best = dist
    return best if best != float("inf") else 0.0


def _analytic_one_step_task_cost(
    env: UAV3D,
    true_state: np.ndarray,
    action: np.ndarray,
    obstacle_margin_weight: float = 0.0,
) -> float:
    """Deterministic (noise-free) one-step lookahead cost, higher = worse.

    Mirrors ``UAV3D.step()``'s reward branches (corridor exit / obstacle
    entry / goal / dense progress shaping), negated, using the env's public
    ``A``/``B`` matrices instead of calling the stateful ``step()``. The
    stochastic process noise term is intentionally omitted: it's zero-mean
    and orthogonal to the attack direction we're searching for.

    ``obstacle_margin_weight`` adds a soft bias toward obstacle proximity even
    when a single step doesn't yet clip an obstacle — real collisions are
    usually a multi-step, compounding event, not something one perturbed step
    always achieves outright.
    """
    action = np.clip(action, env.u_min, env.u_max).astype(np.float32)
    true_state = true_state.astype(np.float32)
    old_pos = np.array([true_state[0], true_state[2], true_state[4]], dtype=np.float32)
    next_state = env.A @ true_state + env.B @ action
    new_pos = np.array([next_state[0], next_state[2], next_state[4]], dtype=np.float32)

    if env._outside_corridor(new_pos):
        return 5.0
    if env._segment_obstacle_entry_t(old_pos, new_pos) is not None:
        return 5.0
    if env._check_goal(new_pos):
        return -10.0

    prev_dist = float(np.linalg.norm(old_pos - env.goal_center))
    new_dist = float(np.linalg.norm(new_pos - env.goal_center))
    progress = prev_dist - new_dist
    cost = -(0.5 * progress - 0.01)
    if obstacle_margin_weight > 0.0:
        cost += obstacle_margin_weight * (-_min_obstacle_surface_distance(new_pos, env.obstacles))
    return float(cost)


def _find_best_delta_raw(
    policy: VariationalPolicy,
    env: UAV3D,
    true_state: np.ndarray,
    x: torch.Tensor,
    radius: float,
    num_candidates: int,
    obstacle_margin_weight: float = 0.0,
) -> torch.Tensor:
    """Search ``||delta|| <= radius`` for the state perturbation that most
    hurts the mission, scoring candidates against the **raw** policy.

    All candidates are batched through one ``policy.deterministic()`` call
    (mode action — stable candidate-to-candidate ranking, no sampling noise
    confounding the search); the actually-executed action at the winning
    ``x + delta`` still comes from the real stochastic ``policy.sample()``.
    """
    dim = x.shape[-1]
    deltas = _random_ball_points(num_candidates, dim, radius, x.device, x.dtype)
    x_batch = x + deltas
    with torch.no_grad():
        actions = policy.deterministic(x_batch)
    actions_np = actions.detach().cpu().numpy()
    costs = np.array([
        _analytic_one_step_task_cost(env, true_state, actions_np[c], obstacle_margin_weight)
        for c in range(actions_np.shape[0])
    ])
    return deltas[int(np.argmax(costs))]


def _find_best_delta_for_smoother(
    smoother: GaussianAlphaTrimmingSmoothing,
    env: UAV3D,
    true_state: np.ndarray,
    radius: float,
    num_candidates: int,
    screen_samples: int,
    obstacle_margin_weight: float = 0.0,
) -> torch.Tensor:
    """Search ``||delta|| <= radius`` for the state perturbation that most
    hurts the mission, scoring candidates against an already-built smoother.

    Exploits ``AlphaTrimmingSmoothing.monte_carlo(z=...)`` reusing the
    smoother's cluster (built once, around the clean anchor) — cost per
    candidate is only ``screen_samples`` policy calls, not a full
    re-clustering.
    """
    x = smoother.x
    dim = x.shape[-1]
    deltas = _random_ball_points(num_candidates, dim, radius, x.device, x.dtype)
    best_idx, best_cost = 0, float("-inf")
    for c in range(deltas.shape[0]):
        draws = smoother.monte_carlo(num_simulations=screen_samples, z=x + deltas[c : c + 1])[0]
        draws_np = draws.detach().cpu().numpy()
        cost = float(np.mean([
            _analytic_one_step_task_cost(env, true_state, draws_np[k], obstacle_margin_weight)
            for k in range(draws_np.shape[0])
        ]))
        if cost > best_cost:
            best_cost, best_idx = cost, c
    return deltas[best_idx]


def _compute_sinkhorn_wasserstein_2_distance(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    reg: float,
    ot_method: str,
) -> float:
    """Compute empirical 2-Wasserstein distance using a POT solver.

    Duplicated verbatim from ``experiments/paper/trajflow_adversarial.py``.
    """
    if x_samples.ndim != 2 or y_samples.ndim != 2:
        raise ValueError("Both sample sets must be 2-D arrays of shape (n, d).")
    if x_samples.shape[1] != y_samples.shape[1]:
        raise ValueError(
            "Both sample sets must have the same feature dimension; "
            f"got {x_samples.shape[1]} and {y_samples.shape[1]}"
        )
    if reg <= 0.0 and ot_method != "emd":
        raise ValueError("sinkhorn regularization (reg) must be positive")

    n_x = x_samples.shape[0]
    n_y = y_samples.shape[0]
    a = np.full(n_x, 1.0 / n_x, dtype=np.float64)
    b = np.full(n_y, 1.0 / n_y, dtype=np.float64)
    cost_matrix = ot.dist(
        x_samples.astype(np.float64, copy=False),
        y_samples.astype(np.float64, copy=False),
        metric="sqeuclidean",
    )

    if ot_method == "emd":
        w2_squared = float(ot.emd2(a, b, cost_matrix))
    else:
        w2_squared = float(
            ot.sinkhorn2(
                a,
                b,
                cost_matrix,
                reg=reg,
                method=ot_method,
                numItermax=5000,
                stopThr=1e-9,
            )
        )

    return float(math.sqrt(max(w2_squared, 0.0)))
