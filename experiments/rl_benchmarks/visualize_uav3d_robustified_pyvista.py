#!/usr/bin/env python3
"""PyVista visualization of UAV3D trajectories driven by a **robustified**
controller.

At every environment step we:
  1. Wrap the VI-BNN policy in ``UAV3DPolicyStochasticFunction``.
  2. Build ``GaussianAlphaTrimmingSmoothing`` on the current (normalized) state
     with Gaussian state-space noise of std ``sigma``.
  3. Either sample from the resulting ``MixtureDirac`` or pick the mode
     (highest partition weight) as the action.

Optionally also rolls out the raw (non-robustified) policy for comparison and
plots both bundles of trajectories in the same scene.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
import torch
from dynamics import UAV3D
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import VariationalPolicy, normalize_state
from visualize_uav3d_pyvista import (
    BIMODAL_START_STATE,
    _create_box,
    _load_uav3d_policy,
)

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing


def _robustified_action(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    state: np.ndarray,
    parameters: Parameters,
    sigma: float,
) -> np.ndarray:
    """Build a smoother at ``state`` and return one action from the
    robustified ``MixtureDirac``.
    """
    x = torch.tensor(normalize_state(state, "uav_3d"), dtype=torch.float32).unsqueeze(0)
    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=parameters,
    )
    distribution = smoother.smooth()
    action_t = distribution.sample(size=())
    return action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    


def _rollout(
    env: UAV3D,
    start_state: np.ndarray,
    horizon: int,
    action_fn,
) -> tuple[np.ndarray, str]:
    env._state = start_state.copy()
    env._step_count = 0
    env._done = False
    start_pos = np.array([start_state[0], start_state[2], start_state[4]], dtype=np.float32)
    env._prev_dist = float(np.linalg.norm(start_pos - env.goal_center))

    state = env._state.copy()
    trace = [start_pos.copy()]
    outcome = "timeout"
    for _ in range(horizon):
        action = action_fn(state)
        _, _reward, state = env.step(action)
        trace.append(np.array([state[0], state[2], state[4]], dtype=np.float32))
        if env.is_done():
            final_pos = np.array([state[0], state[2], state[4]], dtype=np.float32)
            if env._check_goal(final_pos):
                outcome = "goal"
            elif env._check_obstacle_collision(final_pos) or env._outside_corridor(final_pos):
                outcome = "collision"
            break

    return np.stack(trace, axis=0), outcome


def _raw_policy_action_fn(policy: VariationalPolicy):
    def f(state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(normalize_state(state, "uav_3d"), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = policy.sample(state_t)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)
    return f


def _robustified_action_fn(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    parameters: Parameters,
    sigma: float,
    selector: str,
):
    """Returns a closure that remembers the previous chosen action so that
    ``selector='consistent'`` can commit to a mode across steps. The state
    is reset at the start of each trajectory via ``.reset()``."""
    class _Fn:
        def __init__(self):
            self.prev_action: np.ndarray | None = None

        def reset(self):
            self.prev_action = None

        def __call__(self, state: np.ndarray) -> np.ndarray:
            a = _robustified_action(
                stochastic_fn=stochastic_fn,
                state=state,
                parameters=parameters,
                sigma=sigma,
                selector=selector,
                prev_action=self.prev_action,
            )
            self.prev_action = a
            return a

    return _Fn()


def _collect_trajectories(
    env: UAV3D,
    start_state: np.ndarray,
    horizon: int,
    num_trajectories: int,
    action_fn,
    label: str,
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    counts = {"goal": 0, "collision": 0, "timeout": 0}
    traces: list[np.ndarray] = []
    outcomes: list[str] = []
    t0 = time.time()
    for i in range(num_trajectories):
        if hasattr(action_fn, "reset"):
            action_fn.reset()
        trace, outcome = _rollout(env, start_state, horizon, action_fn)
        traces.append(trace)
        outcomes.append(outcome)
        counts[outcome] += 1
        print(f"  [{label}] {i + 1}/{num_trajectories} → {outcome} "
              f"({time.time() - t0:.1f}s elapsed)")
    return traces, outcomes, counts


def _add_env_geometry(plotter: Any, env: UAV3D) -> None:
    for bounds in env.obstacles:
        obstacle = _create_box(bounds)
        plotter.add_mesh(obstacle, color="red", opacity=0.55,
                         smooth_shading=False, lighting=True)
    goal_bounds = (
        float(env.goal_min[0]), float(env.goal_max[0]),
        float(env.goal_min[1]), float(env.goal_max[1]),
        float(env.goal_min[2]), float(env.goal_max[2]),
    )
    goal_box = _create_box(goal_bounds)
    plotter.add_mesh(goal_box, color="limegreen", opacity=0.75, smooth_shading=False)

    start_center = (
        float(BIMODAL_START_STATE[0]),
        float(BIMODAL_START_STATE[2]),
        float(BIMODAL_START_STATE[4]),
    )
    start_marker = pv.Sphere(radius=0.5, center=start_center,
                             theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(start_marker, color="royalblue", opacity=1.0)

    domain_box = pv.Box(bounds=(-15.0, 15.0, -9.0, 9.0, -7.0, 7.0))
    plotter.add_mesh(domain_box, style="wireframe", color="black",
                     line_width=1, opacity=0.35)
    plotter.show_bounds(
        bounds=(-15.0, 15.0, -9.0, 9.0, -7.0, 7.0),
        grid="front", location="outer", all_edges=True, color="black",
        xtitle="X Position (m)", ytitle="Y Position (m)", ztitle="Z Position (m)",
    )
    plotter.add_axes(line_width=2, labels_off=False)


def _add_trajectories(
    plotter: Any,
    traces: list[np.ndarray],
    outcomes: list[str],
    line_color: str,
    end_colors_by_outcome: dict[str, str],
    line_width: float,
    line_opacity: float,
) -> None:
    for points, outcome in zip(traces, outcomes):
        line = pv.lines_from_points(points, close=False)
        plotter.add_mesh(line, color=line_color, line_width=line_width, opacity=line_opacity)
        end_marker = pv.Sphere(
            radius=0.18, center=tuple(points[-1]),
            theta_resolution=16, phi_resolution=16,
        )
        plotter.add_mesh(end_marker, color=end_colors_by_outcome[outcome], opacity=0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--smoothing-N", type=int, default=800,
                        help="Noisy-state samples used inside smooth() per env step.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.6)
    parser.add_argument("--dbscan-min-samples", type=int, default=20)
    parser.add_argument("--selector", type=str, default="consistent",
                        choices=["sample", "mode", "consistent"],
                        help="How to pick an action from the robustified MixtureDirac: "
                             "'mode' picks the heaviest-weight cluster every step, "
                             "'sample' draws from the partition each step, "
                             "'consistent' starts at the heaviest-weight cluster and then "
                             "picks the centroid closest to the previous action (mode commitment).")
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip rolling out the raw policy for comparison.")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="UAV3D process-noise std used when rolling out.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    print(f"Loaded VI-BNN policy from {checkpoint_path} (action_scale={action_scale}).")

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
    parameters = Parameters(
        smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=args.max_clusters,
            coverage=args.coverage,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        ),
        confidence=ConfidenceParameters(num_samples=args.smoothing_N, beta=1e-2),
    )

    env = UAV3D(gaussian_noise=True, noise_std=args.noise_std)

    # ---- raw policy rollouts (optional) ----
    raw_traces: list[np.ndarray] = []
    raw_outcomes: list[str] = []
    raw_counts: dict[str, int] = {"goal": 0, "collision": 0, "timeout": 0}
    if not args.no_raw:
        print("Rolling out raw policy trajectories...")
        raw_traces, raw_outcomes, raw_counts = _collect_trajectories(
            env=env,
            start_state=BIMODAL_START_STATE,
            horizon=args.horizon,
            num_trajectories=args.num_trajectories,
            action_fn=_raw_policy_action_fn(policy),
            label="raw",
        )

    # ---- robustified rollouts ----
    print(f"Rolling out robustified ({args.selector}) trajectories "
          f"— this calls the policy ~{2 * args.smoothing_N} times per env step...")
    rob_traces, rob_outcomes, rob_counts = _collect_trajectories(
        env=env,
        start_state=BIMODAL_START_STATE,
        horizon=args.horizon,
        num_trajectories=args.num_trajectories,
        action_fn=_robustified_action_fn(
            stochastic_fn=stochastic_fn,
            parameters=parameters,
            sigma=args.sigma,
            selector=args.selector,
        ),
        label="robust",
    )

    print("")
    if not args.no_raw:
        print(f"Raw        : goal={raw_counts['goal']} / "
              f"collision={raw_counts['collision']} / timeout={raw_counts['timeout']}")
    print(f"Robustified: goal={rob_counts['goal']} / "
          f"collision={rob_counts['collision']} / timeout={rob_counts['timeout']}")

    # ---- PyVista scene (interactive window; rotate with mouse) ----
    plotter = cast(Any, pv.Plotter(window_size=[1600, 1000], off_screen=False))
    plotter.set_background("white")
    _add_env_geometry(plotter, env)

    outcome_end_colors = {"goal": "seagreen", "collision": "orangered", "timeout": "goldenrod"}
    if not args.no_raw:
        _add_trajectories(
            plotter, raw_traces, raw_outcomes,
            line_color="steelblue",
            end_colors_by_outcome=outcome_end_colors,
            line_width=1.2, line_opacity=0.45,
        )
    _add_trajectories(
        plotter, rob_traces, rob_outcomes,
        line_color="purple",
        end_colors_by_outcome=outcome_end_colors,
        line_width=2.5, line_opacity=0.95,
    )

    legend_entries = []
    if not args.no_raw:
        legend_entries.append(["Raw policy", "steelblue"])
    legend_entries.extend([
        [f"Robustified ({args.selector})", "purple"],
        ["End: goal",      outcome_end_colors["goal"]],
        ["End: collision", outcome_end_colors["collision"]],
        ["End: timeout",   outcome_end_colors["timeout"]],
    ])
    plotter.add_legend(labels=legend_entries, bcolor=(1.0, 1.0, 1.0))

    title_line = (
        f"Robustified UAV3D trajectories — σ={args.sigma}, α={args.alpha}, "
        f"N={args.smoothing_N}, selector={args.selector}"
    )
    plotter.add_text(title_line, position="upper_edge", font_size=10, color="black")

    plotter.camera_position = [(30.0, -52.0, 43.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.show()

    # out_path = Path(args.out)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # plotter.show(screenshot=str(out_path), auto_close=not args.interactive)
    # print(f"Saved trajectory visualization to {out_path}")


if __name__ == "__main__":
    main()
