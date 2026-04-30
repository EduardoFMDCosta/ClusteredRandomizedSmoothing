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
    _add_legend_text,
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
    """Roll out until natural termination (goal / collision / corridor exit).

    The env should be constructed with ``max_steps=None`` so it never times
    out from inside ``step()``; ``horizon`` here is just a safety cap that
    should never bind in practice.
    """
    env._state = start_state.copy()
    env._step_count = 0
    env._done = False
    start_pos = np.array([start_state[0], start_state[2], start_state[4]], dtype=np.float32)
    env._prev_dist = float(np.linalg.norm(start_pos - env.goal_center))

    state = env._state.copy()
    trace = [start_pos.copy()]
    steps = 0
    while not env.is_done() and steps < horizon:
        action = action_fn(state)
        _, _reward, state = env.step(action)
        trace.append(np.array([state[0], state[2], state[4]], dtype=np.float32))
        steps += 1

    # Trust the env's terminal_reason: with swept-segment collision the
    # post-step position may lie past the obstacle, so deriving the outcome
    # from final_pos alone misclassifies those as "timeout".
    outcome = env._terminal_reason if env._terminal_reason is not None else "timeout"
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
):

    def action(state: np.ndarray) -> np.ndarray:
        a = _robustified_action(
            stochastic_fn=stochastic_fn,
            state=state,
            parameters=parameters,
            sigma=sigma,
        )
        return a

    return action


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
        trace, outcome = _rollout(env, start_state, horizon, action_fn)
        traces.append(trace)
        outcomes.append(outcome)
        counts[outcome] += 1
        print(f"  [{label}] {i + 1}/{num_trajectories} → {outcome} "
              f"({time.time() - t0:.1f}s elapsed)")
    return traces, outcomes, counts


def _add_env_geometry(plotter: Any, env: UAV3D) -> None:
    # Obstacles: low fill opacity so trajectories remain readable through them,
    # plus a strong dark-red edge outline so the geometry stays crisp.
    for bounds in env.obstacles:
        obstacle = _create_box(bounds)
        plotter.add_mesh(
            obstacle,
            color="indianred", opacity=1.0,
            show_edges=True, edge_color="firebrick", line_width=1.2,
            smooth_shading=False, lighting=True,
        )
    goal_bounds = (
        float(env.goal_min[0]), float(env.goal_max[0]),
        float(env.goal_min[1]), float(env.goal_max[1]),
        float(env.goal_min[2]), float(env.goal_max[2]),
    )
    goal_box = _create_box(goal_bounds)
    plotter.add_mesh(
        goal_box,
        color="limegreen", opacity=0.45,
        show_edges=True, edge_color="darkgreen", line_width=1.2,
        smooth_shading=False,
    )

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
    color_by_outcome: dict[str, str],
    line_width: float,
    line_opacity: float,
) -> None:
    """Render each trajectory in the (policy, outcome) combo color.

    Outcomes that are missing from ``color_by_outcome`` are skipped — that is
    how we suppress timeouts (which never happen at the operating point we
    care about and just clutter the legend).
    """
    for points, outcome in zip(traces, outcomes):
        color = color_by_outcome.get(outcome)
        if color is None:
            continue
        line = pv.lines_from_points(points, close=False)
        plotter.add_mesh(line, color=color, line_width=line_width, opacity=line_opacity)


def _make_plotter_with_env(env: UAV3D, off_screen: bool) -> Any:
    plotter = cast(Any, pv.Plotter(window_size=[1600, 1200], off_screen=off_screen))
    plotter.set_background("white")
    _add_env_geometry(plotter, env)
    return plotter


def _legend_entries_for(
    outcomes: list[str], color_map: dict[str, str],
) -> list[list[str]]:
    entries: list[list[str]] = []
    for outcome, color in color_map.items():
        n = sum(1 for o in outcomes if o == outcome)
        if n > 0:
            entries.append([f"{outcome}: {n}", color])
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--smoothing-N", type=int, default=30,
                        help="Noisy-state samples used inside smooth() per env step.")
    parser.add_argument("--confidence-N", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.45)
    parser.add_argument("--dbscan-min-samples", type=int, default=50)
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip rolling out the raw policy for comparison.")
    parser.add_argument("--no-alpha", action="store_true",
                        help="Skip rolling out the alpha-smoothing baseline "
                             "(``max_clusters=1``).")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="UAV3D process-noise std used when rolling out.")
    parser.add_argument("--out-dir", type=str,
                        default="experiments/uav_3d/bimodal_figures",
                        help="Directory to write rendered figures into. Each "
                             "policy gets its own PNG; an additional env-only "
                             "PNG is written alongside.")
    parser.add_argument("--interactive", action="store_true",
                        help="Open an interactive window instead of writing to --out. "
                             "Use this to find a better camera, then re-run without "
                             "--interactive to save.")
    parser.add_argument("--camera-preset", type=str, default="manual-iso",
                        choices=["manual-iso", "down-corridor", "topdown-tilted",
                                 "topdown", "iso"],
                        help="'manual-iso' (default) is a hand-tuned 3/4 view "
                             "from the (-X, -Y, +Z) octant that frames the "
                             "obstacle wall face-on; 'down-corridor' sits behind "
                             "the start and looks toward the goal at low "
                             "elevation; 'topdown-tilted' is a slightly tilted "
                             "overhead; 'topdown' is straight overhead; 'iso' is "
                             "the prior 3/4 isometric.")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0,
                        help="After applying --camera-preset, orbit the camera "
                             "around the focal point by this angle in degrees "
                             "(rotation axis = world +Z). Right-hand rule: "
                             "positive values move the camera CCW viewed from "
                             "above, which makes the scene appear to rotate to "
                             "the right. Flip the sign to rotate the other way.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    print(f"Loaded VI-BNN policy from {checkpoint_path} (action_scale={action_scale}).")

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)

    def _make_parameters(max_clusters: int) -> Parameters:
        return Parameters(
            smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
            clustering=ClusteringParameters(
                max_num_clusters=max_clusters,
                coverage=args.coverage,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
            ),
            confidence=ConfidenceParameters(num_samples=args.confidence_N, beta=1e-2),
        )

    parameters_robust = _make_parameters(args.max_clusters)
    parameters_alpha = _make_parameters(1)

    # max_steps=None: rollouts run to natural termination (goal / collision
    # / corridor exit). args.horizon is a safety cap inside _rollout.
    env = UAV3D(gaussian_noise=True, noise_std=args.noise_std, max_steps=None)

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

    # ---- alpha-smoothing baseline (max_clusters=1) ----
    alpha_traces: list[np.ndarray] = []
    alpha_outcomes: list[str] = []
    alpha_counts: dict[str, int] = {"goal": 0, "collision": 0, "timeout": 0}
    if not args.no_alpha:
        print(f"Rolling out alpha-smoothing (max_clusters=1) trajectories "
              f"— this calls the policy ~{2 * args.smoothing_N} times per env step...")
        alpha_traces, alpha_outcomes, alpha_counts = _collect_trajectories(
            env=env,
            start_state=BIMODAL_START_STATE,
            horizon=args.horizon,
            num_trajectories=args.num_trajectories,
            action_fn=_robustified_action_fn(
                stochastic_fn=stochastic_fn,
                parameters=parameters_alpha,
                sigma=args.sigma,
            ),
            label="alpha",
        )

    # ---- robustified (clusterized) rollouts ----
    print(f"Rolling out clusterized robustified trajectories "
          f"— this calls the policy ~{2 * args.smoothing_N} times per env step...")
    rob_traces, rob_outcomes, rob_counts = _collect_trajectories(
        env=env,
        start_state=BIMODAL_START_STATE,
        horizon=args.horizon,
        num_trajectories=args.num_trajectories,
        action_fn=_robustified_action_fn(
            stochastic_fn=stochastic_fn,
            parameters=parameters_robust,
            sigma=args.sigma,
        ),
        label="robust",
    )

    print("")
    if not args.no_raw:
        print(f"Raw   : goal={raw_counts['goal']} / "
              f"collision={raw_counts['collision']} / timeout={raw_counts['timeout']}")
    if not args.no_alpha:
        print(f"Alpha : goal={alpha_counts['goal']} / "
              f"collision={alpha_counts['collision']} / timeout={alpha_counts['timeout']}")
    print(f"Robust: goal={rob_counts['goal']} / "
          f"collision={rob_counts['collision']} / timeout={rob_counts['timeout']}")

    # Color-code each trajectory by outcome only — with separate figures per
    # policy, the policy is already identified by the figure title, so reusing
    # one CB-safe (goal, collision) pair across all panels keeps the viewer
    # from having to remap colors between plots. Pair from Paul Tol's
    # "vibrant" qualitative palette.
    outcome_colors = {"goal": "#0077BB", "collision": "#CC3311"}  # blue / red
    raw_colors   = outcome_colors
    alpha_colors = outcome_colors
    rob_colors   = outcome_colors
    line_width = 2.8
    line_opacity = 0.95

    base_subtitle = (
        f"σ={args.sigma}, α={args.alpha}, N={args.smoothing_N}, "
        f"trajectories={args.num_trajectories}"
    )

    # Camera presets framed around the BIMODAL_START_STATE → goal axis.
    # The start sphere is plotted at (start_x, start_y, start_z); both topdown
    # presets aim the focal at the midpoint between start and goal so the
    # bimodal lobes are centered in the frame.
    start_pos = np.array([
        float(BIMODAL_START_STATE[0]),
        float(BIMODAL_START_STATE[2]),
        float(BIMODAL_START_STATE[4]),
    ], dtype=np.float64)
    goal_pos = env.goal_center.astype(np.float64)
    focal = 0.5 * (start_pos + goal_pos)

    # Direction from start → goal in world coordinates; we set up the
    # "down-corridor" view by stepping *backwards* along that axis from the
    # start, so the camera looks down the obstacle corridor and the two gaps
    # the bimodal lobes thread through are visible face-on. Distances are
    # tuned to give a slight pad around the trajectories — enough to see
    # the obstacle walls without the scene crowding the frame.
    flight = goal_pos - start_pos
    flight_unit = flight / max(float(np.linalg.norm(flight)), 1e-9)
    behind_start = start_pos - 26.0 * flight_unit + np.array([0.0, -9.0, 12.0])

    camera_presets = {
        # Hand-tuned 3/4 view, from interactive session — frames the obstacle
        # wall face-on so the trajectory ring threads visibly between pillars.
        "manual-iso": [
            (-39.2354, -30.9751, 49.5394),  # camera
            (3.1297, 0.2202, -4.9787),  # focal
            (0.5626, 0.4491, 0.6941),  # view-up
        ],
        # 3/4 isometric.
        "iso": [
            (focal[0] + 40.0, focal[1] - 68.0, focal[2] + 56.0),
            tuple(focal),
            (0.0, 0.0, 1.0),
        ],
        # Tilted top-down: slight forward lean keeps a 3-D feel while
        # making both bimodal lobes plainly visible around the obstacles.
        "topdown-tilted": [
            (focal[0], focal[1] - 24.0, focal[2] + 72.0),
            tuple(focal),
            (0.0, 1.0, 0.0),
        ],
        # Pure overhead: best for showing horizontal threading of obstacles.
        "topdown": [
            (focal[0], focal[1], focal[2] + 92.0),
            tuple(focal),
            (0.0, 1.0, 0.0),
        ],
        # Behind-the-start, low elevation, slight Y/Z offset so the obstacle
        # gaps in the wall the trajectories thread through read clearly.
        "down-corridor": [
            tuple(behind_start),
            tuple(focal),
            (0.0, 0.0, 1.0),
        ],
    }
    preset = camera_presets[args.camera_preset]
    if args.camera_yaw_deg != 0.0 and args.camera_preset != "topdown":
        yaw = float(np.deg2rad(args.camera_yaw_deg))
        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        cam = np.array(preset[0], dtype=np.float64)
        rel = cam - focal
        rotated = np.array([
            cos_y * rel[0] - sin_y * rel[1],
            sin_y * rel[0] + cos_y * rel[1],
            rel[2],
        ])
        new_cam = focal + rotated
        preset = [tuple(new_cam.tolist()), tuple(focal.tolist()), preset[2]]

    # ---- Interactive: combined view of every enabled trajectory group, for
    # camera tuning. No files are written.
    if args.interactive:
        plotter = _make_plotter_with_env(env, off_screen=False)
        legend_entries: list[list[str]] = []
        if not args.no_raw:
            _add_trajectories(plotter, raw_traces, raw_outcomes,
                              raw_colors, line_width, line_opacity)
            legend_entries += _legend_entries_for(raw_outcomes, raw_colors)
        if not args.no_alpha:
            _add_trajectories(plotter, alpha_traces, alpha_outcomes,
                              alpha_colors, line_width, line_opacity)
            legend_entries += _legend_entries_for(alpha_outcomes, alpha_colors)
        _add_trajectories(plotter, rob_traces, rob_outcomes,
                          rob_colors, line_width, line_opacity)
        legend_entries += _legend_entries_for(rob_outcomes, rob_colors)
        if legend_entries:
            _add_legend_text(plotter, entries=legend_entries)
        plotter.add_text(f"Bimodal — {base_subtitle}",
                         position="upper_edge", font_size=10, color="black")
        plotter.camera_position = preset

        def _print_camera() -> None:
            cam = plotter.camera_position
            pos, foc, up = tuple(cam[0]), tuple(cam[1]), tuple(cam[2])
            print(
                "[camera] copy-paste preset:\n"
                "    [\n"
                f"        ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}),  # camera\n"
                f"        ({foc[0]:.4f}, {foc[1]:.4f}, {foc[2]:.4f}),  # focal\n"
                f"        ({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f}),  # view-up\n"
                "    ]"
            )

        plotter.add_key_event("c", _print_camera)
        print(
            "[interactive] Rotate/pan/zoom with the mouse. Press 'c' in the "
            "PyVista window to print the current camera position; close the "
            "window when happy and the final position will be printed too."
        )
        plotter.show()
        print("\n[camera] final state on window close:")
        _print_camera()
        return

    # ---- Non-interactive: render up to four separate figures.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(
        out_name: str,
        groups: list[tuple[list[np.ndarray], list[str], dict[str, str]]],
    ) -> None:
        plotter = _make_plotter_with_env(env, off_screen=True)
        legend_entries: list[list[str]] = []
        for traces, outcomes, color_map in groups:
            _add_trajectories(plotter, traces, outcomes,
                              color_map, line_width, line_opacity)
            legend_entries += _legend_entries_for(outcomes, color_map)
        if legend_entries:
            _add_legend_text(plotter, entries=legend_entries)
        plotter.camera_position = preset
        out_path = out_dir / out_name
        plotter.show(screenshot=str(out_path), auto_close=True)
        print(f"Saved {out_path}")

    _render_one(
        out_name="environment.png",
        groups=[],
    )
    if not args.no_raw:
        _render_one(
            out_name="raw_trajectories.png",
            groups=[(raw_traces, raw_outcomes, raw_colors)],
        )
    if not args.no_alpha:
        _render_one(
            out_name="alpha_trajectories.png",
            groups=[(alpha_traces, alpha_outcomes, alpha_colors)],
        )
    _render_one(
        out_name="robust_trajectories.png",
        groups=[(rob_traces, rob_outcomes, rob_colors)],
    )


if __name__ == "__main__":
    main()
