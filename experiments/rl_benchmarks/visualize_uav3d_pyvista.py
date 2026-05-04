#!/usr/bin/env python3
"""
Visualize the UAV3D obstacle layout and goal region using PyVista.
"""
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
import torch
from dynamics import UAV3D
from train_rl import VariationalPolicy, normalize_state

# State where the trained K=2 policy is sharply bimodal AND the raw policy
# is "clutch": some trajectories collide, so the visualization shows a
# non-trivial failure mode that randomized smoothing can mitigate. From this
# far-west, low-y, low-z position the two mixture components encode two
# distinct route bundles toward the goal in the far-east, low-z corner.
# 200-trajectory rollout: ~25-unit cluster separation at quarter-horizon
# with a ~66/34 split, and ~72% goal / 24% collision / 4% timeout. Zero
# initial velocity.
BIMODAL_START_STATE = np.array([-13.5, 0.0, -3.0, 0.0, -5.0, 0.0], dtype=np.float32)


def _create_box(bounds):
    """Create a PyVista box from (x_min, x_max, y_min, y_max, z_min, z_max)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))


def _add_legend_text(
    plotter,
    entries: list[list[str]],
    *,
    x_ndc: float = 0.76,
    margin_ndc: float = 0.04,
    font_size: int = 32,
    window_height: int = 1200,
    line_dy_ndc: float | None = None,
) -> None:
    """Render an upper-right legend using ``add_text`` per entry.

    ``vtkLegendBoxActor`` reserves most of each entry's width for a marker
    glyph and squeezes text into a narrow right strip — even with
    ``face=None`` — which clips counts like ``"collision: 47"`` off the
    right edge of the viewport. Drawing each label as a viewport-space
    text actor sidesteps that and gives left-aligned labels with a gap we
    control directly via ``line_dy_ndc``.

    ``margin_ndc`` is the gap (in normalized device coords) between the
    top of the first label and the top of the viewport; pick ``x_ndc`` so
    the longest label leaves a similar gap from the right edge.
    """
    # ``add_text(position=(x, y), viewport=True)`` treats (x, y) as the
    # lower-left of the text bounding box. vtkTextActor's bounding-box
    # height empirically runs ~1.83 × font_size in pixels (font height plus
    # ascender + descender padding), so the text top sits at
    # ``y + 1.83 * font_size / window_height``.
    text_height_ndc = 1.83 * font_size / float(window_height)
    y_top_ndc = 1.0 - margin_ndc - text_height_ndc
    if line_dy_ndc is None:
        # Default line height = 1.9 × font height keeps the gap proportional
        # to the font size when the caller scales font_size.
        line_dy_ndc = 1.9 * font_size / float(window_height)
    for i, (text, color) in enumerate(entries):
        plotter.add_text(
            text,
            position=(x_ndc, y_top_ndc - i * line_dy_ndc),
            font_size=font_size,
            color=color,
            viewport=True,
        )


def _load_uav3d_policy(checkpoint_path: Path) -> tuple[VariationalPolicy, float]:
    """Load a UAV3D VI-BNN policy checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with model_state_dict (legacy checkpoints are not supported).")

    required_fields = ["state_dim", "action_dim", "hidden_dim", "action_scale"]
    missing_fields = [field for field in required_fields if field not in checkpoint]
    if missing_fields:
        raise ValueError(f"Checkpoint missing required fields: {missing_fields}")

    model_state_dict = checkpoint["model_state_dict"]
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    action_scale = float(checkpoint["action_scale"])
    num_components = int(checkpoint.get("num_components", 1))

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_scale=action_scale,
        num_components=num_components,
    )
    policy.load_state_dict(model_state_dict)
    policy.eval()
    return policy, action_scale


_SAFETY_STEP_CAP = 1000


def _sample_uav3d_trajectories(
    policy: VariationalPolicy | None,
    action_scale: float,
    num_trajectories: int,
    seed: int,
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    """Roll out trajectories from the bimodal start state.

    Rollouts run until natural termination (goal, obstacle collision, or
    corridor exit). The env is constructed with ``max_steps=None`` and the
    inner loop only enforces a generous safety cap.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    trajectories: list[np.ndarray] = []
    outcomes: list[str] = []
    outcome_counts = {"goal": 0, "collision": 0, "timeout": 0}

    env = UAV3D(gaussian_noise=True, noise_std=0.05, max_steps=None)
    start_pos = np.array(
        [BIMODAL_START_STATE[0], BIMODAL_START_STATE[2], BIMODAL_START_STATE[4]],
        dtype=np.float32,
    )
    for _ in range(num_trajectories):
        env._state = BIMODAL_START_STATE.copy()
        env._step_count = 0
        env._done = False
        env._prev_dist = float(np.linalg.norm(start_pos - env.goal_center))
        state = env._state.copy()
        trace = [start_pos.copy()]

        steps = 0
        while not env.is_done() and steps < _SAFETY_STEP_CAP:
            if policy is not None:
                state_t = torch.tensor(
                    normalize_state(state, "uav_3d"), dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    action_t, _raw, _lp = policy.sample(state_t)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                action = np.random.uniform(-action_scale, action_scale, size=(3,)).astype(np.float32)

            _, _reward, state = env.step(action)
            trace.append(np.array([state[0], state[2], state[4]], dtype=np.float32))
            steps += 1

        # Trust the env's classification: with swept-segment collision the
        # post-step position can lie past the obstacle, so deriving the
        # outcome from final_pos alone misclassifies those as "timeout".
        outcome = env._terminal_reason if env._terminal_reason is not None else "timeout"

        trajectories.append(np.stack(trace, axis=0))
        outcomes.append(outcome)
        outcome_counts[outcome] += 1

    return trajectories, outcomes, outcome_counts


def visualize_uav3d_with_pyvista():
    """Visualize the UAV3D environment using PyVista with proper depth rendering."""
    print("Creating 3D UAV visualization using PyVista...")

    env = UAV3D()

    checkpoint_path = Path("experiments/uav_3d/policy.pt")
    policy: VariationalPolicy | None = None
    action_scale = 4.0
    if checkpoint_path.exists():
        try:
            policy, action_scale = _load_uav3d_policy(checkpoint_path)
            print(f"Loaded VI-BNN policy from {checkpoint_path}")
        except Exception as exc:
            print(f"Warning: failed to load policy checkpoint ({exc}). Falling back to random actions.")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. Falling back to random actions.")

    # All trajectories start from the bimodal state so that the two mixture
    # modes (north- vs south-route around the central tower) naturally show
    # up as separated trajectory bundles under normal mixture sampling.
    trajectories, outcomes, outcome_counts = _sample_uav3d_trajectories(
        policy=policy,
        action_scale=action_scale,
        num_trajectories=100,
        seed=42,
    )

    # Domain from JAIR/DynAbs specs.
    x_bounds = (-15.0, 15.0)
    y_bounds = (-9.0, 9.0)
    z_bounds = (-7.0, 7.0)

    # PyVista typing stubs can be inconsistent across versions; keep runtime behavior.
    plotter = cast(Any, pv.Plotter(window_size=[1200, 800]))
    plotter.set_background("white")

    # Draw obstacles (critical regions) in red
    print(f"Drawing {len(env.obstacles)} obstacles...")
    for bounds in env.obstacles:
        obstacle = _create_box(bounds)
        plotter.add_mesh(
            obstacle,
            color="red",
            opacity=0.7,
            smooth_shading=False,
            lighting=True,
        )

    print("Obstacles drawn.")

    # Draw goal region in green
    print("Drawing goal region...")
    goal_bounds = (
        float(env.goal_min[0]),
        float(env.goal_max[0]),
        float(env.goal_min[1]),
        float(env.goal_max[1]),
        float(env.goal_min[2]),
        float(env.goal_max[2]),
    )
    goal_box = _create_box(goal_bounds)
    plotter.add_mesh(goal_box, color="limegreen", opacity=0.8, smooth_shading=False)

    print("Goal region drawn.")

    # Draw start position as a sphere
    start_center = (
        float(BIMODAL_START_STATE[0]),
        float(BIMODAL_START_STATE[2]),
        float(BIMODAL_START_STATE[4]),
    )
    start_marker = pv.Sphere(radius=0.5, center=start_center, theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(start_marker, color="royalblue", opacity=1.0)

    print("Start position marked.")

    # Draw sampled trajectories, coloured by episode outcome.
    print(f"Drawing {len(trajectories)} sampled trajectories...")
    outcome_colors = {"goal": "seagreen", "collision": "orangered", "timeout": "goldenrod"}
    for points, outcome in zip(trajectories, outcomes):
        color = outcome_colors[outcome]
        line = pv.lines_from_points(points, close=False)
        plotter.add_mesh(line, color=color, line_width=1.5, opacity=0.8)

        end_marker = pv.Sphere(
            radius=0.18, center=tuple(points[-1]),
            theta_resolution=16, phi_resolution=16,
        )
        plotter.add_mesh(end_marker, color=color, opacity=0.95)

    _add_legend_text(
        plotter,
        entries=[
            ["Goal-reaching",      outcome_colors["goal"]],
            ["Obstacle collision", outcome_colors["collision"]],
            ["Timeout",            outcome_colors["timeout"]],
        ],
        x_ndc=0.66,  # left-shifted to fit the longer "Obstacle collision"
    )

    print("Trajectories drawn.")

    # Add a transparent domain frame for context.
    domain_box = pv.Box(
        bounds=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1])
    )
    plotter.add_mesh(domain_box, style="wireframe", color="black", line_width=1, opacity=0.35)

    # Axes, labels and grid.
    plotter.show_grid(
        xtitle="X Position (m)",
        ytitle="Y Position (m)",
        ztitle="Z Position (m)",
        color="black",
    )
    plotter.show_bounds(
        bounds=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1], z_bounds[0], z_bounds[1]),
        grid="front",
        location="outer",
        all_edges=True,
        color="black",
        xtitle="X Position (m)",
        ytitle="Y Position (m)",
        ztitle="Z Position (m)",
    )
    plotter.add_axes(line_width=2, labels_off=False)

    # Match previous preferred orientation roughly (elev=55, azim=-60).
    plotter.camera_position = [(30.0, -52.0, 43.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]

    print("Plot configured.")
    print("\nEnvironment details:")
    print(f"  - Start: ({start_center[0]}, {start_center[1]}, {start_center[2]})")
    print(
        "  - Goal: "
        f"x in [{env.goal_min[0]}, {env.goal_max[0]}], "
        f"y in [{env.goal_min[1]}, {env.goal_max[1]}], "
        f"z in [{env.goal_min[2]}, {env.goal_max[2]}]"
    )
    print(f"  - Obstacles: {len(env.obstacles)}")
    print(f"  - Sampled trajectories: {len(trajectories)}")
    print(f"    * Goal-reaching: {outcome_counts['goal']}")
    print(f"    * Collisions: {outcome_counts['collision']}")
    print(f"    * Timeouts: {outcome_counts['timeout']}")

    print("\nShowing interactive view...")
    print("You can rotate the view with mouse and zoom with scroll wheel.")
    print("Close the window when done.")
    plotter.show()


if __name__ == '__main__':
    visualize_uav3d_with_pyvista()
