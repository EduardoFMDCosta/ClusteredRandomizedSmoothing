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
from torch.distributions import Normal
from train_vi_bnn_rl import VariationalPolicy, normalize_state, scale_action_for_env


def _create_box(bounds):
    """Create a PyVista box from (x_min, x_max, y_min, y_max, z_min, z_max)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))


def _load_uav3d_policy(checkpoint_path: Path) -> tuple[VariationalPolicy, float]:
    """Load a UAV3D VI-BNN policy checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with model_state_dict (legacy checkpoints are not supported).")

    required_fields = ["state_dim", "action_dim", "hidden_dim", "prior_std", "state_dependent_std", "env_type", "action_scale"]
    missing_fields = [field for field in required_fields if field not in checkpoint]
    if missing_fields:
        raise ValueError(f"Checkpoint missing required fields: {missing_fields}")

    env_type = str(checkpoint["env_type"])
    if env_type != "uav_3d":
        raise ValueError(f"Checkpoint env_type is {env_type}, expected uav_3d")

    model_state_dict = checkpoint["model_state_dict"]
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    prior_std = float(checkpoint["prior_std"])
    state_dependent_std = bool(checkpoint["state_dependent_std"])
    action_scale = float(checkpoint["action_scale"])

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        prior_std=prior_std,
        state_dependent_std=state_dependent_std,
    )
    policy.load_state_dict(model_state_dict)
    policy.eval()
    return policy, action_scale


def _sample_uav3d_trajectories(
    policy: VariationalPolicy | None,
    action_scale: float,
    num_trajectories: int,
    horizon: int,
    seed: int,
) -> tuple[list[np.ndarray], list[str], dict[str, int]]:
    """Roll out sampled trajectories in UAV3D and return position traces."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    trajectories: list[np.ndarray] = []
    outcomes: list[str] = []
    outcome_counts = {"goal": 0, "collision": 0, "timeout": 0}

    for _ in range(num_trajectories):
        env = UAV3D(gaussian_noise=True, noise_std=0.15)
        state = env.reset()
        trace = [np.array([state[0], state[2], state[4]], dtype=np.float32)]

        outcome = "timeout"
        for _step in range(horizon):
            if policy is not None:
                state_t = torch.tensor(normalize_state(state, "uav_3d"), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, std = policy(state_t)
                    raw_action = Normal(mean, std).sample()
                    normalized_action = torch.tanh(raw_action).squeeze(0).cpu().numpy().astype(np.float32)
                    action = scale_action_for_env(normalized_action, "uav_3d", action_scale=action_scale)
            else:
                action = np.random.uniform(-action_scale, action_scale, size=(3,)).astype(np.float32)

            _, reward, state = env.step(action)
            trace.append(np.array([state[0], state[2], state[4]], dtype=np.float32))

            if env.is_done():
                if reward >= 0.9:
                    outcome = "goal"
                elif reward <= -0.9:
                    outcome = "collision"
                else:
                    outcome = "timeout"
                break

        trajectories.append(np.stack(trace, axis=0))
        outcomes.append(outcome)
        outcome_counts[outcome] += 1

    return trajectories, outcomes, outcome_counts


def visualize_uav3d_with_pyvista():
    """Visualize the UAV3D environment using PyVista with proper depth rendering."""
    print("Creating 3D UAV visualization using PyVista...")

    env = UAV3D()

    checkpoint_path = Path("experiments/uav_3d/vi_bnn_policy.pt")
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

    trajectories, outcomes, outcome_counts = _sample_uav3d_trajectories(
        policy=policy,
        action_scale=action_scale,
        num_trajectories=12,
        horizon=32,
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
    start_pos = env.start_state
    start_center = (float(start_pos[0]), float(start_pos[2]), float(start_pos[4]))
    start_marker = pv.Sphere(radius=0.5, center=start_center, theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(start_marker, color="royalblue", opacity=1.0)

    print("Start position marked.")

    # Draw sampled trajectories.
    print(f"Drawing {len(trajectories)} sampled trajectories...")
    for points, outcome in zip(trajectories, outcomes):
        if outcome == "goal":
            color = "seagreen"
        elif outcome == "collision":
            color = "orangered"
        else:
            color = "goldenrod"

        line = pv.lines_from_points(points, close=False)
        plotter.add_mesh(line, color=color, line_width=3)

        end_marker = pv.Sphere(radius=0.18, center=tuple(points[-1]), theta_resolution=16, phi_resolution=16)
        plotter.add_mesh(end_marker, color=color, opacity=0.95)

    plotter.add_legend(
        labels=[
            ["Goal-reaching", "seagreen"],
            ["Obstacle collision", "orangered"],
            ["Timeout", "goldenrod"],
        ],
        bcolor=(1.0, 1.0, 1.0),
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
