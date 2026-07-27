#!/usr/bin/env python3
"""PyVista visualization of UAV3D trajectories driven by a **robustified**
controller.

Reads a ``RolloutBundle`` produced by ``collect_uav3d_rollouts.py`` (see
``rollout_io.py``) and renders it. This script never simulates anything —
it's the cheap, iterable half of the old combined script, so camera angle /
legend / color / line-style tweaks don't require re-running the benchmark.
Only numpy + pyvista are needed here (no torch, no policy checkpoint).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE, RolloutBundle, load_rollouts


def _create_box(bounds):
    """Create a PyVista box from (x_min, x_max, y_min, y_max, z_min, z_max)."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))


def _add_legend_text(
    plotter,
    entries: list[list[str]],
    *,
    x_ndc: float = 0.78,
    margin_ndc: float = 0.04,
    font_size: int = 24,
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
        # Default line height = 2.3 × font height keeps the gap proportional
        # to the font size when the caller scales font_size.
        line_dy_ndc = 2.3 * font_size / float(window_height)
    for i, (text, color) in enumerate(entries):
        plotter.add_text(
            text,
            position=(x_ndc, y_top_ndc - i * line_dy_ndc),
            font_size=font_size,
            color=color,
            viewport=True,
        )


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
    outcomes: list[str],
    color_map: dict[str, str],
    rate_stats: dict[str, tuple[float, float]] | None = None,
    std_multiplier: float = 1.0,
) -> list[list[str]]:
    """Build (label, color) legend rows. When ``rate_stats`` provides a
    (mean, std) rate — across seeds, not across this one figure's
    trajectories — for an outcome, add it as its own ``mean% ± k*std%`` row
    right under the count so neither line grows too wide for the upper-right
    corner it's anchored in.
    """
    entries: list[list[str]] = []
    for outcome, color in color_map.items():
        n = sum(1 for o in outcomes if o == outcome)
        if n == 0:
            continue
        entries.append([f"{outcome}: {n}", color])
        if rate_stats is not None and outcome in rate_stats:
            mean, std = rate_stats[outcome]
            entries.append(
                [f"  {100 * mean:.1f}% ± {100 * std_multiplier * std:.1f}%", color]
            )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str,
                        default="experiments/uav_3d/bimodal_rollouts.pkl",
                        help="Rollout bundle written by collect_uav3d_rollouts.py.")
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip rendering the raw-policy figure, even if "
                             "it's present in --data.")
    parser.add_argument("--no-alpha", action="store_true",
                        help="Skip rendering the alpha-smoothing figure, even "
                             "if it's present in --data.")
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
    parser.add_argument("--std-multiplier", type=float, default=2.0,
                        help="Multiplier k applied to the across-seed std when "
                             "annotating goal/collision rate as 'mean ± k*std' "
                             "in the rendered legends (k=2 ≈ a 95%% band under "
                             "normality; only meaningful if --data was collected "
                             "with --num-seeds > 1).")
    parser.add_argument("--line-width", type=float, default=2.8,
                        help="Trajectory line width.")
    parser.add_argument("--line-opacity", type=float, default=0.95,
                        help="Trajectory line opacity.")
    args = parser.parse_args()

    bundle: RolloutBundle = load_rollouts(Path(args.data))
    meta = bundle.meta

    policy_names = [n for n in ("raw", "alpha", "robust") if n in bundle.policies]
    if args.no_raw:
        policy_names = [n for n in policy_names if n != "raw"]
    if args.no_alpha:
        policy_names = [n for n in policy_names if n != "alpha"]
    if not policy_names:
        raise ValueError("Nothing to render: --data has no policies left after "
                          "applying --no-raw/--no-alpha.")

    # Geometry-only: noise/seed don't matter here, this env instance is never
    # stepped, just used for its obstacle/goal layout.
    env = UAV3D(gaussian_noise=True, noise_std=meta.get("noise_std", 0.05), max_steps=None)

    # Color-code each trajectory by outcome only — with separate figures per
    # policy, the policy is already identified by the figure title, so reusing
    # one CB-safe (goal, collision) pair across all panels keeps the viewer
    # from having to remap colors between plots. Pair from Paul Tol's
    # "vibrant" qualitative palette.
    outcome_colors = {"goal": "#0077BB", "collision": "#CC3311"}  # blue / red

    subtitle = (
        f"σ={meta['sigma']}, α={meta['alpha']}, N={meta['smoothing_N']}, "
        f"trajectories={meta['num_trajectories']}"
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
        for name in policy_names:
            data = bundle.policies[name]
            _add_trajectories(plotter, data.traces, data.outcomes,
                              outcome_colors, args.line_width, args.line_opacity)
            legend_entries += _legend_entries_for(
                data.outcomes, outcome_colors,
                rate_stats=data.rate_stats, std_multiplier=args.std_multiplier,
            )
        if legend_entries:
            _add_legend_text(plotter, entries=legend_entries)
        plotter.add_text(f"Bimodal — {subtitle}",
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

    # ---- Non-interactive: render one figure per policy, plus an env-only one.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(
        out_name: str,
        groups: list[tuple[list[np.ndarray], list[str], dict[str, str], dict[str, tuple[float, float]] | None]],
    ) -> None:
        plotter = _make_plotter_with_env(env, off_screen=True)
        legend_entries: list[list[str]] = []
        for traces, outcomes, color_map, rate_stats in groups:
            _add_trajectories(plotter, traces, outcomes,
                              color_map, args.line_width, args.line_opacity)
            legend_entries += _legend_entries_for(
                outcomes, color_map,
                rate_stats=rate_stats, std_multiplier=args.std_multiplier,
            )
        if legend_entries:
            _add_legend_text(plotter, entries=legend_entries)
        plotter.camera_position = preset
        out_path = out_dir / out_name
        plotter.show(screenshot=str(out_path), auto_close=True)
        print(f"Saved {out_path}")

    _render_one(out_name="environment.png", groups=[])
    for name in policy_names:
        data = bundle.policies[name]
        _render_one(
            out_name=f"{name}_trajectories.png",
            groups=[(data.traces, data.outcomes, outcome_colors, data.rate_stats)],
        )


if __name__ == "__main__":
    main()
