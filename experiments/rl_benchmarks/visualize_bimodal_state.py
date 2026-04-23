#!/usr/bin/env python3
"""Visualize the GMM policy's bimodal action distribution at a specific state.

Expects a K=2 checkpoint. Rolls out trajectories from the given state with the
first-step component forced (subsequent steps follow the normal mixture
sampling), colouring trajectories by which mode they started in, and pairs
that with the marginal histogram of sampled actions along the axis on which
the mixture splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal

from benchmark_policies import load_policy
from dynamics import UAV3D
from train_vi_bnn_rl import normalize_state


def _forced_component_action(policy, state_t: torch.Tensor, component_idx: int) -> np.ndarray:
    """Sample from a specific mixture component (used only for the first step)."""
    with torch.no_grad():
        means, log_stds, _ = policy.distribution_params(state_t)
        mean = means[:, component_idx, :]
        std = log_stds[:, component_idx, :].exp()
        raw = Normal(mean, std).sample()
        action = torch.tanh(raw) * policy.action_scale
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _rollout(
    policy,
    env: UAV3D,
    start_state: np.ndarray,
    horizon: int,
    first_component: int | None,
) -> tuple[np.ndarray, str]:
    """Run one rollout; first step uses ``first_component`` (if given),
    subsequent steps use the normal mixture sampling."""
    env._state = start_state.copy()
    env._step_count = 0
    env._done = False
    pos = np.array([start_state[0], start_state[2], start_state[4]], dtype=np.float32)
    env._prev_dist = float(np.linalg.norm(pos - env.goal_center))

    trace = [pos.copy()]
    outcome = "timeout"
    for step in range(horizon):
        normed = normalize_state(env._state)
        state_t = torch.from_numpy(normed).unsqueeze(0)
        if step == 0 and first_component is not None:
            action = _forced_component_action(policy, state_t, first_component)
        else:
            with torch.no_grad():
                action_t, _, _ = policy.sample(state_t)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

        _, _, next_state = env.step(action)
        trace.append(np.array([next_state[0], next_state[2], next_state[4]], dtype=np.float32))
        if env.is_done():
            final_pos = np.array([next_state[0], next_state[2], next_state[4]], dtype=np.float32)
            if env._check_goal(final_pos):
                outcome = "goal"
            elif env._check_obstacle_collision(final_pos) or env._outside_corridor(final_pos):
                outcome = "collision"
            break

    return np.stack(trace, axis=0), outcome


def _draw_xy_obstacles(ax, env: UAV3D) -> None:
    """Draw every obstacle's x-y footprint as an outlined rectangle. The
    central tower (z spanning the full corridor) is drawn with a darker fill
    so its role as the obstacle being routed around is visually obvious."""
    for (xm, xM, ym, yM, zm, zM) in env.obstacles:
        z_span = zM - zm
        is_tower = z_span >= 13.0  # Only the central tower spans the full z range.
        fill_alpha = 0.55 if is_tower else 0.18
        edge_alpha = 0.9 if is_tower else 0.4
        rect = plt.Rectangle(
            (xm, ym),
            xM - xm,
            yM - ym,
            facecolor=(0.4, 0.4, 0.4, fill_alpha),
            edgecolor=(0.15, 0.15, 0.15, edge_alpha),
            linewidth=1.0,
            zorder=1 if is_tower else 0,
        )
        ax.add_patch(rect)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/vi_bnn_policy.pt")
    parser.add_argument("--out", type=str, default="experiments/uav_3d/vi_bnn_bimodal_state.png")
    parser.add_argument("--x", type=float, default=-1.82)
    parser.add_argument("--y", type=float, default=-0.34)
    parser.add_argument("--z", type=float, default=0.68)
    parser.add_argument("--trajectories-per-mode", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--action-samples", type=int, default=5000)
    parser.add_argument("--split-axis", type=int, default=1, choices=[0, 1, 2],
                        help="Action axis for the marginal histogram (0=fx, 1=fy, 2=fz).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, _, _ = load_policy(Path(args.checkpoint))
    if policy.num_components != 2:
        raise ValueError(
            f"Script assumes K=2 components, checkpoint has K={policy.num_components}.")

    env = UAV3D(gaussian_noise=True, noise_std=0.05)
    start_state = np.array([args.x, 0.0, args.y, 0.0, args.z, 0.0], dtype=np.float32)
    normed = normalize_state(start_state)
    state_t = torch.from_numpy(normed).unsqueeze(0)

    with torch.no_grad():
        means_raw, log_stds, log_mix_w = policy.distribution_params(state_t)
        gate = log_mix_w.exp().squeeze(0).cpu().numpy()
        mus = (torch.tanh(means_raw) * policy.action_scale).squeeze(0).cpu().numpy()
        stds = log_stds.exp().squeeze(0).cpu().numpy()

    axis_names = ["fx", "fy", "fz"]
    print(f"State pos=({args.x}, {args.y}, {args.z}), zero velocity.")
    print(f"Gate: pi_0={gate[0]:.3f}, pi_1={gate[1]:.3f}")
    print(f"Mode 0 mean action: ({mus[0, 0]:+.2f}, {mus[0, 1]:+.2f}, {mus[0, 2]:+.2f}), std≈{stds[0].mean():.2f}")
    print(f"Mode 1 mean action: ({mus[1, 0]:+.2f}, {mus[1, 1]:+.2f}, {mus[1, 2]:+.2f}), std≈{stds[1].mean():.2f}")

    # Identify which component is "positive" vs "negative" along the split axis
    # so the legend labels match the plotted colours.
    if mus[0, args.split_axis] > mus[1, args.split_axis]:
        pos_mode, neg_mode = 0, 1
    else:
        pos_mode, neg_mode = 1, 0

    # Rollouts for both forced components
    trajectories_by_mode: dict[int, list[tuple[np.ndarray, str]]] = {0: [], 1: []}
    for k in (0, 1):
        for _ in range(args.trajectories_per_mode):
            trajectories_by_mode[k].append(
                _rollout(policy, env, start_state, args.horizon, first_component=k)
            )
    for k in (0, 1):
        outcomes = [o for _, o in trajectories_by_mode[k]]
        print(f"Mode {k}: {outcomes.count('goal')} goal / "
              f"{outcomes.count('collision')} collision / "
              f"{outcomes.count('timeout')} timeout "
              f"(of {len(outcomes)} rollouts)")

    # Sampled actions at the state for the marginal histogram
    with torch.no_grad():
        s_rep = state_t.repeat(args.action_samples, 1)
        sampled_actions, _, _ = policy.sample(s_rep)
    sampled_actions = sampled_actions.cpu().numpy()

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1.0])

    mode_colors = {pos_mode: "tab:blue", neg_mode: "tab:orange"}
    mode_labels = {
        pos_mode: f"Mode {pos_mode} ({axis_names[args.split_axis]}>0)  π={gate[pos_mode]:.2f}",
        neg_mode: f"Mode {neg_mode} ({axis_names[args.split_axis]}<0)  π={gate[neg_mode]:.2f}",
    }

    # Panel 1: top-down x-y
    ax_xy = fig.add_subplot(gs[0, 0])
    _draw_xy_obstacles(ax_xy, env)
    goal_rect = plt.Rectangle(
        (env.goal_min[0], env.goal_min[1]),
        env.goal_max[0] - env.goal_min[0],
        env.goal_max[1] - env.goal_min[1],
        facecolor=(0.3, 0.8, 0.3, 0.35),
        edgecolor="green",
        linewidth=1.2,
        zorder=1,
    )
    ax_xy.add_patch(goal_rect)

    for k in (0, 1):
        for idx, (trace, outcome) in enumerate(trajectories_by_mode[k]):
            lw = 1.4 if outcome == "goal" else 0.9
            alpha = 0.9 if outcome == "goal" else 0.4
            ax_xy.plot(
                trace[:, 0], trace[:, 1],
                color=mode_colors[k], alpha=alpha, linewidth=lw,
                label=mode_labels[k] if idx == 0 else None, zorder=3,
            )
    ax_xy.scatter([args.x], [args.y], s=160, marker="*", color="black",
                  zorder=6, label="Start")
    ax_xy.set_xlim(-15.5, 15.5)
    ax_xy.set_ylim(-9.5, 9.5)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title("Top-down trajectory projection (start colour = forced initial mode)")
    ax_xy.legend(loc="upper right", fontsize=9)
    ax_xy.set_aspect("equal")
    ax_xy.grid(alpha=0.25)

    # Panel 2: side view x-z
    ax_xz = fig.add_subplot(gs[0, 1])
    for (xm, xM, ym, yM, zm, zM) in env.obstacles:
        rect = plt.Rectangle(
            (xm, zm), xM - xm, zM - zm,
            facecolor=(0.4, 0.4, 0.4, 0.25),
            edgecolor=(0.15, 0.15, 0.15, 0.5),
            linewidth=0.8,
        )
        ax_xz.add_patch(rect)
    goal_rect_xz = plt.Rectangle(
        (env.goal_min[0], env.goal_min[2]),
        env.goal_max[0] - env.goal_min[0],
        env.goal_max[2] - env.goal_min[2],
        facecolor=(0.3, 0.8, 0.3, 0.35),
        edgecolor="green",
        linewidth=1.2,
    )
    ax_xz.add_patch(goal_rect_xz)
    for k in (0, 1):
        for idx, (trace, outcome) in enumerate(trajectories_by_mode[k]):
            lw = 1.4 if outcome == "goal" else 0.9
            alpha = 0.9 if outcome == "goal" else 0.4
            ax_xz.plot(
                trace[:, 0], trace[:, 2],
                color=mode_colors[k], alpha=alpha, linewidth=lw, zorder=3,
            )
    ax_xz.scatter([args.x], [args.z], s=160, marker="*", color="black", zorder=6)
    ax_xz.set_xlim(-15.5, 15.5)
    ax_xz.set_ylim(-7.5, 7.5)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("Side trajectory projection (x-z)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(alpha=0.25)

    # Panel 3: marginal histogram on the split axis
    ax_hist = fig.add_subplot(gs[0, 2])
    axis = args.split_axis
    ax_hist.hist(
        sampled_actions[:, axis], bins=80, range=(-4.2, 4.2),
        color="steelblue", edgecolor="black", linewidth=0.3,
    )
    ax_hist.axvline(mus[pos_mode, axis], color=mode_colors[pos_mode], linestyle="--",
                    linewidth=2, label=f"μ mode {pos_mode} = {mus[pos_mode, axis]:+.2f}")
    ax_hist.axvline(mus[neg_mode, axis], color=mode_colors[neg_mode], linestyle="--",
                    linewidth=2, label=f"μ mode {neg_mode} = {mus[neg_mode, axis]:+.2f}")
    ax_hist.set_xlabel(f"{axis_names[axis]} action sample")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Sampled {axis_names[axis]} marginal ({args.action_samples} samples)")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(alpha=0.25)

    fig.suptitle(
        f"Bimodal policy at s = ({args.x:+.2f}, {args.y:+.2f}, {args.z:+.2f}) [zero velocity]  "
        f"— ‖μ_0 − μ_1‖ = {np.linalg.norm(mus[0] - mus[1]):.2f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved bimodal visualization to {out_path}")


if __name__ == "__main__":
    main()
