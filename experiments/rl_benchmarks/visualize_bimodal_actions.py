#!/usr/bin/env python3
"""Plot the action distribution at ``BIMODAL_START_STATE`` (imported from
``visualize_uav3d_pyvista``).

Samples many actions from the trained policy at the fixed state, then draws
the three 2D marginals (fx-fy, fy-fz, fx-fz) and the three 1D marginals
(fx, fy, fz). Component means are overlaid on every panel, sized by the
mixture weight so the dominant mode is visually obvious.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_policies import load_policy
from train_vi_bnn_rl import normalize_state
from visualize_uav3d_pyvista import BIMODAL_START_STATE

AXIS_NAMES = ("fx", "fy", "fz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/vi_bnn_policy.pt")
    parser.add_argument("--out", type=str, default="experiments/uav_3d/vi_bnn_bimodal_actions.png")
    parser.add_argument("--action-samples", type=int, default=20000)
    parser.add_argument("--bins-1d", type=int, default=80)
    parser.add_argument("--bins-2d", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, _, _ = load_policy(Path(args.checkpoint))
    K = policy.num_components
    state = BIMODAL_START_STATE.astype(np.float32)
    pos = (state[0], state[2], state[4])
    normed = normalize_state(state)
    state_t = torch.from_numpy(normed).unsqueeze(0)

    with torch.no_grad():
        means_raw, log_stds, log_mix_w = policy.distribution_params(state_t)
        gate = log_mix_w.exp().squeeze(0).cpu().numpy()
        mus = (torch.tanh(means_raw) * policy.action_scale).squeeze(0).cpu().numpy()
        stds = log_stds.exp().squeeze(0).cpu().numpy()
        sampled, _, _ = policy.sample(state_t.repeat(args.action_samples, 1))
    sampled = sampled.cpu().numpy()

    print(f"State pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}), zero velocity.")
    print(f"K = {K} components")
    for k in range(K):
        print(f"  mode {k}: pi={gate[k]:.3f}  mu=({mus[k, 0]:+.2f}, {mus[k, 1]:+.2f}, {mus[k, 2]:+.2f})  "
              f"std=({stds[k, 0]:.2f}, {stds[k, 1]:.2f}, {stds[k, 2]:.2f})")

    # Colour palette for components (up to K=6).
    palette = ["tab:orange", "tab:cyan", "tab:purple", "tab:olive", "tab:red", "tab:brown"]
    comp_colors = [palette[k % len(palette)] for k in range(K)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Top row: 2D joint marginals, one per axis pair.
    for col, (i, j) in enumerate(combinations(range(3), 2)):
        ax = axes[0, col]
        ax.hist2d(
            sampled[:, i], sampled[:, j],
            bins=args.bins_2d,
            range=[[-4.2, 4.2], [-4.2, 4.2]],
            cmap="viridis",
            cmin=1,
        )
        for k in range(K):
            ax.scatter(
                mus[k, i], mus[k, j],
                marker="X",
                s=80 + 400.0 * gate[k],  # size ~ mixture weight
                facecolor=comp_colors[k],
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
                label=f"mode {k} (π={gate[k]:.2f})",
            )
        ax.set_xlabel(AXIS_NAMES[i])
        ax.set_ylabel(AXIS_NAMES[j])
        ax.set_title(f"{AXIS_NAMES[i]} vs {AXIS_NAMES[j]} joint")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8)

    # Bottom row: 1D marginals per axis.
    for col, axis in enumerate(range(3)):
        ax = axes[1, col]
        ax.hist(
            sampled[:, axis],
            bins=args.bins_1d,
            range=(-4.2, 4.2),
            color="steelblue",
            edgecolor="black",
            linewidth=0.3,
        )
        for k in range(K):
            ax.axvline(
                mus[k, axis],
                color=comp_colors[k],
                linestyle="--",
                linewidth=2.0,
                label=f"μ mode {k} = {mus[k, axis]:+.2f}",
            )
        ax.set_xlabel(f"{AXIS_NAMES[axis]} action")
        ax.set_ylabel("count")
        ax.set_title(f"{AXIS_NAMES[axis]} marginal")
        ax.set_xlim(-4.2, 4.2)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    sep = np.linalg.norm(mus[0] - mus[1]) if K == 2 else float("nan")
    fig.suptitle(
        f"Policy action distribution at s = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) "
        f"[zero velocity] — N={args.action_samples} samples, K={K}"
        + (f", ‖μ₀−μ₁‖ = {sep:.2f}" if K == 2 else ""),
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved action-distribution plot to {out_path}")


if __name__ == "__main__":
    main()
