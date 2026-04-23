#!/usr/bin/env python3
"""Plot a 3D scatter of sampled policy actions at ``BIMODAL_START_STATE``."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_policies import load_policy
from train_vi_bnn_rl import normalize_state
from visualize_uav3d_pyvista import BIMODAL_START_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/vi_bnn_policy.pt")
    parser.add_argument("--out", type=str, default="experiments/uav_3d/vi_bnn_bimodal_actions.png")
    parser.add_argument("--action-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, _, _ = load_policy(Path(args.checkpoint))
    state = BIMODAL_START_STATE.astype(np.float32)
    pos = (state[0], state[2], state[4])
    state_t = torch.from_numpy(normalize_state(state)).unsqueeze(0)

    with torch.no_grad():
        sampled, _, _ = policy.sample(state_t.repeat(args.action_samples, 1))
    sampled = sampled.cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sampled[:, 0], sampled[:, 1], sampled[:, 2],
        s=4, alpha=0.4, c="steelblue",
    )
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(-4.2, 4.2)
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")
    ax.set_zlabel("fz")
    ax.set_title(
        f"Sampled actions at s = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
        f"  (N = {args.action_samples})"
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved action-scatter plot to {out_path}")


if __name__ == "__main__":
    main()
