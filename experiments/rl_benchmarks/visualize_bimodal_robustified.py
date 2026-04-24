#!/usr/bin/env python3
"""Robustify the VI-BNN policy's action distribution at the bimodal state via
(alpha-trimmed) randomized smoothing and compare to the raw action distribution.

The policy is wrapped in ``UAV3DPolicyStochasticFunction``; smoothing adds
Gaussian noise to the (normalized) state and trims per-cluster means of the
resulting action samples, yielding a ``MixtureDirac`` over cluster centroids.

Outputs a single figure with:
  * 3D scatters of raw actions, state-perturbed actions, and robustified
    (MixtureDirac) action samples, each colored by component/cluster.
  * Per-axis marginal histograms overlaying raw vs. state-perturbed vs.
    robustified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS = Path(__file__).resolve()
RL_DIR = THIS.parent
EXPERIMENTS = RL_DIR.parent
SRC = EXPERIMENTS.parent / "src"
for p in (str(SRC), str(EXPERIMENTS), str(RL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_policies import load_policy
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_vi_bnn_rl import normalize_state
from visualize_uav3d_pyvista import BIMODAL_START_STATE

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing


AXIS_NAMES = ("fx", "fy", "fz")


def _assign_to_centroids(actions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = actions[:, None, :] - centroids[None, :, :]
    return np.argmin((diff * diff).sum(axis=-1), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/vi_bnn_policy.pt")
    parser.add_argument("--out", type=str, default="experiments/uav_3d/vi_bnn_bimodal_robustified.png")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Std of the Gaussian state-space smoothing noise (normalized state units).")
    parser.add_argument("--smoothing-N", type=int, default=4000,
                        help="Number of noisy state samples per smoothing call.")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha-trimming fraction (0 <= alpha < 0.5).")
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.45)
    parser.add_argument("--dbscan-min-samples", type=int, default=50)
    parser.add_argument("--raw-samples", type=int, default=4000)
    parser.add_argument("--mc-samples", type=int, default=4000,
                        help="Draws from the robustified MixtureDirac for plotting.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, _, _ = load_policy(Path(args.checkpoint))
    if policy.num_components < 2:
        print(f"[warn] Checkpoint has K={policy.num_components}; bimodality visualization assumes K>=2.")

    base_state = BIMODAL_START_STATE.astype(np.float32)
    normed = normalize_state(base_state)
    x = torch.from_numpy(normed).to(torch.float32).unsqueeze(0)  # shape (1, state_dim)

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

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=args.sigma,
        parameters=parameters,
    )
    centroids = smoother.cluster.centroids[0].detach().cpu().numpy()  # (K, 3)
    probs_partition = smoother.cluster.probs_partition[0].detach().cpu().numpy()
    k_used = int(smoother.cluster.k_per_row[0].item())
    print(f"Clusters used: {k_used} (padded to {smoother.cluster.num_clusters})")
    print(f"Partition weights: {np.round(probs_partition, 3).tolist()}")
    print(f"Centroids (K x 3): {np.round(centroids, 3).tolist()}")

    # Raw actions: policy.sample at the base (unperturbed) state.
    with torch.no_grad():
        s_rep = x.repeat(args.raw_samples, 1)
        raw_actions_t, _, _ = policy.sample(s_rep)
    raw_actions = raw_actions_t.cpu().numpy()

    # State-perturbed actions: policy.sample at state + Gaussian noise. These
    # are the *raw outputs* that feed the smoother — they show how much the
    # action distribution spreads before alpha-trimming.
    with torch.no_grad():
        noise = args.sigma * torch.randn(args.raw_samples, x.shape[-1], dtype=x.dtype)
        s_noisy = x + noise
        perturbed_actions_t, _, _ = policy.sample(s_noisy)
    perturbed_actions = perturbed_actions_t.cpu().numpy()

    # Robustified MixtureDirac samples.
    mc_samples = smoother.monte_carlo(num_simulations=args.mc_samples)  # (1, M, 3)
    mc_samples = mc_samples.squeeze(0).detach().cpu().numpy()

    # Assign each sample to a centroid for coloring.
    raw_assign = _assign_to_centroids(raw_actions, centroids)
    pert_assign = _assign_to_centroids(perturbed_actions, centroids)
    mc_assign = _assign_to_centroids(mc_samples, centroids)

    cluster_colors = plt.get_cmap("tab10")(np.arange(max(3, k_used)))

    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0])

    def _scatter3d(ax, pts, assign, title):
        for j in range(k_used):
            mask = assign == j
            if not np.any(mask):
                continue
            ax.scatter(
                pts[mask, 0], pts[mask, 1], pts[mask, 2],
                s=5, alpha=0.35, color=cluster_colors[j],
                label=f"cluster {j} (π={probs_partition[j]:.2f})",
            )
        ax.scatter(
            centroids[:k_used, 0], centroids[:k_used, 1], centroids[:k_used, 2],
            s=180, marker="X", edgecolor="black", linewidth=1.2,
            color=[cluster_colors[j] for j in range(k_used)],
            zorder=10, label="trimmed centroids",
        )
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.set_zlim(-4.2, 4.2)
        ax.set_xlabel("fx"); ax.set_ylabel("fy"); ax.set_zlabel("fz")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    ax_raw = fig.add_subplot(gs[0, 0], projection="3d")
    _scatter3d(ax_raw, raw_actions, raw_assign,
               f"Raw policy actions at s\n(N = {args.raw_samples})")

    ax_pert = fig.add_subplot(gs[0, 1], projection="3d")
    _scatter3d(ax_pert, perturbed_actions, pert_assign,
               f"State-perturbed actions: policy.sample(s + N(0, σ²I))\n"
               f"σ = {args.sigma} (normalized state units), N = {args.raw_samples}")

    ax_mc = fig.add_subplot(gs[0, 2], projection="3d")
    _scatter3d(ax_mc, mc_samples, mc_assign,
               f"Robustified action distribution (α-trimmed MixtureDirac)\n"
               f"α = {args.alpha}, N = {args.mc_samples}")

    for axis in range(3):
        ax = fig.add_subplot(gs[1, axis])
        bins = np.linspace(-4.2, 4.2, 81)
        ax.hist(raw_actions[:, axis], bins=bins, alpha=0.55,
                color="steelblue", label="raw (policy @ s)", density=True)
        ax.hist(perturbed_actions[:, axis], bins=bins, alpha=0.45,
                color="darkorange",
                label=f"state-perturbed (σ={args.sigma})", density=True)
        ax.set_xlim(-4.2, 4.2)
        ax.set_xlabel(f"{AXIS_NAMES[axis]} action")
        ax.set_ylabel("density")
        ax.set_title(f"{AXIS_NAMES[axis]} marginal")
        ax.grid(alpha=0.2)

        # Overlay the robustified MixtureDirac as stems on a twin axis:
        # its density is a sum of weighted point masses at the centroids,
        # so a probability-mass stem plot reads much better than a histogram.
        ax_stem = ax.twinx()
        for j in range(k_used):
            ax_stem.vlines(
                centroids[j, axis], 0, probs_partition[j],
                color=cluster_colors[j], linewidth=3.0,
                label=f"robustified π_{j} @ {centroids[j, axis]:+.2f}",
            )
            ax_stem.scatter(
                [centroids[j, axis]], [probs_partition[j]],
                color=cluster_colors[j], s=60, zorder=5,
                edgecolor="black", linewidth=0.8,
            )
        ax_stem.set_ylim(0, 1.0)
        ax_stem.set_ylabel("robustified prob. mass")
        if axis == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_stem.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    pos = (base_state[0], base_state[2], base_state[4])
    fig.suptitle(
        f"Robustified action distribution at bimodal state "
        f"s = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
        f"— σ = {args.sigma}, α = {args.alpha}, N = {args.smoothing_N}",
        fontsize=13,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved robustification figure to {out_path}")


if __name__ == "__main__":
    main()
