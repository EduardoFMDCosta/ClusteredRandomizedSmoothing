"""Render three separate figures of the policy's action distribution at the
bimodal state:

  (i)   ``bimodal_raw.png``      — samples from the raw stochastic policy.
  (ii)  ``bimodal_smoothed.png`` — Monte-Carlo draws from the clusterized
                                    alpha-trimmed smoothed predictor (K clusters)
                                    overlaid with the per-cluster output
                                    robustness regions.
  (iii) ``bimodal_alpha.png``    — Monte-Carlo draws from the
                                    ``max_clusters=1`` alpha-smoothing baseline
                                    overlaid with its single output robustness
                                    region.

For (ii) and (iii) the script also runs ``smoother.robust_certification`` at
the chosen input disturbance radius and prints the certified per-cluster /
total probabilities so they can be cited in the paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from benchmark_policies import load_policy
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import normalize_state
from visualize_uav3d_pyvista import BIMODAL_START_STATE

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

AXIS_LIM = 4.2


def _assign_to_centroids(actions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = actions[:, None, :] - centroids[None, :, :]
    return np.argmin((diff * diff).sum(axis=-1), axis=-1)


def _box_edges(lower: np.ndarray, upper: np.ndarray) -> list[tuple[list[float], list[float], list[float]]]:
    xs = (lower[0], upper[0])
    ys = (lower[1], upper[1])
    zs = (lower[2], upper[2])
    corners = [(x, y, z) for x in xs for y in ys for z in zs]
    edge_idx = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    edges = []
    for a, b in edge_idx:
        ca, cb = corners[a], corners[b]
        edges.append(([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]]))
    return edges


def _draw_region(ax, lower: np.ndarray, upper: np.ndarray, color, label: str | None) -> None:
    for i, (xe, ye, ze) in enumerate(_box_edges(lower, upper)):
        ax.plot(xe, ye, ze, color=color, linewidth=2.0, alpha=0.9,
                label=label if i == 0 else None)


def _new_3d_fig():
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_zlim(-AXIS_LIM, AXIS_LIM)
    ax.set_xlabel("fx")
    ax.set_ylabel("fy")
    ax.set_zlabel("fz")
    return fig, ax


def _plot_raw(actions: np.ndarray, out_path: Path, base_state: np.ndarray) -> None:
    fig, ax = _new_3d_fig()
    ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2],
               s=5, alpha=0.35, color="steelblue", label=f"raw samples (N={len(actions)})")
    pos = (base_state[0], base_state[2], base_state[4])
    ax.set_title(
        f"Raw policy actions at s = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_smoothed(
    smoother: GaussianAlphaTrimmingSmoothing,
    mc_samples: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    centroids = smoother.cluster.centroids[0].detach().cpu().numpy()
    probs_partition = smoother.cluster.probs_partition[0].detach().cpu().numpy()
    k_used = int(smoother.cluster.k_per_row[0].item())
    region_lower = smoother.regions.lower[0].detach().cpu().numpy()
    region_upper = smoother.regions.upper[0].detach().cpu().numpy()

    assign = _assign_to_centroids(mc_samples, centroids)
    cluster_colors = plt.get_cmap("tab10")(np.arange(max(3, k_used)))

    fig, ax = _new_3d_fig()
    for j in range(k_used):
        mask = assign == j
        if not np.any(mask):
            continue
        ax.scatter(
            mc_samples[mask, 0], mc_samples[mask, 1], mc_samples[mask, 2],
            s=5, alpha=0.35, color=cluster_colors[j],
            label=f"cluster {j} (π={probs_partition[j]:.2f})",
        )
    ax.scatter(
        centroids[:k_used, 0], centroids[:k_used, 1], centroids[:k_used, 2],
        s=180, marker="X", edgecolor="black", linewidth=1.2,
        color=[cluster_colors[j] for j in range(k_used)],
        zorder=10, label="trimmed centroids",
    )
    for j in range(k_used):
        _draw_region(
            ax, region_lower[j], region_upper[j], cluster_colors[j],
            label=f"region {j}",
        )

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_smoother(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    x: torch.Tensor,
    sigma: float,
    smoothing_N: int,
    alpha: float,
    coverage: float,
    max_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    set_type: str,
) -> GaussianAlphaTrimmingSmoothing:
    parameters = Parameters(
        smoothing=SmoothingParameters(N=smoothing_N, alpha=alpha),
        clustering=ClusteringParameters(
            max_num_clusters=max_clusters,
            coverage=coverage,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            set_type=set_type,
        ),
        confidence=ConfidenceParameters(num_samples=smoothing_N, beta=1e-2),
    )
    return GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=parameters,
    )


def _print_certificate(
    smoother: GaussianAlphaTrimmingSmoothing,
    sigma: float,
    radius: float,
    label: str,
) -> None:
    x = smoother.x
    radius_t = radius * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    certificate = smoother.robust_certification(radius=radius_t)
    probs = certificate.probs[0].detach().cpu().numpy()
    total = float(certificate.total_probs[0].detach().cpu().item())
    k_used = int(smoother.cluster.k_per_row[0].item())
    region_lower = smoother.regions.lower[0].detach().cpu().numpy()
    region_upper = smoother.regions.upper[0].detach().cpu().numpy()
    region_widths = (region_upper - region_lower)[:k_used]

    print(f"\n[{label}] robustness certification")
    print(f"  input disturbance radius (L2, normalized state units): {radius:.4f}")
    print(f"  smoothing sigma (state-space): {sigma:.4f}")
    print(f"  clusters used: {k_used}")
    print(f"  per-cluster certified probability: "
          f"{np.round(probs[:k_used], 4).tolist()}")
    print(f"  total certified probability:       {total:.4f}")
    print("  output region half-widths (per cluster, per axis):")
    for j in range(k_used):
        hw = 0.5 * region_widths[j]
        print(f"    cluster {j}: [±{hw[0]:.3f}, ±{hw[1]:.3f}, ±{hw[2]:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--out-dir", type=str, default="experiments/uav_3d/bimodal_plots")
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
    parser.add_argument("--radius", type=float, default=0.02,
                        help="Input disturbance L2 radius (normalized state units) for the certificate.")
    parser.add_argument("--set-type", type=str, default="AxisAlignedHyperRectangle",
                        choices=["HyperRectangle", "AxisAlignedHyperRectangle",
                                 "EuclideanBall", "TimeStepHyperrectangles"],
                        help="Output region shape per cluster: 'HyperRectangle' is an L∞ "
                             "cube (shared half-width), 'AxisAlignedHyperRectangle' is the "
                             "tightest axis-aligned box around the L2-closest 'coverage' "
                             "fraction of cluster samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy, _, _ = load_policy(Path(args.checkpoint))
    if policy.num_components < 2:
        print(f"[warn] Checkpoint has K={policy.num_components}; bimodality visualization assumes K>=2.")

    base_state = BIMODAL_START_STATE.astype(np.float32)
    normed = normalize_state(base_state)
    x = torch.from_numpy(normed).to(torch.float32).unsqueeze(0)

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = (base_state[0], base_state[2], base_state[4])
    state_str = f"s = ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"

    # ---- (i) Raw predictor samples ----
    with torch.no_grad():
        s_rep = x.repeat(args.raw_samples, 1)
        raw_actions_t, _, _ = policy.sample(s_rep)
    raw_actions = raw_actions_t.cpu().numpy()
    raw_path = out_dir / "bimodal_raw.png"
    _plot_raw(raw_actions, raw_path, base_state)
    print(f"Saved raw predictor figure to {raw_path}")

    # ---- (ii) Clusterized smoothed predictor (K = max_clusters) ----
    smoother_K = _build_smoother(
        stochastic_fn=stochastic_fn, x=x, sigma=args.sigma,
        smoothing_N=args.smoothing_N, alpha=args.alpha, coverage=args.coverage,
        max_clusters=args.max_clusters,
        dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
        set_type=args.set_type,
    )
    mc_K = smoother_K.monte_carlo(num_simulations=args.mc_samples).squeeze(0).detach().cpu().numpy()
    smoothed_path = out_dir / "bimodal_smoothed.png"
    _plot_smoothed(
        smoother_K, mc_K, smoothed_path,
        title=(f"Smoothed predictor at {state_str}\n"
               f"σ = {args.sigma}, α = {args.alpha}, max_clusters = {args.max_clusters}, "
               f"coverage = {args.coverage}"),
    )
    print(f"Saved smoothed predictor figure to {smoothed_path}")
    _print_certificate(smoother_K, sigma=args.sigma, radius=args.radius,
                       label="smoothed predictor")

    # ---- (iii) Alpha-smoothing baseline (K = 1) ----
    smoother_1 = _build_smoother(
        stochastic_fn=stochastic_fn, x=x, sigma=args.sigma,
        smoothing_N=args.smoothing_N, alpha=args.alpha, coverage=args.coverage,
        max_clusters=1,
        dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
        set_type=args.set_type,
    )
    mc_1 = smoother_1.monte_carlo(num_simulations=args.mc_samples).squeeze(0).detach().cpu().numpy()
    alpha_path = out_dir / "bimodal_alpha.png"
    _plot_smoothed(
        smoother_1, mc_1, alpha_path,
        title=(f"Alpha-smoothing (max_clusters = 1) at {state_str}\n"
               f"σ = {args.sigma}, α = {args.alpha}, coverage = {args.coverage}"),
    )
    print(f"Saved alpha-smoothing figure to {alpha_path}")
    _print_certificate(smoother_1, sigma=args.sigma, radius=args.radius,
                       label="alpha-smoothing (K=1)")


if __name__ == "__main__":
    main()
