"""Scan all valid TrajFlow inputs to identify those whose noisy output distribution
``h_w(x + epsilon)`` is genuinely multimodal.

For each input x the script:
  1. Draws ``--num-noisy-samples`` stochastic model evaluations under additive Gaussian noise.
  2. Clusters the resulting 2-D output cloud with DBSCAN (configurable eps/min_samples).
  3. Computes diagnostics:
       - num_dbscan_clusters    : DBSCAN cluster count (noise points excluded)
       - noise_fraction         : fraction of output samples labelled as noise by DBSCAN
       - cluster_fractions      : per-cluster sample fractions, sorted descending
       - secondary_fraction     : fraction of samples in the 2nd largest cluster
       - centroid_separation    : L2 distance between the two largest-cluster centroids
       - within_cluster_spread  : mean per-cluster average distance to centroid
       - separation_ratio       : centroid_separation / within_cluster_spread
       - silhouette_score       : sklearn silhouette score on core (non-noise) points
  4. Classifies the input as "truly multimodal" if:
         secondary_fraction >= --min-secondary-fraction
       AND separation_ratio  >= --min-separation-ratio

Results are saved to ``multimodal_scan.json`` and two diagnostic figures.
Printed summary identifies the multimodal dataset indices.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.utils import _run_dbscan
from stochastic_functions import TrajFlowMeszarosStochasticFunction

try:
    from sklearn.metrics import silhouette_score as _sklearn_silhouette
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data helpers (shared pattern with other experiment scripts)
# ---------------------------------------------------------------------------

def _resolve_gt_file(data_dir: Path) -> Path:
    candidates = [
        data_dir / "Outputs_gt.npy",
        data_dir / "Output_gt.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find ground-truth output file. Expected one of: "
        f"{', '.join(str(p) for p in candidates)}"
    )


def _load_inputs_and_outputs() -> tuple[np.ndarray, np.ndarray]:
    data_dir = EXPERIMENTS / "TrajFlowExperiment" / "Data"
    inputs = np.load(str(data_dir / "Inputs.npy"))
    outputs_gt = np.load(str(_resolve_gt_file(data_dir)))
    return inputs, outputs_gt


def _finite_target_mask(outputs_gt: np.ndarray) -> np.ndarray:
    return np.isfinite(outputs_gt[:, 0, -1, :]).all(axis=1)


def _prepare_position_tensor(
    selected_indices: np.ndarray,
    inputs: np.ndarray,
) -> torch.Tensor:
    """Return (nx, 40) float32 position tensor for the given dataset indices."""
    x_input = inputs[selected_indices, :, :, :2]
    x_input = x_input[:, [1, 0]]
    return torch.from_numpy(x_input.reshape(len(selected_indices), -1)).float()


def _extract_target_initial_position(x_positions: torch.Tensor, num_input_agents: int = 2) -> torch.Tensor:
    nx = x_positions.shape[0]
    num_input_timesteps = x_positions.shape[1] // (2 * num_input_agents)
    x_traj = x_positions.reshape(nx, num_input_agents, num_input_timesteps, 2)
    return x_traj[:, 0, 0, :]


# ---------------------------------------------------------------------------
# Multimodality diagnostics for one input cloud (numpy, cpu)
# ---------------------------------------------------------------------------

def _cluster_diagnostics(
    output_cloud: np.ndarray,
    cluster_strategy: str,
    behavior_anchor: np.ndarray | None,
    dbscan_eps: float,
    dbscan_min_samples: int,
    behavior_stay_radius: float,
    behavior_min_down: float,
    behavior_min_left: float,
    min_secondary_fraction: float,
    min_separation_ratio: float,
) -> dict[str, Any]:
    """Compute full multimodality statistics for a single (N, 2) output cloud."""
    n = len(output_cloud)

    cloud_t = torch.from_numpy(output_cloud.astype(np.float32))
    if cluster_strategy == "dbscan":
        labels_t = _run_dbscan(cloud_t, eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = labels_t.numpy()
        noise_mask = labels == -1
    elif cluster_strategy == "behavior_split":
        if behavior_anchor is None:
            raise ValueError("behavior_anchor is required for behavior_split")
        cluster = Cluster(
            outputs=cloud_t.unsqueeze(0),
            parameters=ClusteringParameters(
                max_num_clusters=2,
                coverage=1.0,
                strategy="behavior_split",
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                behavior_stay_radius=behavior_stay_radius,
                behavior_min_down=behavior_min_down,
                behavior_min_left=behavior_min_left,
            ),
            behavior_anchor=torch.from_numpy(behavior_anchor.astype(np.float32)).unsqueeze(0),
        )
        labels = cluster.assignments[0].detach().cpu().numpy()
        noise_mask = np.zeros(n, dtype=bool)
    else:
        raise ValueError(f"Unsupported cluster_strategy: {cluster_strategy}")

    noise_fraction = float(noise_mask.mean())
    core_labels = labels[~noise_mask]
    core_points = output_cloud[~noise_mask]

    if core_labels.size == 0:
        num_clusters = 0
    else:
        num_clusters = int(core_labels.max()) + 1

    # Per-cluster fractions (relative to total N, including noise points).
    cluster_ids = list(range(num_clusters))
    raw_fractions = [
        float((labels == c).sum()) / n for c in cluster_ids
    ]
    # Sort descending by fraction so index 0 = dominant cluster.
    order = np.argsort(raw_fractions)[::-1]
    cluster_fractions_sorted = [raw_fractions[o] for o in order]
    sorted_cluster_ids = [cluster_ids[o] for o in order]

    secondary_fraction = cluster_fractions_sorted[1] if num_clusters >= 2 else 0.0
    centroid_separation = math.nan
    within_cluster_spread = math.nan
    separation_ratio = math.nan
    silhouette = math.nan

    if num_clusters >= 2:
        # Centroids of the two largest clusters.
        c0_pts = core_points[core_labels == sorted_cluster_ids[0]]
        c1_pts = core_points[core_labels == sorted_cluster_ids[1]]
        c0 = c0_pts.mean(axis=0)
        c1 = c1_pts.mean(axis=0)
        centroid_separation = float(np.linalg.norm(c0 - c1))

        # Within-cluster spread: per-cluster mean L2 distance to centroid.
        spreads: list[float] = []
        for cid in sorted_cluster_ids[:num_clusters]:
            pts = core_points[core_labels == cid]
            if len(pts) >= 2:
                centroid = pts.mean(axis=0)
                dists = np.linalg.norm(pts - centroid, axis=1)
                spreads.append(float(dists.mean()))
        within_cluster_spread = float(np.mean(spreads)) if spreads else 0.0

        if within_cluster_spread > 1e-9:
            separation_ratio = centroid_separation / within_cluster_spread
        else:
            separation_ratio = float("inf")

        # Silhouette score on core points.
        if _SKLEARN_AVAILABLE and len(core_points) >= 2 and len(set(core_labels.tolist())) >= 2:
            silhouette = float(_sklearn_silhouette(core_points, core_labels))

    is_multimodal = (
        num_clusters >= 2
        and secondary_fraction >= min_secondary_fraction
        and (math.isnan(separation_ratio) or separation_ratio >= min_separation_ratio)
    )

    return {
        "num_clusters": num_clusters,
        "num_dbscan_clusters": num_clusters,
        "cluster_strategy": cluster_strategy,
        "noise_fraction": noise_fraction,
        "cluster_fractions": cluster_fractions_sorted,
        "secondary_fraction": secondary_fraction,
        "centroid_separation": centroid_separation if not math.isnan(centroid_separation) else None,
        "within_cluster_spread": within_cluster_spread if not math.isnan(within_cluster_spread) else None,
        "separation_ratio": separation_ratio if not math.isinf(separation_ratio) and not math.isnan(separation_ratio) else None,
        "silhouette_score": silhouette if not math.isnan(silhouette) else None,
        "is_multimodal": is_multimodal,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _plot_multimodality_scatter(
    rows: list[dict[str, Any]],
    min_secondary_fraction: float,
    min_separation_ratio: float,
    output_path: Path,
) -> None:
    """Scatter plot: secondary_fraction vs separation_ratio, highlighting multimodal inputs."""
    _configure_plot_style()

    sec_fracs = np.array([r["secondary_fraction"] for r in rows], dtype=float)
    sep_ratios = np.array(
        [r["separation_ratio"] if r["separation_ratio"] is not None else 0.0 for r in rows],
        dtype=float,
    )
    is_mm = np.array([r["is_multimodal"] for r in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(
        sec_fracs[~is_mm],
        sep_ratios[~is_mm],
        s=20,
        alpha=0.45,
        color="#888888",
        label="Unimodal",
    )
    ax.scatter(
        sec_fracs[is_mm],
        sep_ratios[is_mm],
        s=40,
        alpha=0.85,
        color="#E4572E",
        label="Multimodal",
        zorder=3,
    )
    ax.axvline(min_secondary_fraction, color="#4C78A8", linestyle="--", linewidth=1.0, alpha=0.7, label=f"secondary_fraction = {min_secondary_fraction}")
    ax.axhline(min_separation_ratio, color="#2A9D8F", linestyle="--", linewidth=1.0, alpha=0.7, label=f"separation_ratio = {min_separation_ratio}")
    ax.set_xlabel("Secondary cluster fraction")
    ax.set_ylabel("Separation ratio")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_cluster_count_histogram(
    rows: list[dict[str, Any]],
    cluster_strategy: str,
    output_path: Path,
) -> None:
    """Bar chart of how many inputs have 1, 2, 3, ... clusters."""
    _configure_plot_style()

    counts_raw = [r["num_clusters"] for r in rows]
    max_k = max(counts_raw) if counts_raw else 1
    bins = np.arange(0, max_k + 2) - 0.5
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(counts_raw, bins=bins, color="#4C78A8", alpha=0.75, edgecolor="#4C78A8", linewidth=0.8)
    ax.set_xlabel(f"{cluster_strategy} cluster count per input")
    ax.set_ylabel("Number of inputs")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all valid TrajFlow inputs to find those whose noisy output "
            "distribution h_w(x+eps) is genuinely multimodal."
        )
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=813,
        help="How many valid inputs to scan (taken from the start of the valid index list).",
    )
    parser.add_argument(
        "--num-noisy-samples",
        type=int,
        default=300,
        help="Number of noisy model evaluations h_w(x+eps_i) drawn per input.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=10,
        help="Number of inputs processed at once.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Std of the Gaussian noise added to model inputs.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=("control", "position"),
        default="control",
        help="Input representation (control = smooth in control space u).",
    )
    parser.add_argument(
        "--cluster-strategy",
        type=str,
        choices=("dbscan", "behavior_split"),
        default="dbscan",
        help="Clustering strategy used to split each output cloud.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN neighborhood radius (applied to the 2-D output space).",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=15,
        help="DBSCAN minimum neighborhood size to form a core point.",
    )
    parser.add_argument(
        "--behavior-stay-radius",
        type=float,
        default=1.0,
        help="Stay mode threshold: ||y - y_init|| <= radius (behavior_split only).",
    )
    parser.add_argument(
        "--behavior-min-down",
        type=float,
        default=0.5,
        help="Turn mode threshold on axis x2: y2 - y2_init <= -min_down (behavior_split only).",
    )
    parser.add_argument(
        "--behavior-min-left",
        type=float,
        default=0.0,
        help="Turn mode threshold on axis x1: y1 - y1_init <= -min_left (behavior_split only).",
    )
    parser.add_argument(
        "--min-secondary-fraction",
        type=float,
        default=0.10,
        help="Minimum fraction of samples in the 2nd cluster to be classified as multimodal.",
    )
    parser.add_argument(
        "--min-separation-ratio",
        type=float,
        default=1.5,
        help=(
            "Minimum separation ratio (centroid distance / within-cluster spread) "
            "to be classified as multimodal."
        ),
    )
    parser.add_argument(
        "--noise-sub-batch-size",
        type=int,
        default=300,
        help="Sub-batch size for the noise/sample dimension in stochastic sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis"),
        help="Directory where the JSON results file is saved.",
    )
    args = parser.parse_args()

    if args.num_noisy_samples <= 0:
        raise ValueError("num-noisy-samples must be positive")
    if args.input_batch_size <= 0:
        raise ValueError("input-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if not 0.0 < args.min_secondary_fraction < 1.0:
        raise ValueError("min-secondary-fraction must be in (0, 1)")
    if args.min_separation_ratio <= 0.0:
        raise ValueError("min-separation-ratio must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stochastic_fn = TrajFlowMeszarosStochasticFunction()

    print("Loading dataset...")
    all_inputs, outputs_gt = _load_inputs_and_outputs()
    valid_indices = np.flatnonzero(_finite_target_mask(outputs_gt))
    num_to_scan = min(args.num_inputs, len(valid_indices))
    selected_indices = valid_indices[:num_to_scan].astype(int)
    print(f"Scanning {num_to_scan} valid inputs (of {len(valid_indices)} total valid).")

    per_input_results: list[dict[str, Any]] = []
    num_batches = math.ceil(len(selected_indices) / args.input_batch_size)

    for batch_id, start in enumerate(
        range(0, len(selected_indices), args.input_batch_size), start=1
    ):
        batch_t0 = time.perf_counter()
        batch_indices = selected_indices[start : start + args.input_batch_size]
        nx = len(batch_indices)

        x_position = _prepare_position_tensor(batch_indices, all_inputs)
        target_init = _extract_target_initial_position(x_position).detach().cpu().numpy()

        if args.input_mode == "position":
            model_input: torch.Tensor = x_position
            helper: list[torch.Tensor] = []
        else:
            model_input, helper = stochastic_fn.pos_to_ctrl(x_position)

        noise_dist = MultivariateGaussian(
            mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype),
            cov=(args.sigma ** 2)
            * torch.eye(model_input.shape[-1], dtype=model_input.dtype),
        )

        # Draw noisy samples: shape (nx, num_noisy_samples, 2)
        with torch.no_grad():
            noisy_samples = stochastic_fn.sample_from_noisy_inputs(
                x=model_input,
                smoothing_distribution=noise_dist,
                num_samples=args.num_noisy_samples,
                h=helper,
            )
        noisy_np = noisy_samples.detach().cpu().numpy()  # (nx, N, 2)

        for i in range(nx):
            cloud = noisy_np[i]  # (N, 2)
            diag = _cluster_diagnostics(
                output_cloud=cloud,
                cluster_strategy=args.cluster_strategy,
                behavior_anchor=target_init[i],
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                behavior_stay_radius=args.behavior_stay_radius,
                behavior_min_down=args.behavior_min_down,
                behavior_min_left=args.behavior_min_left,
                min_secondary_fraction=args.min_secondary_fraction,
                min_separation_ratio=args.min_separation_ratio,
            )
            diag["dataset_index"] = int(batch_indices[i])
            per_input_results.append(diag)

        elapsed = time.perf_counter() - batch_t0
        multimodal_in_batch = sum(r["is_multimodal"] for r in per_input_results[-nx:])
        print(
            f"batch {batch_id}/{num_batches}  "
            f"indices {int(batch_indices[0])}..{int(batch_indices[-1])}  "
            f"multimodal={multimodal_in_batch}/{nx}  "
            f"({elapsed:.1f}s)"
        )

    # -----------------------------------------------------------------------
    # Aggregate and classify
    # -----------------------------------------------------------------------
    multimodal_rows = [r for r in per_input_results if r["is_multimodal"]]
    multimodal_indices = [r["dataset_index"] for r in multimodal_rows]

    num_total = len(per_input_results)
    num_multimodal = len(multimodal_rows)
    num_k1 = sum(r["num_clusters"] <= 1 for r in per_input_results)
    num_k2 = sum(r["num_clusters"] == 2 for r in per_input_results)
    num_k3plus = sum(r["num_clusters"] >= 3 for r in per_input_results)

    sec_fracs = [r["secondary_fraction"] for r in per_input_results]
    sep_ratios = [r["separation_ratio"] for r in per_input_results if r["separation_ratio"] is not None]
    silhouettes = [r["silhouette_score"] for r in per_input_results if r["silhouette_score"] is not None]

    aggregate = {
        "num_scanned": num_total,
        "num_multimodal": num_multimodal,
        "fraction_multimodal": float(num_multimodal / num_total) if num_total > 0 else 0.0,
        "num_clusters_distribution": {
            "k<=1": num_k1,
            "k=2": num_k2,
            "k>=3": num_k3plus,
        },
        "mean_secondary_fraction": float(np.mean(sec_fracs)) if sec_fracs else None,
        "mean_separation_ratio": float(np.mean(sep_ratios)) if sep_ratios else None,
        "mean_silhouette_score": float(np.mean(silhouettes)) if silhouettes else None,
    }

    # -----------------------------------------------------------------------
    # Save figures
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    suffix = args.cluster_strategy
    scatter_path = output_dir / f"multimodal_scan_scatter_{suffix}.pdf"
    hist_path = output_dir / f"multimodal_scan_cluster_hist_{suffix}.pdf"
    _plot_multimodality_scatter(
        rows=per_input_results,
        min_secondary_fraction=args.min_secondary_fraction,
        min_separation_ratio=args.min_separation_ratio,
        output_path=scatter_path,
    )
    _plot_cluster_count_histogram(
        rows=per_input_results,
        cluster_strategy=args.cluster_strategy,
        output_path=hist_path,
    )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / f"multimodal_scan_{suffix}.json"
    payload: dict[str, Any] = {
        "config": {
            "num_inputs_scanned": num_to_scan,
            "num_noisy_samples": args.num_noisy_samples,
            "sigma": args.sigma,
            "input_mode": args.input_mode,
            "cluster_strategy": args.cluster_strategy,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
            "behavior_stay_radius": args.behavior_stay_radius,
            "behavior_min_down": args.behavior_min_down,
            "behavior_min_left": args.behavior_min_left,
            "min_secondary_fraction": args.min_secondary_fraction,
            "min_separation_ratio": args.min_separation_ratio,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "multimodal_indices": multimodal_indices,
        "per_input": per_input_results,
        "artifacts": {
            "scatter_plot": str(scatter_path),
            "cluster_hist": str(hist_path),
        },
    }
    _save_json(json_path, payload)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print()
    print("=== Multimodal Scan Summary ===")
    print(f"Strategy              : {args.cluster_strategy}")
    print(f"Inputs scanned        : {num_total}")
    print(f"k<=1                  : {num_k1}  ({100*num_k1/num_total:.1f}%)")
    print(f"k=2                   : {num_k2}  ({100*num_k2/num_total:.1f}%)")
    print(f"k>=3                  : {num_k3plus}  ({100*num_k3plus/num_total:.1f}%)")
    print(f"Truly multimodal      : {num_multimodal} / {num_total}  ({100*num_multimodal/num_total:.1f}%)")
    print(f"  (secondary_fraction >= {args.min_secondary_fraction}"
          f"  AND  separation_ratio >= {args.min_separation_ratio})")
    print()
    if multimodal_indices:
        print(f"Multimodal dataset indices ({len(multimodal_indices)}):")
        print(" ", multimodal_indices)
    else:
        print("No multimodal indices found with current thresholds.")
        print("Try lowering --min-secondary-fraction or --min-separation-ratio,")
        print("or raising --dbscan-eps / lowering --dbscan-min-samples.")
    print()
    print(f"Saved: {json_path}")
    print(f"Saved: {scatter_path}")
    print(f"Saved: {hist_path}")

    if multimodal_rows:
        print()
        print("Top-10 most multimodal inputs (by secondary_fraction):")
        top10 = sorted(multimodal_rows, key=lambda r: r["secondary_fraction"], reverse=True)[:10]
        print(f"  {'idx':>5}  {'k':>4}  {'sec_frac':>9}  {'sep_ratio':>10}  {'silhouette':>10}")
        for r in top10:
            sil = r["silhouette_score"]
            sep = r["separation_ratio"]
            print(
                f"  {r['dataset_index']:>5}  "
                f"{r['num_clusters']:>4}  "
                f"{r['secondary_fraction']:>9.3f}  "
                f"{sep if sep is not None else float('nan'):>10.3f}  "
                f"{sil if sil is not None else float('nan'):>10.3f}"
            )


if __name__ == "__main__":
    main()
