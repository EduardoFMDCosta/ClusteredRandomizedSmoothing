from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import ot
except ImportError as exc:  # pragma: no cover - import guard for optional runtime dependency
    raise ImportError(
        "POT is required for trajflow_wasserstein. Install with `pip install pot`."
    ) from exc


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from plotting import plot_trajflow_wasserstein_histograms
from stochastic_functions import TrajFlowMeszarosStochasticFunction
from trajflow_robustness import (
    _ask_num_indices,
    _choose_indices,
    _extract_target_initial_position,
    _finite_target_mask,
    _load_inputs_and_outputs,
    _load_or_create_fixed_random_order,
    _prepare_inputs_from_arrays,
)


def _compute_sinkhorn_wasserstein_2_distance(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    reg: float,
) -> float:
    """Compute Sinkhorn-approximated empirical 2-Wasserstein distance."""
    if x_samples.ndim != 2 or y_samples.ndim != 2:
        raise ValueError("Both sample sets must be 2-D arrays of shape (n, d).")
    if x_samples.shape[1] != y_samples.shape[1]:
        raise ValueError(
            "Both sample sets must have the same feature dimension; "
            f"got {x_samples.shape[1]} and {y_samples.shape[1]}"
        )
    if reg <= 0.0:
        raise ValueError("sinkhorn regularization (reg) must be positive")

    n_x = x_samples.shape[0]
    n_y = y_samples.shape[0]
    a = np.full(n_x, 1.0 / n_x, dtype=np.float64)
    b = np.full(n_y, 1.0 / n_y, dtype=np.float64)
    cost_matrix = ot.dist(x_samples.astype(np.float64), y_samples.astype(np.float64), metric="sqeuclidean")
    w2_squared = ot.sinkhorn2(a, b, cost_matrix, reg=reg, method="sinkhorn")
    return float(math.sqrt(max(w2_squared, 0.0)))


def _build_reference_parameters(args: argparse.Namespace, max_num_clusters: int) -> Parameters:
    """Match trajflow_experiment parameterization for k=1 and k=2 smoothers."""
    if max_num_clusters == 1:
        clustering = ClusteringParameters(
            max_num_clusters=1,
            coverage=args.coverage,
            strategy="dbscan",
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        )
    else:
        clustering = ClusteringParameters(
            max_num_clusters=max_num_clusters,
            coverage=args.coverage,
            strategy=args.cluster_strategy,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            behavior_stay_radius=args.behavior_stay_radius,
            behavior_min_down=args.behavior_min_down,
            behavior_min_left=args.behavior_min_left,
        )

    return Parameters(
        smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
        clustering=clustering,
        confidence=ConfidenceParameters(num_samples=args.confidence_num_samples, beta=1e-5),
    )


def _compute_batch_wasserstein_rows(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    batch_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
) -> list[dict[str, Any]]:
    x_position = _prepare_inputs_from_arrays(
        selected_indices=batch_indices,
        inputs=inputs,
        outputs_gt=outputs_gt,
    )
    cluster_anchor = _extract_target_initial_position(x_position)

    if args.input_mode == "position":
        model_input = x_position
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x_position)

    smoother_k1 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_reference_parameters(args, max_num_clusters=1),
        h=helper,
        cluster_anchor=cluster_anchor,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_reference_parameters(args, max_num_clusters=2),
        h=helper,
        cluster_anchor=cluster_anchor,
    )

    base_noise = MultivariateGaussian(
        mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
        cov=(args.sigma**2)
        * torch.eye(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
    )

    noisy_samples = stochastic_fn.sample_from_noisy_inputs(
        x=model_input,
        smoothing_distribution=base_noise,
        num_samples=args.distribution_samples,
        h=helper,
        noise_sub_batch_size=args.noise_sub_batch_size,
    )
    smoothed_k1_samples = smoother_k1.monte_carlo(num_simulations=args.distribution_samples)
    smoothed_k2_samples = smoother_k2.monte_carlo(num_simulations=args.distribution_samples)

    noisy_np = noisy_samples.detach().cpu().numpy()
    k1_np = smoothed_k1_samples.detach().cpu().numpy()
    k2_np = smoothed_k2_samples.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for i, dataset_idx in enumerate(batch_indices.tolist()):
        w2_noisy_k1 = _compute_sinkhorn_wasserstein_2_distance(
            noisy_np[i],
            k1_np[i],
            reg=args.sinkhorn_reg,
        )
        w2_noisy_k2 = _compute_sinkhorn_wasserstein_2_distance(
            noisy_np[i],
            k2_np[i],
            reg=args.sinkhorn_reg,
        )
        rows.append(
            {
                "dataset_index": int(dataset_idx),
                "w2_noisy_vs_k1": float(w2_noisy_k1),
                "w2_noisy_vs_k2": float(w2_noisy_k2),
            }
        )

    return rows


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical 2-Wasserstein distances between noisy TrajFlow outputs "
            "h_w(x+epsilon) and smoothed outputs from k=1 and k=2 smoothers."
        )
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=None,
        help=(
            "How many valid inputs to evaluate when selected-indices is not provided. "
            "If omitted, the script asks interactively. Use 0 for all valid indices."
        ),
    )
    parser.add_argument(
        "--selected-indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices. Overrides --num-inputs.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=5,
        help="How many selected inputs to process at once.",
    )
    parser.add_argument(
        "--distribution-samples",
        type=int,
        default=1000,
        help="Number of samples drawn from each distribution per input.",
    )
    parser.add_argument(
        "--sinkhorn-reg",
        type=float,
        default=0.1,
        help="Entropic regularization used by POT Sinkhorn approximation.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian input perturbation std used for noisy sampling and smoothing.",
    )
    parser.add_argument(
        "--smoothing-N",
        type=int,
        default=10,
        help="N for alpha-trimming smoothing.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Trimming parameter alpha.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.80,
        help="Coverage parameter for cluster regions.",
    )
    parser.add_argument(
        "--cluster-strategy",
        type=str,
        choices=("dbscan", "behavior_split"),
        default="behavior_split",
        help="Clustering strategy for k=2 smoothing.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=("control", "position"),
        default="control",
        help="Input representation for smoothing and noisy perturbation.",
    )
    parser.add_argument(
        "--behavior-stay-radius",
        type=float,
        default=1.0,
        help="Stay mode threshold: ||y - y_init|| <= radius.",
    )
    parser.add_argument(
        "--behavior-min-down",
        type=float,
        default=0.5,
        help="Turn mode threshold on axis x2: y2 - y2_init <= -min_down.",
    )
    parser.add_argument(
        "--behavior-min-left",
        type=float,
        default=0.0,
        help="Turn mode threshold on axis x1: y1 - y1_init <= -min_left.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        help="DBSCAN eps for clustering inside smoothing.",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=100,
        help="DBSCAN min_samples for clustering inside smoothing.",
    )
    parser.add_argument(
        "--noise-sub-batch-size",
        type=int,
        default=300,
        help="Sub-batch size along noise/sample dimension for stochastic sampling.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=1000,
        help="num_samples for confidence parameter used in smoothing setup.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis" / "wasserstein"),
        help="Directory where Wasserstein JSON is saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where Wasserstein histogram figure is saved.",
    )
    parser.add_argument(
        "--histogram-name",
        type=str,
        default="trajflow_wasserstein_histograms.pdf",
        help="Filename for the side-by-side histogram figure.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=24,
        help="Number of bins per histogram.",
    )
    parser.add_argument(
        "--fixed-index-order-file",
        type=str,
        default="",
        help=(
            "Optional path to the persistent random ordering JSON. "
            "If omitted, uses <experiments-data>/fixed_random_index_order.json."
        ),
    )
    args = parser.parse_args()

    if args.input_batch_size <= 0:
        raise ValueError("input-batch-size must be positive")
    if args.distribution_samples <= 0:
        raise ValueError("distribution-samples must be positive")
    if args.noise_sub_batch_size <= 0:
        raise ValueError("noise-sub-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if args.sinkhorn_reg <= 0.0:
        raise ValueError("sinkhorn-reg must be positive")
    if args.hist_bins <= 0:
        raise ValueError("hist-bins must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stochastic_fn = TrajFlowMeszarosStochasticFunction()

    all_inputs, outputs_gt = _load_inputs_and_outputs()
    valid_indices = np.flatnonzero(_finite_target_mask(outputs_gt))
    analysis_dir = Path(args.analysis_dir)
    fixed_order_path = (
        Path(args.fixed_index_order_file)
        if args.fixed_index_order_file
        else EXPERIMENTS / "data" / "fixed_random_index_order.json"
    )
    fixed_random_order = _load_or_create_fixed_random_order(
        path=fixed_order_path,
        valid_indices=valid_indices,
        seed=args.seed,
    )

    if args.num_inputs is None and not (args.selected_indices or "").strip():
        print(f"Available valid indices in dataset: {len(valid_indices)}")
        args.num_inputs = _ask_num_indices(
            "How many xs should be analyzed? (0 for all) ",
            len(valid_indices),
        )
    if args.num_inputs is None:
        args.num_inputs = 0

    selected_indices = _choose_indices(
        fixed_random_order=fixed_random_order,
        num_inputs=args.num_inputs,
        selected_indices_text=args.selected_indices,
    )

    all_rows: list[dict[str, Any]] = []
    num_batches = (len(selected_indices) + args.input_batch_size - 1) // args.input_batch_size
    for batch_id, start in enumerate(range(0, len(selected_indices), args.input_batch_size), start=1):
        t0 = time.perf_counter()
        batch_indices = selected_indices[start : start + args.input_batch_size]
        print(
            f"processing batch {batch_id}/{num_batches} "
            f"with {len(batch_indices)} dataset indices"
        )
        rows = _compute_batch_wasserstein_rows(
            args=args,
            stochastic_fn=stochastic_fn,
            batch_indices=batch_indices,
            inputs=all_inputs,
            outputs_gt=outputs_gt,
        )
        all_rows.extend(rows)
        print(f"batch {batch_id}/{num_batches} completed in {time.perf_counter() - t0:.1f}s")

    w_noisy_vs_k1 = np.array([float(row["w2_noisy_vs_k1"]) for row in all_rows], dtype=float)
    w_noisy_vs_k2 = np.array([float(row["w2_noisy_vs_k2"]) for row in all_rows], dtype=float)

    histogram_path = Path(args.figures_dir) / args.histogram_name
    plot_trajflow_wasserstein_histograms(
        w_noisy_vs_k1=w_noisy_vs_k1,
        w_noisy_vs_k2=w_noisy_vs_k2,
        output_path=histogram_path,
        bins=args.hist_bins,
    )

    summary = {
        "config": {
            "selected_indices": selected_indices.tolist(),
            "distribution_samples": int(args.distribution_samples),
            "ot_method": "sinkhorn",
            "sinkhorn_reg": float(args.sinkhorn_reg),
            "sigma": float(args.sigma),
            "smoothing_N": int(args.smoothing_N),
            "alpha": float(args.alpha),
            "coverage": float(args.coverage),
            "cluster_strategy": args.cluster_strategy,
            "input_mode": args.input_mode,
            "behavior_stay_radius": float(args.behavior_stay_radius),
            "behavior_min_down": float(args.behavior_min_down),
            "behavior_min_left": float(args.behavior_min_left),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "noise_sub_batch_size": int(args.noise_sub_batch_size),
            "confidence_num_samples": int(args.confidence_num_samples),
            "input_batch_size": int(args.input_batch_size),
            "seed": int(args.seed),
            "fixed_index_order_file": str(fixed_order_path),
            "hist_bins": int(args.hist_bins),
            "histogram_path": str(histogram_path),
        },
        "aggregate": {
            "mean_w2_noisy_vs_k1": float(np.mean(w_noisy_vs_k1)),
            "std_w2_noisy_vs_k1": float(np.std(w_noisy_vs_k1)),
            "mean_w2_noisy_vs_k2": float(np.mean(w_noisy_vs_k2)),
            "std_w2_noisy_vs_k2": float(np.std(w_noisy_vs_k2)),
        },
        "per_input": all_rows,
    }

    json_path = analysis_dir / "trajflow_wasserstein.json"
    _save_json(json_path, summary)

    print("=== TrajFlow Wasserstein Experiment ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"saved: {json_path}")
    print(f"saved: {histogram_path}")


if __name__ == "__main__":
    main()