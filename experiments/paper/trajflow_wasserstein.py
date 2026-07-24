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
from plotting import plot_trajflow_wasserstein_histograms, plot_wasserstein_per_timestep
from stochastic_functions import (
    TrajFlowMeszarosStochasticFunction,
    TrajFlowMeszarosFullTrajectoryStochasticFunction,
)
from trajflow_robustness import (
    _ask_yes_no,
    _ask_num_indices,
    _choose_indices,
    _extract_target_initial_position,
    _finite_target_mask,
    _load_json,
    _load_inputs_and_outputs,
    _load_or_create_fixed_random_order,
    _prepare_inputs_from_arrays,
)


def _compute_sinkhorn_wasserstein_2_distance(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    reg: float,
    ot_method: str,
) -> float:
    """Compute empirical 2-Wasserstein distance using a POT solver."""
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
    cost_matrix = ot.dist(
        x_samples.astype(np.float64, copy=False),
        y_samples.astype(np.float64, copy=False),
        metric="sqeuclidean",
    )

    if ot_method == "emd":
        w2_squared = float(ot.emd2(a, b, cost_matrix))
    else:
        # Use POT's log-domain Sinkhorn variant for improved numerical stability.
        w2_squared = float(
            ot.sinkhorn2(
                a,
                b,
                cost_matrix,
                reg=reg,
                method=ot_method,
                numItermax=5000,
                stopThr=1e-9,
            )
        )

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


def _clean_reference_samples(
    stochastic_fn: TrajFlowMeszarosStochasticFunction | TrajFlowMeszarosFullTrajectoryStochasticFunction,
    model_input: torch.Tensor,
    helper: list[torch.Tensor],
    num_samples: int,
    noise_sub_batch_size: int,
) -> torch.Tensor:
    """Draw samples from h_w(x) — the original model without added input noise.

    This provides a clean reference distribution for Wasserstein comparisons,
    allowing evaluation of how well each smoother preserves the true (unperturbed)
    predictive distribution.
    """
    nx = model_input.shape[0]
    # Expand input along the sample dimension; TrajFlow is stochastic by itself.
    results: list[torch.Tensor] = []
    for start in range(0, num_samples, noise_sub_batch_size):
        end = min(start + noise_sub_batch_size, num_samples)
        chunk = end - start
        x_chunk = model_input.unsqueeze(1).expand(-1, chunk, -1)
        if helper:
            h_chunk = [h_i.unsqueeze(1).expand(-1, chunk, -1) for h_i in helper]
            results.append(stochastic_fn(x_chunk, h=h_chunk))
        else:
            results.append(stochastic_fn(x_chunk))
    return torch.cat(results, dim=1)  # (nx, num_samples, output_dim)


def _compute_batch_wasserstein_rows(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction | TrajFlowMeszarosFullTrajectoryStochasticFunction,
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

    # For trajectory_dbscan we do not need an endpoint anchor.
    anchor_k2 = (
        None if args.cluster_strategy == "trajectory_dbscan"
        else cluster_anchor
    )

    smoother_k1 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_reference_parameters(args, max_num_clusters=1),
        h=helper,
        cluster_anchor=None,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_reference_parameters(args, max_num_clusters=2),
        h=helper,
        cluster_anchor=anchor_k2,
    )

    # Reference distribution: h_w(x) — clean forward pass, no input noise.
    # Changed from h_w(x+epsilon) to better reflect the original model's behavior.
    clean_samples = _clean_reference_samples(
        stochastic_fn=stochastic_fn,
        model_input=model_input,
        helper=helper,
        num_samples=args.distribution_samples,
        noise_sub_batch_size=args.noise_sub_batch_size,
    )
    smoothed_k1_samples = smoother_k1.monte_carlo(num_simulations=args.distribution_samples)
    smoothed_k2_samples = smoother_k2.monte_carlo(num_simulations=args.distribution_samples)

    clean_np = clean_samples.detach().cpu().numpy()      # (nx, n_samples, output_dim)
    k1_np = smoothed_k1_samples.detach().cpu().numpy()
    k2_np = smoothed_k2_samples.detach().cpu().numpy()

    output_dim = clean_np.shape[-1]
    # Detect trajectory mode: full output (num_timesteps * 2) vs endpoint (2).
    is_trajectory = output_dim > 2
    num_timesteps = output_dim // 2 if is_trajectory else 1

    rows: list[dict[str, Any]] = []
    for i, dataset_idx in enumerate(batch_indices.tolist()):
        # Full-space Wasserstein (endpoint distance for endpoint mode, full trajectory for traj mode).
        w2_clean_k1 = _compute_sinkhorn_wasserstein_2_distance(
            clean_np[i],
            k1_np[i],
            reg=args.sinkhorn_reg,
            ot_method=args.ot_method,
        )
        w2_clean_k2 = _compute_sinkhorn_wasserstein_2_distance(
            clean_np[i],
            k2_np[i],
            reg=args.sinkhorn_reg,
            ot_method=args.ot_method,
        )
        row: dict[str, Any] = {
            "dataset_index": int(dataset_idx),
            "w2_clean_vs_k1": float(w2_clean_k1),
            "w2_clean_vs_k2": float(w2_clean_k2),
        }

        # Per-timestep Wasserstein: computed in 2-D slice at each prediction step.
        if is_trajectory:
            w2_per_t_k1: list[float] = []
            w2_per_t_k2: list[float] = []
            for t in range(num_timesteps):
                sl = slice(2 * t, 2 * t + 2)
                w2_t_k1 = _compute_sinkhorn_wasserstein_2_distance(
                    clean_np[i, :, sl],
                    k1_np[i, :, sl],
                    reg=args.sinkhorn_reg,
                    ot_method=args.ot_method,
                )
                w2_t_k2 = _compute_sinkhorn_wasserstein_2_distance(
                    clean_np[i, :, sl],
                    k2_np[i, :, sl],
                    reg=args.sinkhorn_reg,
                    ot_method=args.ot_method,
                )
                w2_per_t_k1.append(float(w2_t_k1))
                w2_per_t_k2.append(float(w2_t_k2))
            row["w2_per_timestep_k1"] = w2_per_t_k1
            row["w2_per_timestep_k2"] = w2_per_t_k2

        rows.append(row)

    return rows


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical 2-Wasserstein distances between clean TrajFlow outputs "
            "h_w(x) and smoothed outputs from k=1 and k=2 smoothers."
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
        "--output-mode",
        type=str,
        choices=("endpoint", "trajectory"),
        default="endpoint",
        help=(
            "Whether to use 2-D endpoint outputs (endpoint) or full 40-D predicted "
            "trajectories (trajectory).  Trajectory mode also produces a per-timestep "
            "Wasserstein plot showing how W2 evolves over the prediction horizon."
        ),
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
        default=150,
        help="Number of samples drawn from each distribution per input.",
    )
    parser.add_argument(
        "--sinkhorn-reg",
        type=float,
        default=0.1,
        help="Entropic regularization used by POT Sinkhorn approximation.",
    )
    parser.add_argument(
        "--ot-method",
        type=str,
        choices=("sinkhorn_log", "sinkhorn_stabilized", "sinkhorn", "emd"),
        default="sinkhorn_log",
        help=(
            "POT OT solver. 'sinkhorn_log' is usually the most stable regularized option; "
            "'emd' computes exact OT (slower, no reg used)."
        ),
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
        choices=("dbscan", "behavior_split", "trajectory_dbscan"),
        default=None,
        help=(
            "Clustering strategy for k=2 smoothing. "
            "Defaults to 'behavior_split' for endpoint mode and 'trajectory_dbscan' for trajectory mode."
        ),
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
        default=None,
        help=(
            "DBSCAN eps for clustering inside smoothing. "
            "Defaults to 0.5 for endpoint mode and 8.0 for trajectory mode."
        ),
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=35,
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
        default=100,
        help="num_samples used to fit cluster boundaries. Clopper-Pearson bounds are not"
        " needed here, so 100 is sufficient.",
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
        "--timestep-plot-name",
        type=str,
        default="trajflow_wasserstein_per_timestep.pdf",
        help="Filename for the per-timestep Wasserstein figure (trajectory mode only).",
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
    parser.add_argument(
        "--regenerate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If set, recomputes data even when a saved JSON already exists. "
            "If --no-regenerate is set, loads from existing JSON and only plots. "
            "If omitted, the script asks interactively when JSON already exists."
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
    if args.ot_method != "emd" and args.sinkhorn_reg <= 0.0:
        raise ValueError("sinkhorn-reg must be positive")
    if args.hist_bins <= 0:
        raise ValueError("hist-bins must be positive")

    # Apply mode-dependent defaults.
    if args.cluster_strategy is None:
        args.cluster_strategy = "behavior_split" if args.output_mode == "endpoint" else "trajectory_dbscan"
    if args.dbscan_eps is None:
        args.dbscan_eps = 0.5 if args.output_mode == "endpoint" else 8.0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "trajflow_wasserstein.json"
    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif json_path.exists():
        do_regenerate = _ask_yes_no(
            "Found existing Wasserstein data JSON.\nRegenerate? [y/n]: "
        )
    else:
        do_regenerate = True

    if do_regenerate:
        if args.output_mode == "trajectory":
            stochastic_fn: TrajFlowMeszarosStochasticFunction | TrajFlowMeszarosFullTrajectoryStochasticFunction = (
                TrajFlowMeszarosFullTrajectoryStochasticFunction()
            )
        else:
            stochastic_fn = TrajFlowMeszarosStochasticFunction()
        all_inputs, outputs_gt = _load_inputs_and_outputs()
        valid_indices = np.flatnonzero(_finite_target_mask(outputs_gt))

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
    else:
        payload = _load_json(json_path)
        loaded_rows = payload.get("per_input", [])
        if not isinstance(loaded_rows, list):
            raise ValueError(f"Invalid per_input list in {json_path}")
        all_rows = loaded_rows
        selected_from_config = payload.get("config", {}).get("selected_indices", [])
        selected_indices = np.array(selected_from_config, dtype=int)
        fixed_order_path = (
            Path(args.fixed_index_order_file)
            if args.fixed_index_order_file
            else EXPERIMENTS / "data" / "fixed_random_index_order.json"
        )
        print(f"loaded: {json_path}")

    if len(all_rows) == 0:
        raise ValueError("No per-input Wasserstein rows found to plot.")

    # Support both old key names (w2_noisy_*) and new names (w2_clean_*).
    def _get_row_w2(row: dict[str, Any], key_new: str, key_old: str) -> float:
        if key_new in row:
            return float(row[key_new])
        if key_old in row:
            return float(row[key_old])
        raise KeyError(f"Row missing both '{key_new}' and '{key_old}' keys")

    w_clean_vs_k1 = np.array(
        [_get_row_w2(r, "w2_clean_vs_k1", "w2_noisy_vs_k1") for r in all_rows], dtype=float
    )
    w_clean_vs_k2 = np.array(
        [_get_row_w2(r, "w2_clean_vs_k2", "w2_noisy_vs_k2") for r in all_rows], dtype=float
    )

    histogram_path = Path(args.figures_dir) / args.histogram_name
    plot_trajflow_wasserstein_histograms(
        w_noisy_vs_k1=w_clean_vs_k1,
        w_noisy_vs_k2=w_clean_vs_k2,
        output_path=histogram_path,
        bins=args.hist_bins,
    )

    # Per-timestep Wasserstein plot (trajectory mode).
    if args.output_mode == "trajectory" or any("w2_per_timestep_k1" in r for r in all_rows):
        rows_with_ts = [r for r in all_rows if "w2_per_timestep_k1" in r and "w2_per_timestep_k2" in r]
        if rows_with_ts:
            w2_ts_k1 = np.array([r["w2_per_timestep_k1"] for r in rows_with_ts], dtype=float)
            w2_ts_k2 = np.array([r["w2_per_timestep_k2"] for r in rows_with_ts], dtype=float)
            timestep_path = Path(args.figures_dir) / args.timestep_plot_name
            plot_wasserstein_per_timestep(
                w2_k1=w2_ts_k1,
                w2_k2=w2_ts_k2,
                output_path=timestep_path,
            )
            print(f"saved: {timestep_path}")

    if do_regenerate:
        summary = {
            "config": {
                "selected_indices": selected_indices.tolist(),
                "output_mode": args.output_mode,
                "distribution_samples": int(args.distribution_samples),
                "ot_method": args.ot_method,
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
                "mean_w2_clean_vs_k1": float(np.mean(w_clean_vs_k1)),
                "std_w2_clean_vs_k1": float(np.std(w_clean_vs_k1)),
                "mean_w2_clean_vs_k2": float(np.mean(w_clean_vs_k2)),
                "std_w2_clean_vs_k2": float(np.std(w_clean_vs_k2)),
            },
            "per_input": all_rows,
        }

        _save_json(json_path, summary)

    print("=== TrajFlow Wasserstein Experiment ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    if do_regenerate:
        print(f"saved: {json_path}")
    else:
        print(f"loaded: {json_path}")
    print(f"saved: {histogram_path}")


if __name__ == "__main__":
    main()