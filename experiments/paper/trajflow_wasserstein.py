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
    RuntimeHeuristics,
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
from experiment_configs import WASSERSTEIN as cfg


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


def _build_reference_parameters(max_num_clusters: int) -> Parameters:
    """Build Parameters for k=1 or k=2 smoother using the WASSERSTEIN config."""
    if max_num_clusters == 1:
        clustering = ClusteringParameters(
            max_num_clusters=1,
            coverage=cfg.coverage,
            strategy="dbscan",
            dbscan_eps=cfg.dbscan_eps,
            dbscan_min_samples=cfg.dbscan_min_samples,
        )
    else:
        clustering = ClusteringParameters(
            max_num_clusters=max_num_clusters,
            coverage=cfg.coverage,
            strategy=cfg.cluster_strategy,
            dbscan_eps=cfg.dbscan_eps,
            dbscan_min_samples=cfg.dbscan_min_samples,
            behavior_stay_radius=cfg.behavior_stay_radius,
            behavior_min_down=cfg.behavior_min_down,
            behavior_min_left=cfg.behavior_min_left,
        )
    return Parameters(
        smoothing=SmoothingParameters(N=cfg.smoothing_N, alpha=cfg.alpha),
        clustering=clustering,
        confidence=ConfidenceParameters(num_samples=cfg.confidence_num_samples, beta=1e-5),
        runtime=RuntimeHeuristics(noise_sub_batch_size=cfg.noise_sub_batch_size),
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

    if cfg.input_mode == "position":
        model_input = x_position
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x_position)

    # For endpoint_dbscan we do not need an endpoint anchor (data-driven from
    # model outputs). behavior_split requires the initial position of the target agent.
    anchor_k2 = (
        None if cfg.cluster_strategy == "endpoint_dbscan"
        else cluster_anchor
    )

    smoother_k1 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=_build_reference_parameters(max_num_clusters=1),
        h=helper,
        cluster_anchor=None,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=_build_reference_parameters(max_num_clusters=2),
        h=helper,
        cluster_anchor=anchor_k2,
    )

    # Reference distribution: h_w(x) — clean forward pass, no input noise.
    clean_samples = _clean_reference_samples(
        stochastic_fn=stochastic_fn,
        model_input=model_input,
        helper=helper,
        num_samples=cfg.distribution_samples,
        noise_sub_batch_size=cfg.noise_sub_batch_size,
    )
    smoothed_k1_samples = smoother_k1.monte_carlo(num_simulations=cfg.distribution_samples)
    smoothed_k2_samples = smoother_k2.monte_carlo(num_simulations=cfg.distribution_samples)

    clean_np = clean_samples.detach().cpu().numpy()      # (nx, n_samples, output_dim)
    k1_np = smoothed_k1_samples.detach().cpu().numpy()
    k2_np = smoothed_k2_samples.detach().cpu().numpy()

    # Record how many clusters were actually found for each input so callers
    # can filter to inputs where k=2 was genuinely observed.
    k2_k_per_row: list[int] = smoother_k2.cluster.k_per_row.tolist()

    output_dim = clean_np.shape[-1]
    # Detect trajectory mode: full output (num_timesteps * 2) vs endpoint (2).
    is_trajectory = output_dim > 2
    num_timesteps = output_dim // 2 if is_trajectory else 1

    rows: list[dict[str, Any]] = []
    for i, dataset_idx in enumerate(batch_indices.tolist()):
        row: dict[str, Any] = {
            "dataset_index": int(dataset_idx),
            "k_per_row_k2": int(k2_k_per_row[i]),
        }

        if is_trajectory:
            # All metrics are 2-D, evaluated only at the selected timesteps so
            # that the computation stays tractable while still capturing the
            # temporal trend.  The main histogram key reuses the last selected
            # timestep (= final position), which is the same quantity as
            # endpoint mode and avoids inflating W2 over 40-D trajectory space.
            eval_indices = [
                t for t in cfg.timestep_eval_indices if t < num_timesteps
            ]
            w2_per_t_k1: list[float] = []
            w2_per_t_k2: list[float] = []
            for t in eval_indices:
                sl = slice(2 * t, 2 * t + 2)
                w2_t_k1 = _compute_sinkhorn_wasserstein_2_distance(
                    clean_np[i, :, sl],
                    k1_np[i, :, sl],
                    reg=cfg.sinkhorn_reg,
                    ot_method=cfg.ot_method,
                )
                w2_t_k2 = _compute_sinkhorn_wasserstein_2_distance(
                    clean_np[i, :, sl],
                    k2_np[i, :, sl],
                    reg=cfg.sinkhorn_reg,
                    ot_method=cfg.ot_method,
                )
                w2_per_t_k1.append(float(w2_t_k1))
                w2_per_t_k2.append(float(w2_t_k2))
            row["w2_per_timestep_k1"] = w2_per_t_k1
            row["w2_per_timestep_k2"] = w2_per_t_k2
            row["w2_timestep_indices"] = eval_indices
            # Histogram key = last evaluated (= final) timestep 2-D W2.
            row["w2_clean_vs_k1"] = w2_per_t_k1[-1]
            row["w2_clean_vs_k2"] = w2_per_t_k2[-1]
        else:
            # Endpoint mode: the full output is already 2-D.
            w2_clean_k1 = _compute_sinkhorn_wasserstein_2_distance(
                clean_np[i],
                k1_np[i],
                reg=cfg.sinkhorn_reg,
                ot_method=cfg.ot_method,
            )
            w2_clean_k2 = _compute_sinkhorn_wasserstein_2_distance(
                clean_np[i],
                k2_np[i],
                reg=cfg.sinkhorn_reg,
                ot_method=cfg.ot_method,
            )
            row["w2_clean_vs_k1"] = float(w2_clean_k1)
            row["w2_clean_vs_k2"] = float(w2_clean_k2)

        rows.append(row)

    return rows


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """All smoothing/model parameters are in experiment_configs.py (WASSERSTEIN)."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute empirical 2-Wasserstein distances between clean TrajFlow outputs\n"
            "h_w(x) and smoothed outputs from k=1 and k=2 smoothers.\n"
            "Edit experiment_configs.WASSERSTEIN to change model/smoothing parameters."
        )
    )
    parser.add_argument("--num-inputs", type=int, default=0,
                        help="How many valid inputs to evaluate (0 = all). Ignored if --selected-indices is set.")
    parser.add_argument("--selected-indices", type=str, default="",
                        help="Comma-separated dataset indices. Overrides --num-inputs.")
    parser.add_argument("--regenerate", action=argparse.BooleanOptionalAction, default=None,
                        help="Re-run even when a saved JSON already exists.")
    parser.add_argument("--analysis-dir", default=str(EXPERIMENTS / "data" / "analysis" / "wasserstein"))
    parser.add_argument("--figures-dir", default=str(EXPERIMENTS / "data" / "figures"))
    parser.add_argument("--histogram-name", default="trajflow_wasserstein_histograms.pdf")
    parser.add_argument("--timestep-plot-name", default="trajflow_wasserstein_per_timestep.pdf")
    parser.add_argument("--fixed-index-order-file", default="")
    args = parser.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "trajflow_wasserstein.json"
    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif json_path.exists():
        do_regenerate = _ask_yes_no("Found existing Wasserstein data JSON.\nRegenerate? [y/n]: ")
    else:
        do_regenerate = True

    if do_regenerate:
        if cfg.output_mode == "trajectory":
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
            path=fixed_order_path, valid_indices=valid_indices, seed=cfg.seed,
        )

        num_inputs = args.num_inputs
        if num_inputs == 0 and not (args.selected_indices or "").strip():
            print(f"Available valid indices in dataset: {len(valid_indices)}")
            num_inputs = _ask_num_indices(
                "How many xs should be analyzed? (0 for all) ", len(valid_indices)
            )

        selected_indices = _choose_indices(
            fixed_random_order=fixed_random_order,
            num_inputs=num_inputs,
            selected_indices_text=args.selected_indices,
        )

        all_rows: list[dict[str, Any]] = []
        num_batches = (len(selected_indices) + cfg.input_batch_size - 1) // cfg.input_batch_size
        for batch_id, start in enumerate(range(0, len(selected_indices), cfg.input_batch_size), start=1):
            t0 = time.perf_counter()
            batch_indices = selected_indices[start : start + cfg.input_batch_size]
            print(f"processing batch {batch_id}/{num_batches} with {len(batch_indices)} dataset indices")
            rows = _compute_batch_wasserstein_rows(
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
        selected_indices = np.array(payload.get("config", {}).get("selected_indices", []), dtype=int)
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

    # Focus on inputs where the k=2 smoother genuinely found 2 clusters.
    # Rows without the key (e.g. loaded from older JSON) are kept by default.
    rows_for_plot = [r for r in all_rows if r.get("k_per_row_k2", 2) >= 2]
    if not rows_for_plot:
        rows_for_plot = all_rows  # safety fallback
    n_filtered = len(all_rows) - len(rows_for_plot)
    if n_filtered > 0:
        print(f"  ({n_filtered} input(s) with <2 clusters excluded from plots; {len(rows_for_plot)} remaining)")

    w_clean_vs_k1 = np.array(
        [_get_row_w2(r, "w2_clean_vs_k1", "w2_noisy_vs_k1") for r in rows_for_plot], dtype=float
    )
    w_clean_vs_k2 = np.array(
        [_get_row_w2(r, "w2_clean_vs_k2", "w2_noisy_vs_k2") for r in rows_for_plot], dtype=float
    )

    histogram_path = Path(args.figures_dir) / args.histogram_name
    plot_trajflow_wasserstein_histograms(
        w_noisy_vs_k1=w_clean_vs_k1,
        w_noisy_vs_k2=w_clean_vs_k2,
        output_path=histogram_path,
        bins=cfg.hist_bins,
    )

    # Per-timestep Wasserstein plot (trajectory mode only).
    if cfg.output_mode == "trajectory" or any("w2_per_timestep_k1" in r for r in rows_for_plot):
        rows_with_ts = [r for r in rows_for_plot if "w2_per_timestep_k1" in r and "w2_per_timestep_k2" in r]
        if rows_with_ts:
            w2_ts_k1 = np.array([r["w2_per_timestep_k1"] for r in rows_with_ts], dtype=float)
            w2_ts_k2 = np.array([r["w2_per_timestep_k2"] for r in rows_with_ts], dtype=float)
            ts_indices = np.array(rows_with_ts[0].get(
                "w2_timestep_indices", list(range(w2_ts_k1.shape[1]))
            ), dtype=int)
            timestep_path = Path(args.figures_dir) / args.timestep_plot_name
            plot_wasserstein_per_timestep(
                w2_k1=w2_ts_k1, w2_k2=w2_ts_k2,
                output_path=timestep_path, timesteps=ts_indices,
            )
            print(f"saved: {timestep_path}")

    if do_regenerate:
        summary = {
            "config": {
                "selected_indices": selected_indices.tolist(),
                "output_mode": cfg.output_mode,
                "distribution_samples": cfg.distribution_samples,
                "ot_method": cfg.ot_method,
                "sinkhorn_reg": cfg.sinkhorn_reg,
                "sigma": cfg.sigma,
                "smoothing_N": cfg.smoothing_N,
                "alpha": cfg.alpha,
                "coverage": cfg.coverage,
                "cluster_strategy": cfg.cluster_strategy,
                "behavior_stay_radius": cfg.behavior_stay_radius,
                "confidence_num_samples": cfg.confidence_num_samples,
                "input_batch_size": cfg.input_batch_size,
                "seed": cfg.seed,
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