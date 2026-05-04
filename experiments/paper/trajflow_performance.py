from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RuntimeHeuristics,
    SmoothingParameters,
)
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from plotting import plot_trajflow_performance_from_json
from stochastic_functions import TrajFlowMeszarosStochasticFunction


def _parse_indices(text: str) -> np.ndarray:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("selected-indices must contain at least one integer")
    return np.array([int(v) for v in values], dtype=int)


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
    if outputs_gt.ndim != 4 or outputs_gt.shape[-1] != 2:
        raise ValueError(
            "Expected GT outputs with shape (num_samples, num_agents, num_timesteps, 2); "
            f"got {tuple(outputs_gt.shape)}"
        )
    if outputs_gt.shape[1] != 1:
        raise ValueError(
            "Expected GT outputs to contain exactly one target-agent trajectory; "
            f"got shape {tuple(outputs_gt.shape)}"
        )
    if outputs_gt.shape[2] < 1:
        raise ValueError(f"GT outputs have invalid shape: {tuple(outputs_gt.shape)}")

    return np.isfinite(outputs_gt[:, 0, -1, :]).all(axis=1)


def _prepare_inputs_and_targets(selected_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, outputs_gt = _load_inputs_and_outputs()
    return _prepare_inputs_and_targets_from_arrays(
        selected_indices=selected_indices,
        inputs=inputs,
        outputs_gt=outputs_gt,
    )


def _prepare_inputs_and_targets_from_arrays(
    selected_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = _finite_target_mask(outputs_gt)

    num_samples = inputs.shape[0]
    if outputs_gt.shape[0] != num_samples:
        raise ValueError(
            "Inputs and GT outputs have different sample counts: "
            f"Inputs={num_samples}, GT={outputs_gt.shape[0]}"
        )

    if np.any(selected_indices < 0) or np.any(selected_indices >= num_samples):
        raise ValueError(
            "selected-indices contains out-of-range values. "
            f"Allowed range is [0, {num_samples - 1}]"
        )
    invalid_selected = selected_indices[~valid_mask[selected_indices]]
    if invalid_selected.size > 0:
        raise ValueError(
            "selected-indices contains rows with non-finite final GT targets: "
            f"{invalid_selected.tolist()}"
        )

    # Keep only x,y channels for TrajFlow input and put target agent at index 0.
    x_input = inputs[selected_indices, :, :, :2]
    x_input = x_input[:, [1, 0]]
    x = torch.from_numpy(x_input.reshape(len(selected_indices), -1)).float()

    # GT target is final (x,y) point of the tracked/target agent.
    y_target = torch.from_numpy(outputs_gt[selected_indices, 0, -1, :]).float()
    if not torch.isfinite(y_target).all():
        raise ValueError("Encountered non-finite GT targets after selection")

    return x, y_target


def _choose_indices(
    valid_indices: np.ndarray,
    num_inputs: int,
    selected_indices_text: str | None,
) -> np.ndarray:
    if selected_indices_text is not None and selected_indices_text.strip():
        return _parse_indices(selected_indices_text)

    if num_inputs <= 0:
        return valid_indices.astype(int)
    if num_inputs > len(valid_indices):
        raise ValueError(
            f"num-inputs={num_inputs} exceeds available valid GT samples={len(valid_indices)}"
        )
    return valid_indices[:num_inputs].astype(int)


def _sample_original(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    model_input: torch.Tensor,
    helper: list[torch.Tensor],
    num_samples: int,
) -> torch.Tensor:
    x_rep = model_input.unsqueeze(1).expand(-1, num_samples, -1)
    if helper:
        helper_rep = [h_i.unsqueeze(1).expand(-1, num_samples, -1) for h_i in helper]
        return stochastic_fn(x_rep, h=helper_rep)
    return stochastic_fn(x_rep)


def _compute_distance_stats(samples: torch.Tensor, y_target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(
            "Expected samples with shape (nx, num_samples, 2), "
            f"got {tuple(samples.shape)}"
        )
    if y_target.ndim != 2 or y_target.shape[-1] != 2:
        raise ValueError(
            "Expected y_target with shape (nx, 2), "
            f"got {tuple(y_target.shape)}"
        )

    if not torch.isfinite(samples).all():
        raise ValueError("Encountered non-finite predictor samples")
    if not torch.isfinite(y_target).all():
        raise ValueError("Encountered non-finite GT targets")

    distances = torch.linalg.norm(samples - y_target.unsqueeze(1), dim=-1)
    if not torch.isfinite(distances).all():
        raise ValueError("Encountered non-finite distances")

    mean_distance = distances.mean(dim=1).detach().cpu().numpy()
    std_distance = distances.std(dim=1, unbiased=False).detach().cpu().numpy()
    return mean_distance, std_distance


def _extract_target_initial_position(x_positions: torch.Tensor, num_input_agents: int = 2) -> torch.Tensor:
    nx = x_positions.shape[0]
    num_input_timesteps = x_positions.shape[1] // (2 * num_input_agents)
    x_traj = x_positions.reshape(nx, num_input_agents, num_input_timesteps, 2)
    return x_traj[:, 0, 0, :]


def _build_parameters(args: argparse.Namespace, max_num_clusters: int) -> Parameters:
    return Parameters(
        smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=max_num_clusters,
            coverage=args.coverage,
            strategy=args.cluster_strategy,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            behavior_stay_radius=args.behavior_stay_radius,
            behavior_min_down=args.behavior_min_down,
            behavior_min_left=args.behavior_min_left,
        ),
        confidence=ConfidenceParameters(num_samples=args.confidence_num_samples, beta=1e-5),
        runtime=RuntimeHeuristics(noise_sub_batch_size=args.noise_sub_batch_size),
    )


def _compute_batch_metrics(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    batch_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    x, y_target = _prepare_inputs_and_targets_from_arrays(
        selected_indices=batch_indices,
        inputs=inputs,
        outputs_gt=outputs_gt,
    )
    cluster_anchor = _extract_target_initial_position(x)

    if args.input_mode == "position":
        model_input = x
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x)

    base_noise = MultivariateGaussian(
        mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
        cov=(args.sigma ** 2)
        * torch.eye(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
    )

    smoother_single = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_parameters(args, max_num_clusters=1),
        h=helper,
        cluster_anchor=cluster_anchor,
    )
    smoother_multi = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_parameters(args, max_num_clusters=2),
        h=helper,
        cluster_anchor=cluster_anchor,
    )

    samples_original = _sample_original(
        stochastic_fn=stochastic_fn,
        model_input=model_input,
        helper=helper,
        num_samples=args.metric_samples,
    )
    samples_noisy = stochastic_fn.sample_from_noisy_inputs(
        x=model_input,
        smoothing_distribution=base_noise,
        num_samples=args.metric_samples,
        h=helper,
    )
    samples_smoothed_k1 = smoother_single.monte_carlo(num_simulations=args.metric_samples)
    samples_smoothed_k2 = smoother_multi.monte_carlo(num_simulations=args.metric_samples)

    metrics: dict[str, dict[str, np.ndarray]] = {
        "original": {},
        "noisy_input": {},
        "smoothed_k1": {},
        "smoothed_k2": {},
    }
    metrics["original"]["mean"], metrics["original"]["std"] = _compute_distance_stats(samples_original, y_target)
    metrics["noisy_input"]["mean"], metrics["noisy_input"]["std"] = _compute_distance_stats(samples_noisy, y_target)
    metrics["smoothed_k1"]["mean"], metrics["smoothed_k1"]["std"] = _compute_distance_stats(samples_smoothed_k1, y_target)
    metrics["smoothed_k2"]["mean"], metrics["smoothed_k2"]["std"] = _compute_distance_stats(samples_smoothed_k2, y_target)
    return metrics


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _compute_tail_stats_table(
    metrics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, float]]]:
    quantiles = (90, 95)
    table: dict[str, dict[str, dict[str, float]]] = {}
    for metric_field, key_label in (("mean", "mean_distance"), ("std", "std_distance")):
        table[key_label] = {}
        for name, arrays in metrics.items():
            arr = arrays[metric_field]
            row: dict[str, float] = {"mean": float(arr.mean()), "max": float(arr.max())}
            for pct in quantiles:
                threshold = float(np.percentile(arr, pct))
                tail = arr[arr >= threshold]
                row[f"q{pct}"] = threshold
                row[f"cvar{pct}"] = float(tail.mean()) if len(tail) > 0 else threshold
            table[key_label][name] = row
    return table


def _ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please reply with 'y' or 'n'.")


def _ask_num_indices(prompt: str, available: int) -> int:
    while True:
        answer = input(prompt).strip()
        try:
            value = int(answer)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < 0:
            print("Please enter a non-negative integer.")
            continue
        if value > available:
            print(f"Please enter a value between 0 and {available}.")
            continue
        return value


def _count_indices_in_metrics_json(metrics_json_path: Path) -> int:
    with metrics_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("per_input", [])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TrajFlow predictors against GT final position using distance metrics."
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=0,
        help="How many valid inputs to evaluate when selected-indices is not provided. Use <=0 for all valid indices.",
    )
    parser.add_argument(
        "--selected-indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices. Overrides --num-inputs.",
    )
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=100,
        help="Number of stochastic samples per input used for distance metrics.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=5,
        help="How many selected inputs to process at once when computing metrics.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian input perturbation std used for noisy and smoothed predictors.",
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
        help="Sub-batch size along the noise/sample dimension inside stochastic sampling.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=500,
        help="num_samples for confidence parameter (used in smoothing object setup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for input selection and torch sampling.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis" / "performance"),
        help="Directory where raw metrics JSON is saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where selected-index plots are saved.",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=100,
        help="Top-k inputs with largest distance from the target initial position to GT final position. Use <=0 for all.",
    )
    parser.add_argument(
        "--plot-tail-quantile",
        type=float,
        default=0.90,
        help="Tail quantile used for the CVaR marker in selected plots.",
    )
    parser.add_argument(
        "--regenerate-data",
        type=str,
        choices=("y", "n"),
        default=None,
        help="Optional non-interactive override: y to regenerate, n to use existing JSON.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "performance_metrics.json"

    if args.regenerate_data is None:
        regenerate_data = _ask_yes_no(
            "Regenerate performance data now? [y/n] "
        )
    else:
        regenerate_data = args.regenerate_data == "y"

    if not regenerate_data:
        if not json_path.exists():
            print(f"Existing JSON not found at {json_path}. Falling back to regeneration.")
            regenerate_data = True
        else:
            available_json_indices = _count_indices_in_metrics_json(json_path)
            print(f"Available indices in existing JSON: {available_json_indices}")
            if args.regenerate_data is None:
                args.plot_top_k = _ask_num_indices(
                    "How many indices should be used from JSON? (0 for all) ",
                    available_json_indices,
                )
            print(f"Using existing JSON: {json_path}")
            plot_trajflow_performance_from_json(
                metrics_json=str(json_path),
                top_k=args.plot_top_k,
                tail_quantile=args.plot_tail_quantile,
                output_dir=args.figures_dir,
            )
            return

    if args.regenerate_data is None and not (args.selected_indices or "").strip():
        _, outputs_gt = _load_inputs_and_outputs()
        available_dataset_indices = int(np.sum(_finite_target_mask(outputs_gt)))
        print(f"Available valid indices in dataset: {available_dataset_indices}")
        args.num_inputs = _ask_num_indices(
            "How many indices should be regenerated? (0 for all) ",
            available_dataset_indices,
        )

    if args.metric_samples <= 0:
        raise ValueError("metric-samples must be positive")
    if args.input_batch_size <= 0:
        raise ValueError("input-batch-size must be positive")
    if args.noise_sub_batch_size <= 0:
        raise ValueError("noise-sub-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stochastic_fn = TrajFlowMeszarosStochasticFunction()

    all_inputs, outputs_gt = _load_inputs_and_outputs()
    valid_indices = np.flatnonzero(_finite_target_mask(outputs_gt))
    selected_indices = _choose_indices(
        valid_indices=valid_indices,
        num_inputs=args.num_inputs,
        selected_indices_text=args.selected_indices,
    )
    metric_names = ("original", "noisy_input", "smoothed_k1", "smoothed_k2")
    metrics_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"mean": [], "std": []}
        for name in metric_names
    }
    num_batches = (len(selected_indices) + args.input_batch_size - 1) // args.input_batch_size
    for batch_id, start in enumerate(range(0, len(selected_indices), args.input_batch_size), start=1):
        batch_t0 = time.perf_counter()
        batch_indices = selected_indices[start : start + args.input_batch_size]
        print(
            f"processing batch {batch_id}/{num_batches} "
            f"with dataset indices {int(batch_indices[0])}..{int(batch_indices[-1])}"
        )
        batch_metrics = _compute_batch_metrics(
            args=args,
            stochastic_fn=stochastic_fn,
            batch_indices=batch_indices,
            inputs=all_inputs,
            outputs_gt=outputs_gt,
        )
        print(f"batch {batch_id}/{num_batches} completed in {time.perf_counter() - batch_t0:.1f}s")
        for name in metric_names:
            metrics_parts[name]["mean"].append(batch_metrics[name]["mean"])
            metrics_parts[name]["std"].append(batch_metrics[name]["std"])

    metrics: dict[str, dict[str, np.ndarray]] = {
        name: {
            "mean": np.concatenate(metrics_parts[name]["mean"]),
            "std": np.concatenate(metrics_parts[name]["std"]),
        }
        for name in metric_names
    }

    per_input_rows = []
    for i, dataset_idx in enumerate(selected_indices.tolist()):
        per_input_rows.append(
            {
                "dataset_index": int(dataset_idx),
                "original": {
                    "mean_distance": float(metrics["original"]["mean"][i]),
                    "std_distance": float(metrics["original"]["std"][i]),
                },
                "noisy_input": {
                    "mean_distance": float(metrics["noisy_input"]["mean"][i]),
                    "std_distance": float(metrics["noisy_input"]["std"][i]),
                },
                "smoothed_k1": {
                    "mean_distance": float(metrics["smoothed_k1"]["mean"][i]),
                    "std_distance": float(metrics["smoothed_k1"]["std"][i]),
                },
                "smoothed_k2": {
                    "mean_distance": float(metrics["smoothed_k2"]["mean"][i]),
                    "std_distance": float(metrics["smoothed_k2"]["std"][i]),
                },
            }
        )

    aggregate = {
        key: {
            "mean_of_mean_distance": float(np.mean(metrics[key]["mean"])),
            "mean_of_std_distance": float(np.mean(metrics[key]["std"])),
        }
        for key in metrics
    }

    payload = {
        "config": {
            "selected_indices": selected_indices.tolist(),
            "metric_samples": args.metric_samples,
            "sigma": args.sigma,
            "smoothing_N": args.smoothing_N,
            "alpha": args.alpha,
            "coverage": args.coverage,
            "cluster_strategy": args.cluster_strategy,
            "input_mode": args.input_mode,
            "behavior_stay_radius": args.behavior_stay_radius,
            "behavior_min_down": args.behavior_min_down,
            "behavior_min_left": args.behavior_min_left,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
            "noise_sub_batch_size": args.noise_sub_batch_size,
            "confidence_num_samples": args.confidence_num_samples,
            "input_batch_size": args.input_batch_size,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "per_input": per_input_rows,
    }
    _save_json(json_path, payload)

    tail_stats_path = analysis_dir / "performance_tail_stats.json"
    _save_json(tail_stats_path, _compute_tail_stats_table(metrics))

    print("=== TrajFlow Performance GT Comparison ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"metric_samples: {args.metric_samples}")
    print(f"saved: {json_path}")
    print(f"saved: {tail_stats_path}")
    print("aggregate metrics:")
    for name, values in aggregate.items():
        print(
            f"  - {name}: "
            f"mean_of_mean_distance={values['mean_of_mean_distance']:.4f}, "
            f"mean_of_std_distance={values['mean_of_std_distance']:.4f}"
        )

    plot_trajflow_performance_from_json(
        metrics_json=str(json_path),
        top_k=args.plot_top_k,
        tail_quantile=args.plot_tail_quantile,
        output_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
