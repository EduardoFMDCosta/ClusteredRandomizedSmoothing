from __future__ import annotations

import argparse
import json
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

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RuntimeHeuristics,
    SmoothingParameters,
)
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
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

    y_target = torch.from_numpy(outputs_gt[selected_indices, 0, -1, :]).float()
    if not torch.isfinite(y_target).all():
        raise ValueError("Encountered non-finite GT targets after selection")

    return x, y_target


def _choose_indices(
    valid_indices: np.ndarray,
    num_inputs: int,
    selected_indices_text: str | None,
    seed: int,
) -> np.ndarray:
    if selected_indices_text is not None and selected_indices_text.strip():
        return _parse_indices(selected_indices_text)

    if num_inputs <= 0:
        raise ValueError("num-inputs must be positive")
    if num_inputs > len(valid_indices):
        raise ValueError(
            f"num-inputs={num_inputs} exceeds available valid GT samples={len(valid_indices)}"
        )

    rng = np.random.default_rng(seed)
    return rng.choice(valid_indices, size=num_inputs, replace=False).astype(int)


def _project_l2_ball(delta: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0.0:
        return torch.zeros_like(delta)
    norms = torch.linalg.norm(delta, dim=1, keepdim=True)
    scales = torch.clamp(radius / torch.clamp(norms, min=1e-12), max=1.0)
    return delta * scales


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


def _convert_position_input(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_position: torch.Tensor,
    input_mode: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if input_mode == "position":
        return x_position, []
    model_input, helper = stochastic_fn.pos_to_ctrl(x_position)
    return model_input, helper


def _estimate_original_objective(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_position: torch.Tensor,
    y_target: torch.Tensor,
    input_mode: str,
    num_samples: int,
) -> torch.Tensor:
    model_input, helper = _convert_position_input(
        stochastic_fn=stochastic_fn,
        x_position=x_position,
        input_mode=input_mode,
    )
    samples_original = _sample_original(
        stochastic_fn=stochastic_fn,
        model_input=model_input,
        helper=helper,
        num_samples=num_samples,
    )
    distances = torch.linalg.norm(samples_original - y_target.unsqueeze(1), dim=-1)
    return distances.mean(dim=1)


def _compute_adversarial_delta_spsa(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_clean: torch.Tensor,
    y_target: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    radius = args.attack_radius
    if radius <= 0.0:
        zero = torch.zeros_like(x_clean)
        clean_obj = _estimate_original_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
        )
        clean_np = clean_obj.detach().cpu().numpy()
        return zero, clean_np, clean_np.copy()

    delta = torch.zeros_like(x_clean)

    clean_obj = _estimate_original_objective(
        stochastic_fn=stochastic_fn,
        x_position=x_clean,
        y_target=y_target,
        input_mode=args.input_mode,
        num_samples=args.attack_eval_samples,
    )
    best_obj = clean_obj.clone()
    best_delta = delta.clone()

    for step in range(1, args.attack_steps + 1):
        direction = torch.randn_like(delta)
        direction = direction / torch.clamp(
            torch.linalg.norm(direction, dim=1, keepdim=True),
            min=1e-12,
        )

        delta_plus = _project_l2_ball(delta + args.attack_probe * direction, radius)
        delta_minus = _project_l2_ball(delta - args.attack_probe * direction, radius)

        obj_plus = _estimate_original_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta_plus,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
        )
        obj_minus = _estimate_original_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta_minus,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
        )

        scale = ((obj_plus - obj_minus) / (2.0 * args.attack_probe)).unsqueeze(1)
        grad_est = scale * direction

        delta = _project_l2_ball(delta + args.attack_step_size * grad_est, radius)

        obj_current = _estimate_original_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
        )

        improved = obj_current > best_obj
        if improved.any():
            improved_mask = improved.unsqueeze(1)
            best_obj = torch.where(improved, obj_current, best_obj)
            best_delta = torch.where(improved_mask, delta, best_delta)

        if args.attack_log_every > 0 and (
            step % args.attack_log_every == 0 or step == args.attack_steps
        ):
            mean_gain = float((best_obj - clean_obj).mean().item())
            print(
                f"  attack step {step}/{args.attack_steps}: "
                f"mean objective gain={mean_gain:.4f}"
            )

    clean_np = clean_obj.detach().cpu().numpy()
    best_np = best_obj.detach().cpu().numpy()
    return best_delta, clean_np, best_np


def _compute_batch_metrics(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_adv_position: torch.Tensor,
    y_target: torch.Tensor,
) -> dict[str, dict[str, np.ndarray]]:
    cluster_anchor = _extract_target_initial_position(x_adv_position)
    model_input, helper = _convert_position_input(
        stochastic_fn=stochastic_fn,
        x_position=x_adv_position,
        input_mode=args.input_mode,
    )

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
    samples_smoothed_k3 = smoother_multi.monte_carlo(num_simulations=args.metric_samples)

    metrics: dict[str, dict[str, np.ndarray]] = {
        "original": {},
        "noisy_input": {},
        "smoothed_k1": {},
        "smoothed_k3": {},
    }
    metrics["original"]["mean"], metrics["original"]["std"] = _compute_distance_stats(samples_original, y_target)
    metrics["noisy_input"]["mean"], metrics["noisy_input"]["std"] = _compute_distance_stats(samples_noisy, y_target)
    metrics["smoothed_k1"]["mean"], metrics["smoothed_k1"]["std"] = _compute_distance_stats(samples_smoothed_k1, y_target)
    metrics["smoothed_k3"]["mean"], metrics["smoothed_k3"]["std"] = _compute_distance_stats(samples_smoothed_k3, y_target)
    return metrics


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


def _plot_metric_histogram(
    values: dict[str, np.ndarray],
    xlabel: str,
    output_path: Path,
    tail_quantile: float,
) -> None:
    _configure_plot_style()

    colors = {
        "original": "#222222",
        "noisy_input": "#E4572E",
        "smoothed_k1": "#4C78A8",
        "smoothed_k3": "#2A9D8F",
    }
    mean_color = "#1B1B1B"
    tail_color = "#C73E1D"
    titles = {
        "original": r"$h_w(x + \delta)$",
        "noisy_input": r"$h_w(x + \delta + \varepsilon)$",
        "smoothed_k1": r"$\alpha$-smoothing",
        "smoothed_k3": r"Clustered $\alpha$-smoothing",
    }
    ordered_keys = ("original", "noisy_input", "smoothed_k1", "smoothed_k3")

    clean_values = {
        key: values[key]
        for key in ordered_keys
    }
    all_values = np.concatenate([clean_values[key] for key in ordered_keys])
    tail_threshold = float(np.quantile(all_values, tail_quantile))
    tail_probability = {
        key: float(np.mean(clean_values[key] > tail_threshold))
        for key in ordered_keys
    }
    min_v = float(np.min(all_values))
    max_v = float(np.max(all_values))
    if np.isclose(min_v, max_v):
        bins = np.linspace(min_v - 0.5, max_v + 0.5, 15)
    else:
        bins = np.linspace(min_v, max_v, 15)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.11, top=0.95, hspace=0.28, wspace=0.16)

    for panel_index, (ax, key) in enumerate(zip(axes.flat, ordered_keys)):
        data = clean_values[key]
        weights = np.full(data.shape, 1.0 / len(data), dtype=float)
        hist = ax.hist(
            data,
            bins=bins,
            weights=weights,
            density=False,
            alpha=0.35,
            linewidth=1.1,
            edgecolor=colors[key],
            histtype="stepfilled",
            color=colors[key],
        )
        mean_line = ax.axvline(
            float(np.mean(data)),
            color=mean_color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.95,
            label="Mean",
        )
        tail_line = ax.axvline(
            tail_threshold,
            color=tail_color,
            linestyle="-.",
            linewidth=1.1,
            alpha=0.95,
            label=f"Quantile at {tail_quantile:.0%}",
        )
        ax.set_title(titles[key])
        if panel_index >= 2:
            ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle=":", alpha=0.35)

        if panel_index == 0:
            legend_handles = [mean_line, tail_line]
            legend_labels = ["Mean", f"Quantile at {tail_quantile:.0%}"]
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                fontsize=8,
                frameon=True,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial TrajFlow experiment: optimize bounded perturbations that maximize "
            "original predictor error, then compare original/noisy/smoothed predictors on x+delta."
        )
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=300,
        help="How many valid inputs to sample randomly when selected-indices is not provided.",
    )
    parser.add_argument(
        "--selected-indices",
        type=str,
        default="2,6,7,8,14,21,23,35,41,42,43,46,48,51,56,57,59,61,62,64,67,70,77,78,81,82,83,86,89,91,95,107,114,115,119,121,122,124,126,127,131,133,134,135,137,140,144,152,153,156,157,158,159,160,163,165,166,167,168,169,173,174,178,182,183,185,188,192,195,197,198,203,204,205,207,209,210,213,221,222,223,226,227,228,234,236,237,238,240,244,246,247,248,249,250,251,252,253,254,255,257,258,260,261,262,263,264,267,268,269,271,272,273,274,275,276,277,278,279,281,282,283,285,290,295,296,299,309,318,327,329,331,332,334,335,338,340,343,344,348,349,351,357,358,359,361,363,365,373,374,378,384,386,388,389,390,391,393,399,405,406,407,413,414,415,423,428,431,432,434,436,438,439,441,442,443,453,456,458,460,462,464,466,468,471,472,481,482,483,484,487,489,492,494,495,496,502,503,507,510,512,513,514,516,520,525,526,527,528,529,530,531,533,534,535,538,540,541,543,546,547,548,549,551,554,557,563,564,565,568,569,570,574,576,579,583,585,588,591,592,594,596,597,599,600,601,602,603,605,607,610,612,615,616,617,618,620,621,626,627,628,629,630,631,633,634,639,644,645,646,647,648,656,661,662,663,664,665,666,670,671,672,675,676,679,680,681,682,684,685,687,688,689,692,693,694,697,700,702,703,704,710,714,715,722,724,726,732,733,736,746,750,751,752,753,756,757,759,760,764,767,769,770,771,772,773,776,777,778,779,780,781,783,784,785,786,787,788,790,791,793,794,798,799,801,802,804,807,808,811,812,813,815,816,817,818,820,821,822,824,825,826,827,830,834,835,836,837,839,841,844,848,850,851,852,856,858,859,860,861,862,863,864,867,870,872,873,874,875,879,883,884",
        help="Optional comma-separated dataset indices. Overrides random --num-inputs sampling.",
    )
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=500,
        help="Number of stochastic samples per input used for final distance metrics.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=5,
        help="How many selected inputs to process at once for adversarial search and metrics.",
    )
    parser.add_argument(
        "--attack-radius",
        type=float,
        default=1.0,
        help="L2 radius r for adversarial perturbation delta in input x-space.",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=12,
        help="Number of SPSA ascent steps used to optimize delta.",
    )
    parser.add_argument(
        "--attack-probe",
        type=float,
        default=0.05,
        help="SPSA probe scale used for finite-difference objective probing.",
    )
    parser.add_argument(
        "--attack-step-size",
        type=float,
        default=0.25,
        help="Step size for projected SPSA ascent updates.",
    )
    parser.add_argument(
        "--attack-eval-samples",
        type=int,
        default=64,
        help="Number of stochastic samples for each adversarial objective estimate.",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.90,
        help="Quantile defining common tail threshold tau for histogram tail probability P(X>tau).",
    )
    parser.add_argument(
        "--attack-log-every",
        type=int,
        default=4,
        help="Print attack progress every N steps (0 disables).",
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
        help="Clustering strategy for k=3 smoothing.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=("control", "position"),
        default="control",
        help="Input representation used by predictor/smoother internals.",
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
        default=500,
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
        default=1000,
        help="num_samples for confidence parameter (used in smoothing setup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for index selection and stochastic search.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where plots are saved.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis"),
        help="Directory where raw metrics JSON is saved.",
    )
    args = parser.parse_args()

    if args.metric_samples <= 0:
        raise ValueError("metric-samples must be positive")
    if args.input_batch_size <= 0:
        raise ValueError("input-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if args.attack_radius < 0.0:
        raise ValueError("attack-radius must be non-negative")
    if args.attack_steps <= 0:
        raise ValueError("attack-steps must be positive")
    if args.attack_probe <= 0.0:
        raise ValueError("attack-probe must be positive")
    if args.attack_step_size <= 0.0:
        raise ValueError("attack-step-size must be positive")
    if args.attack_eval_samples <= 0:
        raise ValueError("attack-eval-samples must be positive")
    if not (0.0 < args.tail_quantile < 1.0):
        raise ValueError("tail-quantile must be in (0, 1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stochastic_fn = TrajFlowMeszarosStochasticFunction()

    all_inputs, outputs_gt = _load_inputs_and_outputs()
    valid_indices = np.flatnonzero(_finite_target_mask(outputs_gt))
    selected_indices = _choose_indices(
        valid_indices=valid_indices,
        num_inputs=args.num_inputs,
        selected_indices_text=args.selected_indices,
        seed=args.seed,
    )

    metric_names = ("original", "noisy_input", "smoothed_k1", "smoothed_k3")
    metrics_parts: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"mean": [], "std": []}
        for name in metric_names
    }

    attack_rows: list[dict[str, Any]] = []

    num_batches = (len(selected_indices) + args.input_batch_size - 1) // args.input_batch_size
    for batch_id, start in enumerate(range(0, len(selected_indices), args.input_batch_size), start=1):
        batch_t0 = time.perf_counter()
        batch_indices = selected_indices[start : start + args.input_batch_size]
        print(
            f"processing batch {batch_id}/{num_batches} "
            f"with dataset indices {int(batch_indices[0])}..{int(batch_indices[-1])}"
        )

        x_clean, y_target = _prepare_inputs_and_targets_from_arrays(
            selected_indices=batch_indices,
            inputs=all_inputs,
            outputs_gt=outputs_gt,
        )

        attack_t0 = time.perf_counter()
        delta_adv, clean_obj, adv_obj = _compute_adversarial_delta_spsa(
            args=args,
            stochastic_fn=stochastic_fn,
            x_clean=x_clean,
            y_target=y_target,
        )
        x_adv = x_clean + delta_adv
        delta_norms = torch.linalg.norm(delta_adv, dim=1).detach().cpu().numpy()
        print(f"  attack finished in {time.perf_counter() - attack_t0:.1f}s")

        metrics_t0 = time.perf_counter()
        batch_metrics = _compute_batch_metrics(
            args=args,
            stochastic_fn=stochastic_fn,
            x_adv_position=x_adv,
            y_target=y_target,
        )
        print(f"  metrics finished in {time.perf_counter() - metrics_t0:.1f}s")

        for name in metric_names:
            metrics_parts[name]["mean"].append(batch_metrics[name]["mean"])
            metrics_parts[name]["std"].append(batch_metrics[name]["std"])

        for i, dataset_idx in enumerate(batch_indices.tolist()):
            attack_rows.append(
                {
                    "dataset_index": int(dataset_idx),
                    "clean_objective": float(clean_obj[i]),
                    "adversarial_objective": float(adv_obj[i]),
                    "objective_gain": float(adv_obj[i] - clean_obj[i]),
                    "delta_l2_norm": float(delta_norms[i]),
                    "delta_flat": delta_adv[i].detach().cpu().tolist(),
                }
            )

        print(f"batch {batch_id}/{num_batches} completed in {time.perf_counter() - batch_t0:.1f}s")

    metrics: dict[str, dict[str, np.ndarray]] = {
        name: {
            "mean": np.concatenate(metrics_parts[name]["mean"]),
            "std": np.concatenate(metrics_parts[name]["std"]),
        }
        for name in metric_names
    }

    output_dir = Path(args.output_dir)
    mean_plot = output_dir / "trajflow_adversarial_average_distance.pdf"
    std_plot = output_dir / "trajflow_adversarial_std_distance.pdf"

    _plot_metric_histogram(
        values={k: metrics[k]["mean"] for k in metrics},
        xlabel="Average error",
        output_path=mean_plot,
        tail_quantile=args.tail_quantile,
    )
    _plot_metric_histogram(
        values={k: metrics[k]["std"] for k in metrics},
        xlabel="Average error std deviation",
        output_path=std_plot,
        tail_quantile=args.tail_quantile,
    )

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "adversarial_metrics.json"

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
                "smoothed_k3": {
                    "mean_distance": float(metrics["smoothed_k3"]["mean"][i]),
                    "std_distance": float(metrics["smoothed_k3"]["std"][i]),
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
    mean_tail_threshold = float(
        np.quantile(np.concatenate([metrics[key]["mean"] for key in metrics]), args.tail_quantile)
    )
    std_tail_threshold = float(
        np.quantile(np.concatenate([metrics[key]["std"] for key in metrics]), args.tail_quantile)
    )
    for key in metrics:
        aggregate[key]["tail_probability_mean"] = float(np.mean(metrics[key]["mean"] > mean_tail_threshold))
        aggregate[key]["tail_probability_std"] = float(np.mean(metrics[key]["std"] > std_tail_threshold))

    attack_gain = np.array([row["objective_gain"] for row in attack_rows], dtype=float)
    attack_norms = np.array([row["delta_l2_norm"] for row in attack_rows], dtype=float)

    payload = {
        "config": {
            "selected_indices": selected_indices.tolist(),
            "num_inputs": args.num_inputs,
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
            "attack_radius": args.attack_radius,
            "attack_steps": args.attack_steps,
            "attack_probe": args.attack_probe,
            "attack_step_size": args.attack_step_size,
            "attack_eval_samples": args.attack_eval_samples,
            "tail_quantile": args.tail_quantile,
        },
        "aggregate": aggregate,
        "attack_summary": {
            "mean_objective_gain": float(np.mean(attack_gain)),
            "median_objective_gain": float(np.median(attack_gain)),
            "mean_delta_l2_norm": float(np.mean(attack_norms)),
            "max_delta_l2_norm": float(np.max(attack_norms)),
        },
        "per_input": per_input_rows,
        "adversarial_optimization": attack_rows,
        "artifacts": {
            "mean_plot": str(mean_plot),
            "std_plot": str(std_plot),
        },
    }
    _save_json(json_path, payload)

    print("=== TrajFlow Adversarial GT Comparison ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"metric_samples: {args.metric_samples}")
    print(
        "attack summary: "
        f"mean_gain={payload['attack_summary']['mean_objective_gain']:.4f}, "
        f"mean_delta_l2={payload['attack_summary']['mean_delta_l2_norm']:.4f}"
    )
    print(f"saved: {mean_plot}")
    print(f"saved: {std_plot}")
    print(f"saved: {json_path}")
    print("aggregate metrics:")
    for name, values in aggregate.items():
        print(
            f"  - {name}: "
            f"mean_of_mean_distance={values['mean_of_mean_distance']:.4f}, "
            f"mean_of_std_distance={values['mean_of_std_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
