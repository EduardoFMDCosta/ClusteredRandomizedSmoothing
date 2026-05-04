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
from plotting import plot_trajflow_adversarial_from_json
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
        return valid_indices.astype(int)
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


def _compute_behavior_probabilities(
    samples: torch.Tensor,
    anchor: torch.Tensor,
    stay_radius: float,
    min_down: float,
    min_left: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(
            "Expected samples with shape (nx, num_samples, 2), "
            f"got {tuple(samples.shape)}"
        )
    if anchor.ndim != 2 or anchor.shape[-1] != 2 or anchor.shape[0] != samples.shape[0]:
        raise ValueError(
            "Expected anchor with shape (nx, 2) matching samples; "
            f"got samples={tuple(samples.shape)}, anchor={tuple(anchor.shape)}"
        )

    delta = samples - anchor.unsqueeze(1)
    stay_mask = torch.linalg.norm(delta, dim=-1) <= stay_radius
    turn_mask = (
        (delta[..., 1] <= -min_down)
        & (delta[..., 0] <= -min_left)
        & (~stay_mask)
    )
    p_stay = stay_mask.float().mean(dim=1)
    p_turn = turn_mask.float().mean(dim=1)
    return p_stay, p_turn


def _estimate_cluster_separation_objective(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_position: torch.Tensor,
    y_target: torch.Tensor,
    input_mode: str,
    num_samples: int,
    stay_radius: float,
    min_down: float,
    min_left: float,
    dominance_target: float = 0.60,
    minority_target: float = 0.10,
    coverage_target: float = 0.70,
    separation_weight: float = 0.75,
    dominance_penalty: float = 20.0,
    minority_penalty: float = 8.0,
    coverage_penalty: float = 12.0,
    dominant_is_stay: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate the regime where k=1 degrades while k=2 stays correct.

    The objective rewards a large gap between the raw-sample mean error (a
    crude proxy for k=1) and the dominant-cluster error (a proxy for k=2),
    while also encouraging mode separation and penalizing collapse of the clean
    dominant mode or disappearance of the minority mode.
    """
    model_input, helper = _convert_position_input(
        stochastic_fn=stochastic_fn,
        x_position=x_position,
        input_mode=input_mode,
    )
    samples = _sample_original(
        stochastic_fn=stochastic_fn,
        model_input=model_input,
        helper=helper,
        num_samples=num_samples,
    )
    anchor = _extract_target_initial_position(x_position)
    p_stay, p_turn = _compute_behavior_probabilities(
        samples=samples,
        anchor=anchor,
        stay_radius=stay_radius,
        min_down=min_down,
        min_left=min_left,
    )

    if dominant_is_stay is None:
        dominant_is_stay = p_stay >= p_turn
    else:
        dominant_is_stay = dominant_is_stay.to(device=p_stay.device)

    delta = samples - anchor.unsqueeze(1)
    stay_mask = torch.linalg.norm(delta, dim=-1) <= stay_radius
    turn_mask = (
        (delta[..., 1] <= -min_down)
        & (delta[..., 0] <= -min_left)
        & (~stay_mask)
    )

    dominant_mask = torch.where(
        dominant_is_stay.unsqueeze(1),
        stay_mask,
        turn_mask,
    )
    minor_mask = torch.where(
        dominant_is_stay.unsqueeze(1),
        turn_mask,
        stay_mask,
    )

    eps = 1e-8
    dominant_sum = (samples * dominant_mask.unsqueeze(-1).float()).sum(dim=1)
    dominant_count = dominant_mask.float().sum(dim=1, keepdim=True).clamp(min=eps)
    dominant_mean = dominant_sum / dominant_count

    minor_sum = (samples * minor_mask.unsqueeze(-1).float()).sum(dim=1)   # (nx, 2)
    minor_count = minor_mask.float().sum(dim=1, keepdim=True).clamp(min=eps)  # (nx, 1)
    minor_mean = minor_sum / minor_count  # (nx, 2)

    p_dominant = torch.where(dominant_is_stay, p_stay, p_turn)
    p_minor = torch.where(dominant_is_stay, p_turn, p_stay)
    covered_mass = torch.clamp(p_dominant + p_minor, min=eps)
    classified_mean = (
        p_dominant.unsqueeze(1) * dominant_mean
        + p_minor.unsqueeze(1) * minor_mean
    ) / covered_mass.unsqueeze(1)

    mean_error = torch.linalg.norm(classified_mean - y_target, dim=-1)
    dominant_error = torch.linalg.norm(dominant_mean - y_target, dim=-1)
    separation = torch.linalg.norm(dominant_mean - minor_mean, dim=-1)

    dominance_shortfall = torch.relu(dominance_target - p_dominant)
    minority_shortfall = torch.relu(minority_target - p_minor)
    coverage_shortfall = torch.relu(coverage_target - covered_mass)
    objective = (
        mean_error
        - dominant_error
        + separation_weight * p_minor * separation
        - dominance_penalty * dominance_shortfall
        - minority_penalty * minority_shortfall
        - coverage_penalty * coverage_shortfall
    )

    return objective, p_stay, p_turn, dominant_is_stay


def _compute_adversarial_delta_spsa(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    x_clean: torch.Tensor,
    y_target: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    radius = args.attack_radius
    clean_obj, clean_p_stay, clean_p_turn, dominant_is_stay = _estimate_cluster_separation_objective(
        stochastic_fn=stochastic_fn,
        x_position=x_clean,
        y_target=y_target,
        input_mode=args.input_mode,
        num_samples=args.attack_eval_samples,
        stay_radius=args.behavior_stay_radius,
        min_down=args.behavior_min_down,
        min_left=args.behavior_min_left,
        dominant_is_stay=None,
    )

    if radius <= 0.0:
        zero = torch.zeros_like(x_clean)
        clean_np = clean_obj.detach().cpu().numpy()
        details = {
            "clean_p_stay": clean_p_stay.detach().cpu().numpy(),
            "clean_p_turn": clean_p_turn.detach().cpu().numpy(),
            "adv_p_stay": clean_p_stay.detach().cpu().numpy(),
            "adv_p_turn": clean_p_turn.detach().cpu().numpy(),
            "dominant_is_stay": dominant_is_stay.detach().cpu().numpy().astype(bool),
        }
        return zero, clean_np, clean_np.copy(), details

    delta = torch.zeros_like(x_clean)
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

        obj_plus, _, _, _ = _estimate_cluster_separation_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta_plus,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
            stay_radius=args.behavior_stay_radius,
            min_down=args.behavior_min_down,
            min_left=args.behavior_min_left,
            dominant_is_stay=dominant_is_stay,
        )
        obj_minus, _, _, _ = _estimate_cluster_separation_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta_minus,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
            stay_radius=args.behavior_stay_radius,
            min_down=args.behavior_min_down,
            min_left=args.behavior_min_left,
            dominant_is_stay=dominant_is_stay,
        )

        scale = ((obj_plus - obj_minus) / (2.0 * args.attack_probe)).unsqueeze(1)
        grad_est = scale * direction

        delta = _project_l2_ball(delta + args.attack_step_size * grad_est, radius)

        obj_current, _, _, _ = _estimate_cluster_separation_objective(
            stochastic_fn=stochastic_fn,
            x_position=x_clean + delta,
            y_target=y_target,
            input_mode=args.input_mode,
            num_samples=args.attack_eval_samples,
            stay_radius=args.behavior_stay_radius,
            min_down=args.behavior_min_down,
            min_left=args.behavior_min_left,
            dominant_is_stay=dominant_is_stay,
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

    _, adv_p_stay, adv_p_turn, _ = _estimate_cluster_separation_objective(
        stochastic_fn=stochastic_fn,
        x_position=x_clean + best_delta,
        y_target=y_target,
        input_mode=args.input_mode,
        num_samples=args.attack_eval_samples,
        stay_radius=args.behavior_stay_radius,
        min_down=args.behavior_min_down,
        min_left=args.behavior_min_left,
        dominant_is_stay=dominant_is_stay,
    )

    clean_np = clean_obj.detach().cpu().numpy()
    best_np = best_obj.detach().cpu().numpy()
    details = {
        "clean_p_stay": clean_p_stay.detach().cpu().numpy(),
        "clean_p_turn": clean_p_turn.detach().cpu().numpy(),
        "adv_p_stay": adv_p_stay.detach().cpu().numpy(),
        "adv_p_turn": adv_p_turn.detach().cpu().numpy(),
        "dominant_is_stay": dominant_is_stay.detach().cpu().numpy().astype(bool),
    }
    return best_delta, clean_np, best_np, details


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
        description=(
            "Adversarial TrajFlow experiment with the cluster-separation objective: optimize "
            "bounded perturbations on the original model, then compare original/noisy/smoothed "
            "predictors on x+delta."
        )
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=0,
        help="How many valid inputs to sample when selected-indices is not provided. Use <=0 for all valid indices.",
    )
    parser.add_argument(
        "--selected-indices",
        type=str,
        default="",
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
        default=10,
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
        default=0.22,
        help="Step size for projected SPSA ascent updates.",
    )
    parser.add_argument(
        "--attack-eval-samples",
        type=int,
        default=32,
        help="Number of stochastic samples for each adversarial objective estimate.",
    )
    parser.add_argument(
        "--attack-objective",
        type=str,
        choices=("cluster_separation",),
        default="cluster_separation",
        help=(
            "Objective optimized by SPSA on the original model. "
            "cluster_separation pushes the minor-mode cluster away from GT while "
            "preserving a strong dominant mode to amplify the k=1 vs k=2 gap."
        ),
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
        help="Clustering strategy for k=2 smoothing.",
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
        help="num_samples for confidence parameter (used in smoothing setup).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for index selection and stochastic search.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis" / "adversarial"),
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
        "--regenerate-data",
        type=str,
        choices=("y", "n"),
        default=None,
        help="Optional non-interactive override: y to regenerate, n to use existing JSON.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "adversarial_metrics.json"

    if args.regenerate_data is None:
        regenerate_data = _ask_yes_no(
            "Regenerate adversarial data now? [y/n] "
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
            plot_trajflow_adversarial_from_json(
                metrics_json=str(json_path),
                top_k=args.plot_top_k,
                tail_quantile=args.tail_quantile,
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

    metric_names = ("original", "noisy_input", "smoothed_k1", "smoothed_k2")
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
        delta_adv, clean_obj, adv_obj, attack_details = _compute_adversarial_delta_spsa(
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
                    "clean_p_stay": float(attack_details["clean_p_stay"][i]),
                    "clean_p_turn": float(attack_details["clean_p_turn"][i]),
                    "adversarial_p_stay": float(attack_details["adv_p_stay"][i]),
                    "adversarial_p_turn": float(attack_details["adv_p_turn"][i]),
                    "dominant_clean_behavior": (
                        "stay"
                        if bool(attack_details["dominant_is_stay"][i])
                        else "turn"
                    ),
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
            "attack_objective": args.attack_objective,
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
    }
    _save_json(json_path, payload)

    tail_stats_path = analysis_dir / "adversarial_tail_stats.json"
    _save_json(tail_stats_path, _compute_tail_stats_table(metrics))

    print("=== TrajFlow Adversarial GT Comparison ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"metric_samples: {args.metric_samples}")
    print(
        "attack summary: "
        f"mean_gain={payload['attack_summary']['mean_objective_gain']:.4f}, "
        f"mean_delta_l2={payload['attack_summary']['mean_delta_l2_norm']:.4f}"
    )
    print(f"saved: {json_path}")
    print(f"saved: {tail_stats_path}")
    print("aggregate metrics:")
    for name, values in aggregate.items():
        print(
            f"  - {name}: "
            f"mean_of_mean_distance={values['mean_of_mean_distance']:.4f}, "
            f"mean_of_std_distance={values['mean_of_std_distance']:.4f}"
        )

    plot_trajflow_adversarial_from_json(
        metrics_json=str(json_path),
        top_k=args.plot_top_k,
        tail_quantile=args.tail_quantile,
        output_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
