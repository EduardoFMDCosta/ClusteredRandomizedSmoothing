from __future__ import annotations

import argparse
import csv
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


def _prepare_inputs_from_arrays(
    selected_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
) -> torch.Tensor:
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

    x_input = inputs[selected_indices, :, :, :2]
    x_input = x_input[:, [1, 0]]
    x = torch.from_numpy(x_input.reshape(len(selected_indices), -1)).float()
    if not torch.isfinite(x).all():
        raise ValueError("Encountered non-finite input values after selection")

    return x


def _choose_indices(
    fixed_random_order: np.ndarray,
    num_inputs: int,
    selected_indices_text: str | None,
) -> np.ndarray:
    if selected_indices_text is not None and selected_indices_text.strip():
        return _parse_indices(selected_indices_text)

    if num_inputs <= 0:
        return fixed_random_order.astype(int)
    if num_inputs > len(fixed_random_order):
        raise ValueError(
            f"num-inputs={num_inputs} exceeds available valid GT samples={len(fixed_random_order)}"
        )
    return fixed_random_order[:num_inputs].astype(int)


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


def _extract_target_initial_position(x_positions: torch.Tensor, num_input_agents: int = 2) -> torch.Tensor:
    nx = x_positions.shape[0]
    num_input_timesteps = x_positions.shape[1] // (2 * num_input_agents)
    x_traj = x_positions.reshape(nx, num_input_agents, num_input_timesteps, 2)
    return x_traj[:, 0, 0, :]


def _slice_helper(helper: list[torch.Tensor], idx: int) -> list[torch.Tensor]:
    if not helper:
        return []
    return [h[idx: idx + 1] for h in helper]


def _random_sphere_points(
    num_points: int,
    dimension: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vec = torch.randn((num_points, dimension), device=device, dtype=dtype)
    vec = vec / torch.linalg.norm(vec, dim=1, keepdim=True).clamp(min=1e-12)
    return radius * vec


def _find_variance_maximizing_delta(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    model_input: torch.Tensor,
    helper: list[torch.Tensor],
    sigma: float,
    radius: float,
    num_candidates: int,
    num_samples: int,
) -> torch.Tensor:
    if radius <= 0.0:
        return torch.zeros_like(model_input)
    if num_candidates <= 0:
        raise ValueError("attack-num-candidates must be positive")
    if num_samples <= 0:
        raise ValueError("attack-num-samples must be positive")

    dimension = model_input.shape[-1]
    base_noise = MultivariateGaussian(
        mean=torch.zeros(dimension, dtype=model_input.dtype, device=model_input.device),
        cov=(sigma ** 2) * torch.eye(dimension, dtype=model_input.dtype, device=model_input.device),
    )

    delta = torch.zeros_like(model_input)
    for i in range(model_input.shape[0]):
        x_i = model_input[i: i + 1]
        h_i = _slice_helper(helper, i)

        candidates = _random_sphere_points(
            num_points=num_candidates,
            dimension=dimension,
            radius=radius,
            device=model_input.device,
            dtype=model_input.dtype,
        )

        best_score = float("-inf")
        best_delta = candidates[0]
        for c in range(num_candidates):
            delta_c = candidates[c: c + 1]
            x_adv = x_i + delta_c
            samples = stochastic_fn.sample_from_noisy_inputs(
                x=x_adv,
                smoothing_distribution=base_noise,
                num_samples=num_samples,
                h=h_i,
            )
            # Score = trace of empirical covariance of noisy outputs.
            score = float(samples.var(dim=1, unbiased=False).sum().item())
            if score > best_score:
                best_score = score
                best_delta = delta_c.squeeze(0)

        delta[i] = best_delta

    return delta


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_fixed_random_order(valid_indices: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(valid_indices)
    return order.astype(int)


def _load_or_create_fixed_random_order(
    path: Path,
    valid_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        payload = _load_json(path)
        ordered_indices = payload.get("ordered_indices", None)
        if not isinstance(ordered_indices, list):
            raise ValueError(f"Invalid ordered_indices in {path}")

        order = np.array([int(v) for v in ordered_indices], dtype=int)
        valid_set = set(int(v) for v in valid_indices.tolist())
        order_set = set(int(v) for v in order.tolist())

        if len(order) == len(valid_indices) and order_set == valid_set:
            return order

    order = _build_fixed_random_order(valid_indices=valid_indices, seed=seed)
    payload = {
        "seed": int(seed),
        "num_valid_indices": int(len(valid_indices)),
        "ordered_indices": order.tolist(),
    }
    _save_json(path, payload)
    return order


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "",
                "Clustered alpha-smoothing",
                "",
                "",
                "",
                "",
                "",
                "alpha-smoothing",
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "x_index",
                "set_containment_k2",
                "mc_set_containment_k2",
                "total_area_k2",
                "bound_region_1_k2",
                "bound_region_2_k2",
                "total_bound_k2",
                "set_containment_k1",
                "mc_set_containment_k1",
                "total_area_k1",
                "total_bound_k1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["x_index"],
                    row["set_containment_k2"],
                    row["mc_set_containment_k2"],
                    row["total_area_k2"],
                    row["bound_region_1_k2"],
                    row["bound_region_2_k2"],
                    row["total_bound_k2"],
                    row["set_containment_k1"],
                    row["mc_set_containment_k1"],
                    row["total_area_k1"],
                    row["total_bound_k1"],
                ]
            )


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No rows to print.")
        return

    group_header = [
        "",
        "Clustered alpha-smoothing",
        "",
        "",
        "",
        "",
        "",
        "alpha-smoothing",
        "",
        "",
        "",
    ]
    header = [
        "x_index",
        "set_contain_k2",
        "mc_set_contain_k2",
        "total_area_k2",
        "bound_r1_k2",
        "bound_r2_k2",
        "total_k2",
        "set_contain_k1",
        "mc_set_contain_k1",
        "total_area_k1",
        "total_k1",
    ]
    body = []
    for row in rows:
        body.append(
            [
                str(row["x_index"]),
                _format_float(float(row["set_containment_k2"])),
                _format_float(float(row["mc_set_containment_k2"])),
                _format_float(float(row["total_area_k2"])),
                _format_float(float(row["bound_region_1_k2"])),
                _format_float(float(row["bound_region_2_k2"])),
                _format_float(float(row["total_bound_k2"])),
                _format_float(float(row["set_containment_k1"])),
                _format_float(float(row["mc_set_containment_k1"])),
                _format_float(float(row["total_area_k1"])),
                _format_float(float(row["total_bound_k1"])),
            ]
        )

    widths = [
        max(len(header[i]), len(group_header[i]), *(len(r[i]) for r in body))
        for i in range(len(header))
    ]

    def _fmt(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    print(_fmt(group_header))
    print(_fmt(header))
    print("-+-".join("-" * w for w in widths))
    for values in body:
        print(_fmt(values))


def _compute_batch_table_rows(
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
        parameters=_build_parameters(args, max_num_clusters=1),
        h=helper,
        cluster_anchor=cluster_anchor,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=args.sigma,
        parameters=_build_parameters(args, max_num_clusters=2),
        h=helper,
        cluster_anchor=cluster_anchor,
    )

    radius = torch.full(
        (model_input.shape[0], 1),
        fill_value=args.certification_radius,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    cert_k2 = smoother_k2.robust_certification(radius=radius)
    cert_k1 = smoother_k1.robust_certification(radius=radius)

    # Uniform flow: keep clusters/certification from clean x, then evaluate
    # containment metrics on attacked x + delta (delta=0 when radius=0).
    delta = torch.zeros_like(model_input)
    if args.certification_radius > 0.0:
        delta = _find_variance_maximizing_delta(
            stochastic_fn=stochastic_fn,
            model_input=model_input,
            helper=helper,
            sigma=args.sigma,
            radius=args.certification_radius,
            num_candidates=args.attack_num_candidates,
            num_samples=args.attack_num_samples,
        )
    model_input_eval = model_input + delta

    base_noise = MultivariateGaussian(
        mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
        cov=(args.sigma ** 2)
        * torch.eye(model_input.shape[-1], dtype=model_input.dtype, device=model_input.device),
    )

    noisy_samples = stochastic_fn.sample_from_noisy_inputs(
        x=model_input_eval,
        smoothing_distribution=base_noise,
        num_samples=args.containment_samples,
        h=helper,
    )
    noisy_inside_k2 = smoother_k2.regions.contains(noisy_samples).any(dim=-1)
    noisy_inside_k1 = smoother_k1.regions.contains(noisy_samples).any(dim=-1)
    containment_k2 = noisy_inside_k2.float().mean(dim=1)
    containment_k1 = noisy_inside_k1.float().mean(dim=1)

    mc_samples_k2 = smoother_k2.monte_carlo(num_simulations=args.mc_containment_samples, z=model_input_eval)
    mc_samples_k1 = smoother_k1.monte_carlo(num_simulations=args.mc_containment_samples, z=model_input_eval)
    mc_inside_k2 = smoother_k2.regions.contains(mc_samples_k2).any(dim=-1)
    mc_inside_k1 = smoother_k1.regions.contains(mc_samples_k1).any(dim=-1)
    mc_containment_k2 = mc_inside_k2.float().mean(dim=1)
    mc_containment_k1 = mc_inside_k1.float().mean(dim=1)

    cert_k2_probs = cert_k2.probs.detach().cpu()
    cert_k2_total = cert_k2.total_probs.detach().cpu()
    cert_k1_total = cert_k1.total_probs.detach().cpu()
    k2_used = smoother_k2.cluster.k_per_row.detach().cpu().to(torch.int64)
    k1_used = smoother_k1.cluster.k_per_row.detach().cpu().to(torch.int64)

    k2_volume = smoother_k2.regions.volume.detach().cpu().to(torch.float64)
    k1_volume = smoother_k1.regions.volume.detach().cpu().to(torch.float64)
    k2_total_area = torch.zeros(k2_volume.shape[0], dtype=torch.float64)
    k1_total_area = torch.zeros(k1_volume.shape[0], dtype=torch.float64)
    for i in range(k2_volume.shape[0]):
        k2_total_area[i] = k2_volume[i, : int(k2_used[i].item())].sum()
        k1_total_area[i] = k1_volume[i, : int(k1_used[i].item())].sum()

    rows: list[dict[str, Any]] = []
    containment_k2_cpu = containment_k2.detach().cpu().numpy()
    containment_k1_cpu = containment_k1.detach().cpu().numpy()
    mc_containment_k2_cpu = mc_containment_k2.detach().cpu().numpy()
    mc_containment_k1_cpu = mc_containment_k1.detach().cpu().numpy()
    k2_total_area_cpu = k2_total_area.numpy()
    k1_total_area_cpu = k1_total_area.numpy()

    for i, dataset_idx in enumerate(batch_indices.tolist()):
        row: dict[str, Any] = {
            "x_index": int(dataset_idx),
            "set_containment_k2": float(containment_k2_cpu[i]),
            "mc_set_containment_k2": float(mc_containment_k2_cpu[i]),
            "total_area_k2": float(k2_total_area_cpu[i]),
            "bound_region_1_k2": float("nan"),
            "bound_region_2_k2": float("nan"),
            "total_bound_k2": float(cert_k2_total[i].item()),
            "set_containment_k1": float(containment_k1_cpu[i]),
            "mc_set_containment_k1": float(mc_containment_k1_cpu[i]),
            "total_area_k1": float(k1_total_area_cpu[i]),
            "total_bound_k1": float(cert_k1_total[i].item()),
        }
        if int(k2_used[i].item()) >= 1 and cert_k2_probs.shape[1] >= 1:
            row["bound_region_1_k2"] = float(cert_k2_probs[i, 0].item())
        if int(k2_used[i].item()) >= 2 and cert_k2_probs.shape[1] >= 2:
            row["bound_region_2_k2"] = float(cert_k2_probs[i, 1].item())
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-input TrajFlow table with noisy set-containment and "
            "robust certification bounds for k=1 and k=2 smoothers."
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
        "--containment-samples",
        type=int,
        default=1000,
        help="Number of h_w(x+epsilon) noisy samples per input used to estimate set-containment proportion.",
    )
    parser.add_argument(
        "--mc-containment-samples",
        type=int,
        default=1000,
        help="Number of Monte Carlo samples from each smoother used to estimate self-containment.",
    )
    parser.add_argument(
        "--attack-num-candidates",
        type=int,
        default=64,
        help="Number of random attack candidates per input when certification-radius > 0.",
    )
    parser.add_argument(
        "--attack-num-samples",
        type=int,
        default=128,
        help="Noisy samples per attack candidate used to score output variance.",
    )
    parser.add_argument(
        "--certification-radius",
        type=float,
        default=0.0,
        help="Input perturbation radius used in robust certification.",
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
        help="Sub-batch size along the noise/sample dimension inside stochastic sampling.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=1000,
        help="num_samples for confidence parameter (used in smoothing object setup).",
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
        default=str(EXPERIMENTS / "data" / "analysis" / "performance"),
        help="Directory where table JSON/CSV are saved.",
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
    if args.containment_samples <= 0:
        raise ValueError("containment-samples must be positive")
    if args.mc_containment_samples <= 0:
        raise ValueError("mc-containment-samples must be positive")
    if args.attack_num_candidates <= 0:
        raise ValueError("attack-num-candidates must be positive")
    if args.attack_num_samples <= 0:
        raise ValueError("attack-num-samples must be positive")
    if args.noise_sub_batch_size <= 0:
        raise ValueError("noise-sub-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if args.certification_radius < 0.0:
        raise ValueError("certification-radius must be non-negative")

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

    table_rows: list[dict[str, Any]] = []
    num_batches = (len(selected_indices) + args.input_batch_size - 1) // args.input_batch_size
    for batch_id, start in enumerate(range(0, len(selected_indices), args.input_batch_size), start=1):
        batch_t0 = time.perf_counter()
        batch_indices = selected_indices[start : start + args.input_batch_size]
        print(
            f"processing batch {batch_id}/{num_batches} "
            f"with {len(batch_indices)} dataset indices"
        )
        rows = _compute_batch_table_rows(
            args=args,
            stochastic_fn=stochastic_fn,
            batch_indices=batch_indices,
            inputs=all_inputs,
            outputs_gt=outputs_gt,
        )
        table_rows.extend(rows)
        print(f"batch {batch_id}/{num_batches} completed in {time.perf_counter() - batch_t0:.1f}s")

    json_path = analysis_dir / "performance_table.json"
    csv_path = analysis_dir / "performance_table.csv"

    payload = {
        "config": {
            "selected_indices": selected_indices.tolist(),
            "containment_samples": args.containment_samples,
            "mc_containment_samples": args.mc_containment_samples,
            "attack_num_candidates": args.attack_num_candidates,
            "attack_num_samples": args.attack_num_samples,
            "certification_radius": args.certification_radius,
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
            "fixed_index_order_file": str(fixed_order_path),
        },
        "column_groups": [
            {"label": "Clustered alpha-smoothing", "columns": [1, 2, 3, 4, 5, 6]},
            {"label": "alpha-smoothing", "columns": [7, 8, 9, 10]},
        ],
        "columns": [
            "x_index",
            "set_containment_k2",
            "mc_set_containment_k2",
            "total_area_k2",
            "bound_region_1_k2",
            "bound_region_2_k2",
            "total_bound_k2",
            "set_containment_k1",
            "mc_set_containment_k1",
            "total_area_k1",
            "total_bound_k1",
        ],
        "rows": table_rows,
    }

    _save_json(json_path, payload)
    _save_csv(csv_path, table_rows)

    print("=== TrajFlow Performance Table ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    _print_table(table_rows)


if __name__ == "__main__":
    main()
