from __future__ import annotations

import argparse
import csv
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

from randomized_smoothing.dataclasses import (  # noqa: E402
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RuntimeHeuristics,
    SmoothingParameters,
)
from randomized_smoothing.distributions import MultivariateGaussian  # noqa: E402
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing  # noqa: E402
from stochastic_functions import TrajFlowMeszarosStochasticFunction  # noqa: E402


def _parse_indices(text: str) -> np.ndarray:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("selected-indices must contain at least one integer")
    return np.array([int(v) for v in values], dtype=int)


def _parse_radius_values(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("radius-values must contain at least one value")
    radii = [float(v) for v in values]
    if any(r < 0.0 for r in radii):
        raise ValueError("all radius values must be non-negative")
    return sorted(set(radii))


def _parse_alpha_values(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("alpha-values must contain at least one value")
    alphas = [float(v) for v in values]
    if any(a < 0.0 or a >= 1.0 for a in alphas):
        raise ValueError("all alpha values must satisfy 0 <= alpha < 1")
    return sorted(set(alphas))


def _alpha_tag(alpha: float) -> str:
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _with_alpha_suffix(path: Path, alpha: float) -> Path:
    return path.with_name(f"{path.stem}_alpha{_alpha_tag(alpha)}{path.suffix}")


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


def _ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter y or n.")


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


def _save_csv(path: Path, rows: list[dict[str, Any]], header: list[str], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[k] for k in keys])


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


def _extract_target_initial_position(x_positions: torch.Tensor, num_input_agents: int = 2) -> torch.Tensor:
    nx = x_positions.shape[0]
    num_input_timesteps = x_positions.shape[1] // (2 * num_input_agents)
    x_traj = x_positions.reshape(nx, num_input_agents, num_input_timesteps, 2)
    return x_traj[:, 0, 0, :]


def _slice_helper(helper: list[torch.Tensor], idx: int) -> list[torch.Tensor]:
    if not helper:
        return []
    return [h[idx : idx + 1] for h in helper]


def _random_ball_points(
    num_points: int,
    dimension: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if radius <= 0.0 or num_points <= 1:
        return torch.zeros((1, dimension), device=device, dtype=dtype)

    vec = torch.randn((num_points - 1, dimension), device=device, dtype=dtype)
    vec = vec / torch.linalg.norm(vec, dim=1, keepdim=True).clamp(min=1e-12)
    scales = torch.rand((num_points - 1, 1), device=device, dtype=dtype)
    scales = radius * scales.pow(1.0 / float(dimension))
    sampled = scales * vec
    origin = torch.zeros((1, dimension), device=device, dtype=dtype)
    return torch.cat((origin, sampled), dim=0)


def _build_parameters(args: argparse.Namespace, max_num_clusters: int) -> Parameters:
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
            max_num_clusters=2,
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
        runtime=RuntimeHeuristics(noise_sub_batch_size=args.noise_sub_batch_size),
    )


def _estimate_noisy_containment(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    smoother: GaussianAlphaTrimmingSmoothing,
    x_single: torch.Tensor,
    helper_single: list[torch.Tensor],
    sigma: float,
    num_samples: int,
    noise_sub_batch_size: int,
) -> float:
    dimension = x_single.shape[-1]
    base_noise = MultivariateGaussian(
        mean=torch.zeros(dimension, dtype=x_single.dtype, device=x_single.device),
        cov=(sigma**2) * torch.eye(dimension, dtype=x_single.dtype, device=x_single.device),
    )
    noisy_samples = stochastic_fn.sample_from_noisy_inputs(
        x=x_single,
        smoothing_distribution=base_noise,
        num_samples=num_samples,
        h=helper_single,
        noise_sub_batch_size=noise_sub_batch_size,
    )
    inside = smoother.regions.contains(noisy_samples).any(dim=-1)
    return float(inside.float().mean().item())


def _estimate_mc_containment(
    smoother: GaussianAlphaTrimmingSmoothing,
    z_single: torch.Tensor,
    num_samples: int,
    row_index: int,
) -> float:
    z_full = smoother.x.detach().clone()
    z_full[row_index : row_index + 1] = z_single
    mc_samples = smoother.monte_carlo(num_simulations=num_samples, z=z_full)
    inside = smoother.regions.contains(mc_samples).any(dim=-1)
    return float(inside[row_index].float().mean().item())


def _find_attack_minimizing_containment(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    smoother: GaussianAlphaTrimmingSmoothing,
    x_single: torch.Tensor,
    helper_single: list[torch.Tensor],
    sigma: float,
    radius: float,
    num_candidates: int,
    containment_samples: int,
    mc_containment_samples: int,
    noise_sub_batch_size: int,
    objective: str,
    warm_start_delta: torch.Tensor | None = None,
    row_index: int | None = None,
) -> tuple[torch.Tensor, float]:
    if objective not in ("noisy", "mc"):
        raise ValueError(f"Unsupported objective: {objective}")

    dimension = x_single.shape[-1]
    candidates = _random_ball_points(
        num_points=max(num_candidates, 1),
        dimension=dimension,
        radius=radius,
        device=x_single.device,
        dtype=x_single.dtype,
    )
    if warm_start_delta is not None:
        warm = warm_start_delta.to(device=x_single.device, dtype=x_single.dtype).view(1, dimension)
        warm_norm = float(torch.linalg.norm(warm).item())
        if warm_norm <= radius + 1e-12:
            candidates = torch.cat((candidates, warm), dim=0)

    best_delta = candidates[0].clone()
    best_score = float("inf")
    for c in range(candidates.shape[0]):
        delta = candidates[c : c + 1]
        z_adv = x_single + delta
        if objective == "noisy":
            containment = _estimate_noisy_containment(
                stochastic_fn=stochastic_fn,
                smoother=smoother,
                x_single=z_adv,
                helper_single=helper_single,
                sigma=sigma,
                num_samples=containment_samples,
                noise_sub_batch_size=noise_sub_batch_size,
            )
        else:
            if row_index is None:
                raise ValueError("row_index is required for mc objective")
            containment = _estimate_mc_containment(
                smoother=smoother,
                z_single=z_adv,
                num_samples=mc_containment_samples,
                row_index=row_index,
            )

        if containment < best_score:
            best_score = containment
            best_delta = delta.squeeze(0).clone()

    return best_delta, float(best_score)


def _compute_batch_radius_rows(
    args: argparse.Namespace,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    batch_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
    radii: list[float],
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

    smoothers = {
        "k2": smoother_k2,
        "k1": smoother_k1,
    }

    # Important: smoothers are fit once on clean inputs and then reused.
    # Attacks only perturb evaluation points z=x+delta for containment/MC.

    rows: list[dict[str, Any]] = []
    # Warm-start cache keyed by (smoother, objective, local input index).
    warm_starts: dict[tuple[str, str, int], torch.Tensor] = {}
    for radius in radii:
        radius_tensor = torch.full(
            (model_input.shape[0], 1),
            fill_value=radius,
            dtype=model_input.dtype,
            device=model_input.device,
        )

        cert_k2 = smoother_k2.robust_certification(radius=radius_tensor).total_probs.detach().cpu().numpy()
        cert_k1 = smoother_k1.robust_certification(radius=radius_tensor).total_probs.detach().cpu().numpy()
        cert_map = {
            "k2": cert_k2,
            "k1": cert_k1,
        }

        for i, dataset_idx in enumerate(batch_indices.tolist()):
            x_i = model_input[i : i + 1]
            helper_i = _slice_helper(helper, i)

            for smoother_key, smoother in smoothers.items():
                cert_total = float(cert_map[smoother_key][i])
                noisy_key = (smoother_key, "noisy", i)
                mc_key = (smoother_key, "mc", i)

                noisy_delta, noisy_prop = _find_attack_minimizing_containment(
                    stochastic_fn=stochastic_fn,
                    smoother=smoother,
                    x_single=x_i,
                    helper_single=helper_i,
                    sigma=args.sigma,
                    radius=radius,
                    num_candidates=args.attack_num_candidates,
                    containment_samples=args.containment_samples,
                    mc_containment_samples=args.mc_containment_samples,
                    noise_sub_batch_size=args.noise_sub_batch_size,
                    objective="noisy",
                    warm_start_delta=warm_starts.get(noisy_key),
                    row_index=i,
                )
                mc_delta, mc_prop = _find_attack_minimizing_containment(
                    stochastic_fn=stochastic_fn,
                    smoother=smoother,
                    x_single=x_i,
                    helper_single=helper_i,
                    sigma=args.sigma,
                    radius=radius,
                    num_candidates=args.attack_num_candidates,
                    containment_samples=args.containment_samples,
                    mc_containment_samples=args.mc_containment_samples,
                    noise_sub_batch_size=args.noise_sub_batch_size,
                    objective="mc",
                    warm_start_delta=warm_starts.get(mc_key),
                    row_index=i,
                )
                warm_starts[noisy_key] = noisy_delta.detach().clone()
                warm_starts[mc_key] = mc_delta.detach().clone()

                rows.append(
                    {
                        "x_index": int(dataset_idx),
                        "radius": float(radius),
                        "smoother": smoother_key,
                        "cert_total": cert_total,
                        "noisy_adv_prop": float(noisy_prop),
                        "mc_adv_prop": float(mc_prop),
                        "gap_noisy": float(noisy_prop - cert_total),
                        "gap_mc": float(mc_prop - cert_total),
                        "noisy_attack_norm": float(torch.linalg.norm(noisy_delta).item()),
                        "mc_attack_norm": float(torch.linalg.norm(mc_delta).item()),
                    }
                )

    return rows


def _summarize_by_radius(rows: list[dict[str, Any]], radii: list[float]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    smoother_order = ["k2", "k1"]

    for smoother in smoother_order:
        for radius in radii:
            subset = [r for r in rows if r["smoother"] == smoother and float(r["radius"]) == float(radius)]
            if not subset:
                continue

            noisy_gaps = np.array([float(r["gap_noisy"]) for r in subset], dtype=float)
            mc_gaps = np.array([float(r["gap_mc"]) for r in subset], dtype=float)
            noisy_props = np.array([float(r["noisy_adv_prop"]) for r in subset], dtype=float)

            summary_rows.append(
                {
                    "smoother": smoother,
                    "radius": float(radius),
                    "num_inputs": int(len(subset)),
                    "gap_noisy_mean": float(np.mean(noisy_gaps)),
                    "gap_noisy_std": float(np.std(noisy_gaps, ddof=0)),
                    "gap_mc_mean": float(np.mean(mc_gaps)),
                    "gap_mc_std": float(np.std(mc_gaps, ddof=0)),
                    "noisy_adv_prop_mean": float(np.mean(noisy_props)),
                    "noisy_adv_prop_std": float(np.std(noisy_props, ddof=0)),
                }
            )

    return summary_rows


def _directional_band(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ax: plt.Axes,
    color: str,
    alpha: float,
) -> None:
    if np.all(mean >= 0.0):
        lower = mean
        upper = mean + std
    elif np.all(mean <= 0.0):
        lower = mean - std
        upper = mean
    else:
        lower = mean - std
        upper = mean + std
    ax.fill_between(x, lower, upper, color=color, alpha=alpha)


def _plot_side_by_side_gaps(summary_rows: list[dict[str, Any]], output_path: Path, alpha: float) -> None:
    if not summary_rows:
        raise ValueError("summary_rows must be non-empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=False)
        left_ax, right_ax = axes

        alpha_display = f"{alpha:g}"
        smoother_lines = [
            ("k2", "#F58518", "s", rf"Ours ($\alpha={alpha_display}$)"),
            ("k1", "#4C78A8", "o", rf"$\alpha$-smoothing ($\alpha={alpha_display}$)"),
        ]

        for smoother, color, marker, label in smoother_lines:
            subset = [r for r in summary_rows if r["smoother"] == smoother]
            subset = sorted(subset, key=lambda r: float(r["radius"]))
            x = np.array([float(r["radius"]) for r in subset], dtype=float)

            gap_mc_mean = np.array([float(r["gap_mc_mean"]) for r in subset], dtype=float)
            gap_mc_std = np.array([float(r["gap_mc_std"]) for r in subset], dtype=float)
            noisy_prop_mean = np.array([float(r["noisy_adv_prop_mean"]) for r in subset], dtype=float)
            noisy_prop_std = np.array([float(r["noisy_adv_prop_std"]) for r in subset], dtype=float)

            # Left panel: theorem-oriented gap with directional uncertainty band.
            _directional_band(
                x=x,
                mean=gap_mc_mean,
                std=gap_mc_std,
                ax=left_ax,
                color=color,
                alpha=0.18,
            )
            left_ax.plot(x, gap_mc_mean, marker=marker, linewidth=1.6, color=color, label=label)

            # Right panel: noisy containment proportion with symmetric mean +/- std band.
            right_lower = np.clip(noisy_prop_mean - noisy_prop_std, 0.0, 1.0)
            right_upper = np.clip(noisy_prop_mean + noisy_prop_std, 0.0, 1.0)
            right_ax.fill_between(
                x,
                right_lower,
                right_upper,
                color=color,
                alpha=0.18,
            )
            right_ax.plot(x, noisy_prop_mean, marker=marker, linewidth=1.6, color=color, label=label)

        left_ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
        left_ax.set_xlabel(r"Robustness radius $r$")
        left_ax.set_ylabel("Probability gap")
        left_ax.grid(True, linestyle=":", alpha=0.35)
        left_ax.legend(loc="best", frameon=True)

        right_ax.set_xlabel(r"Robustness radius $r$")
        right_ax.set_ylabel(r"Proportion $h_w(x+\delta+\varepsilon) \in \mathcal{R}$")
        right_ax.grid(True, linestyle=":", alpha=0.35)
        right_ax.legend(loc="best", frameon=True)

        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def _print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        print("No summary rows to display.")
        return

    header = [
        "Smoother",
        "Radius",
        "n",
        "Gap MC mean",
        "Gap MC std",
        "Noisy prop mean",
        "Noisy prop std",
    ]
    keys = [
        "smoother",
        "radius",
        "num_inputs",
        "gap_mc_mean",
        "gap_mc_std",
        "noisy_adv_prop_mean",
        "noisy_adv_prop_std",
    ]

    def _format(v: Any) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v)

    body = [[_format(r[k]) for k in keys] for r in summary_rows]
    widths = [max(len(header[i]), *(len(row[i]) for row in body)) for i in range(len(header))]

    def _fmt_row(row: list[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print(_fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in body:
        print(_fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TrajFlow robustness-gap analysis: for each x_i and each radius, compute robust "
            "certificate total probability and two adversarial containment gaps for k=2 and k=1 smoothers."
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
        "--radius-values",
        type=str,
        default="0.0,0.1,0.5,0.8",
        help="Comma-separated robustness radii.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=5,
        help="How many selected inputs to process at once.",
    )
    parser.add_argument(
        "--attack-num-candidates",
        type=int,
        default=20,
        help="Number of random attack candidates in the ball ||delta||<=radius for each objective.",
    )
    parser.add_argument(
        "--containment-samples",
        type=int,
        default=500,
        help="Noisy samples used to estimate containment for h_w(x+delta+epsilon).",
    )
    parser.add_argument(
        "--mc-containment-samples",
        type=int,
        default=500,
        help="Monte Carlo samples used to estimate smoother containment at (x+delta).",
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
        default=8,
        help="N for alpha-trimming smoothing.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional single trimming parameter alpha. Overrides --alpha-values when provided.",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.0,0.2,0.4",
        help="Comma-separated alpha values to evaluate when --alpha is not provided.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.85,
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
        default=200,
        help="Sub-batch size along the noise/sample dimension inside stochastic sampling.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=2000,
        help="num_samples for confidence parameter used in smoothing object setup.",
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
        help="Directory where robustness analysis JSON/CSV are saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where the side-by-side gaps figure is saved.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="trajflow_robustness_gaps.pdf",
        help="Filename for side-by-side gap plots (k2 and k1).",
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
            "If --no-regenerate is set, loads from existing JSON. "
            "If omitted, the script asks interactively when JSON already exists."
        ),
    )

    args = parser.parse_args()

    if args.input_batch_size <= 0:
        raise ValueError("input-batch-size must be positive")
    if args.attack_num_candidates <= 0:
        raise ValueError("attack-num-candidates must be positive")
    if args.containment_samples <= 0:
        raise ValueError("containment-samples must be positive")
    if args.mc_containment_samples <= 0:
        raise ValueError("mc-containment-samples must be positive")
    if args.noise_sub_batch_size <= 0:
        raise ValueError("noise-sub-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")

    radii = _parse_radius_values(args.radius_values)
    alpha_values = [float(args.alpha)] if args.alpha is not None else _parse_alpha_values(args.alpha_values)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    analysis_dir = Path(args.analysis_dir)
    figures_dir = Path(args.figures_dir)
    rows_json_base_path = analysis_dir / "trajflow_robustness_gap_rows.json"
    rows_csv_base_path = analysis_dir / "trajflow_robustness_gap_rows.csv"
    summary_json_base_path = analysis_dir / "trajflow_robustness_gap_summary.json"
    summary_csv_base_path = analysis_dir / "trajflow_robustness_gap_summary.csv"
    fig_base_path = figures_dir / args.plot_name

    rows_json_paths = [_with_alpha_suffix(rows_json_base_path, alpha) for alpha in alpha_values]

    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif any(path.exists() for path in rows_json_paths):
        do_regenerate = _ask_yes_no(
            "Found existing alpha-specific robustness data.\nRegenerate? [y/n]: "
        )
    else:
        do_regenerate = True

    if do_regenerate:
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

    for alpha in alpha_values:
        args.alpha = alpha
        rows_json_path = _with_alpha_suffix(rows_json_base_path, alpha)
        rows_csv_path = _with_alpha_suffix(rows_csv_base_path, alpha)
        summary_json_path = _with_alpha_suffix(summary_json_base_path, alpha)
        summary_csv_path = _with_alpha_suffix(summary_csv_base_path, alpha)
        fig_path = _with_alpha_suffix(fig_base_path, alpha)

        if do_regenerate:
            all_rows: list[dict[str, Any]] = []
            num_batches = (len(selected_indices) + args.input_batch_size - 1) // args.input_batch_size
            for batch_id, start in enumerate(range(0, len(selected_indices), args.input_batch_size), start=1):
                t0 = time.perf_counter()
                batch_indices = selected_indices[start : start + args.input_batch_size]
                print(
                    f"alpha={alpha:.3f}: processing batch {batch_id}/{num_batches} "
                    f"with {len(batch_indices)} dataset indices"
                )
                batch_rows = _compute_batch_radius_rows(
                    args=args,
                    stochastic_fn=stochastic_fn,
                    batch_indices=batch_indices,
                    inputs=all_inputs,
                    outputs_gt=outputs_gt,
                    radii=radii,
                )
                all_rows.extend(batch_rows)
                print(
                    f"alpha={alpha:.3f}: batch {batch_id}/{num_batches} "
                    f"completed in {time.perf_counter() - t0:.1f}s"
                )

            row_config = {
                "selected_indices": selected_indices.tolist(),
                "radii": [float(r) for r in radii],
                "attack_num_candidates": args.attack_num_candidates,
                "containment_samples": args.containment_samples,
                "mc_containment_samples": args.mc_containment_samples,
                "sigma": args.sigma,
                "smoothing_N": args.smoothing_N,
                "alpha": args.alpha,
                "alpha_values": [float(a) for a in alpha_values],
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
            }

            _save_json(rows_json_path, {"config": row_config, "rows": all_rows})
            _save_csv(
                rows_csv_path,
                rows=all_rows,
                header=[
                    "x_index",
                    "radius",
                    "smoother",
                    "cert_total",
                    "noisy_adv_prop",
                    "mc_adv_prop",
                    "gap_noisy",
                    "gap_mc",
                    "noisy_attack_norm",
                    "mc_attack_norm",
                ],
                keys=[
                    "x_index",
                    "radius",
                    "smoother",
                    "cert_total",
                    "noisy_adv_prop",
                    "mc_adv_prop",
                    "gap_noisy",
                    "gap_mc",
                    "noisy_attack_norm",
                    "mc_attack_norm",
                ],
            )
            print(f"saved: {rows_json_path}")
            print(f"saved: {rows_csv_path}")
        else:
            payload = _load_json(rows_json_path)
            selected_indices = np.array(payload.get("config", {}).get("selected_indices", []), dtype=int)
            loaded_rows = payload.get("rows", [])
            if not isinstance(loaded_rows, list):
                raise ValueError(f"Invalid rows list in {rows_json_path}")
            all_rows = loaded_rows
            print(f"loaded: {rows_json_path}")

        summary_rows = _summarize_by_radius(all_rows, radii=radii)
        _save_json(summary_json_path, {"rows": summary_rows})
        _save_csv(
            summary_csv_path,
            rows=summary_rows,
            header=[
                "smoother",
                "radius",
                "num_inputs",
                "gap_mc_mean",
                "gap_mc_std",
                "noisy_adv_prop_mean",
                "noisy_adv_prop_std",
            ],
            keys=[
                "smoother",
                "radius",
                "num_inputs",
                "gap_mc_mean",
                "gap_mc_std",
                "noisy_adv_prop_mean",
                "noisy_adv_prop_std",
            ],
        )
        _plot_side_by_side_gaps(summary_rows=summary_rows, output_path=fig_path, alpha=alpha)

        print(f"saved: {summary_json_path}")
        print(f"saved: {summary_csv_path}")
        print(f"saved: {fig_path}")
        if selected_indices.size > 0:
            print(f"alpha={alpha:.3f} selected_indices: {selected_indices.tolist()}")
        _print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
