from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
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

from randomized_smoothing.dataclasses import (  # noqa: E402
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RuntimeHeuristics,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing  # noqa: E402
from stochastic_functions import TrajFlowMeszarosStochasticFunction  # noqa: E402
from plotting import plot_robustness_bounds_only  # noqa: E402
from experiment_configs import ROBUSTNESS as cfg  # noqa: E402


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
    ordered_unique: list[float] = []
    seen: set[float] = set()
    for alpha in alphas:
        if alpha not in seen:
            seen.add(alpha)
            ordered_unique.append(alpha)
    return ordered_unique


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


def _build_parameters(cfg_inst: Any, max_num_clusters: int) -> Parameters:
    if max_num_clusters == 1:
        clustering = ClusteringParameters(
            max_num_clusters=1,
            coverage=cfg_inst.coverage,
            strategy="dbscan",
            dbscan_eps=cfg_inst.dbscan_eps,
            dbscan_min_samples=cfg_inst.dbscan_min_samples,
        )
    else:
        clustering = ClusteringParameters(
            max_num_clusters=2,
            coverage=cfg_inst.coverage,
            strategy=cfg_inst.cluster_strategy,
            dbscan_eps=cfg_inst.dbscan_eps,
            dbscan_min_samples=cfg_inst.dbscan_min_samples,
            behavior_stay_radius=cfg_inst.behavior_stay_radius,
            behavior_min_down=cfg_inst.behavior_min_down,
            behavior_min_left=cfg_inst.behavior_min_left,
        )

    return Parameters(
        smoothing=SmoothingParameters(N=cfg_inst.smoothing_N, alpha=cfg_inst.alpha),
        clustering=clustering,
        confidence=ConfidenceParameters(num_samples=cfg_inst.confidence_num_samples, beta=1e-5),
        runtime=RuntimeHeuristics(noise_sub_batch_size=cfg_inst.noise_sub_batch_size),
    )


def _compute_batch_certification_rows(
    cfg_inst: Any,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    batch_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
    radii: list[float],
) -> list[dict[str, Any]]:
    """Compute certified probability lower bounds for k=1 and k=2 smoothers.

    For each (input, radius) pair returns the total certified probability from
    both smoothers.  No adversarial search is performed; this is a pure
    certification pass.
    """
    x_position = _prepare_inputs_from_arrays(
        selected_indices=batch_indices,
        inputs=inputs,
        outputs_gt=outputs_gt,
    )
    cluster_anchor = _extract_target_initial_position(x_position)

    if cfg_inst.input_mode == "position":
        model_input = x_position
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x_position)

    smoother_k1 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg_inst.sigma,
        parameters=_build_parameters(cfg_inst, max_num_clusters=1),
        h=helper,
        cluster_anchor=None,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg_inst.sigma,
        parameters=_build_parameters(cfg_inst, max_num_clusters=2),
        h=helper,
        cluster_anchor=cluster_anchor,
    )

    # Only keep inputs where the k=2 smoother actually found 2 clusters.
    has_two_clusters: np.ndarray = (smoother_k2.cluster.k_per_row == 2).cpu().numpy()

    rows: list[dict[str, Any]] = []
    for radius in radii:
        radius_tensor = torch.full(
            (model_input.shape[0], 1),
            fill_value=radius,
            dtype=model_input.dtype,
            device=model_input.device,
        )
        cert_k1 = smoother_k1.robust_certification(radius=radius_tensor).total_probs.detach().cpu().numpy()
        cert_k2 = smoother_k2.robust_certification(radius=radius_tensor).total_probs.detach().cpu().numpy()

        for i, dataset_idx in enumerate(batch_indices.tolist()):
            if not has_two_clusters[i]:
                continue
            rows.append(
                {
                    "x_index": int(dataset_idx),
                    "radius": float(radius),
                    "cert_k1": float(cert_k1[i]),
                    "cert_k2": float(cert_k2[i]),
                }
            )

    return rows


def main() -> None:
    """All smoothing/model parameters are in experiment_configs.py (ROBUSTNESS)."""
    parser = argparse.ArgumentParser(
        description=(
            "TrajFlow robustness bounds analysis — for each x_i and radius, compute\n"
            "certified probability lower bounds for k=1 and k=2 smoothers.\n"
            "Edit experiment_configs.ROBUSTNESS to change model/smoothing parameters."
        )
    )
    parser.add_argument("--num-inputs", type=int, default=0,
                        help="How many valid inputs to evaluate (0 = all). Ignored if --selected-indices is set.")
    parser.add_argument("--selected-indices", type=str, default="",
                        help="Comma-separated dataset indices. Overrides --num-inputs.")
    parser.add_argument("--regenerate", action=argparse.BooleanOptionalAction, default=None,
                        help="Re-run even when a saved JSON already exists.")
    parser.add_argument("--analysis-dir", default=str(EXPERIMENTS / "data" / "analysis" / "performance"))
    parser.add_argument("--figures-dir", default=str(EXPERIMENTS / "data" / "figures"))
    parser.add_argument("--plot-name", default="trajflow_robustness_bounds.pdf")
    parser.add_argument("--fixed-index-order-file", default="")
    args = parser.parse_args()

    radii = cfg.radius_values
    alpha_values = cfg.alpha_values

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    analysis_dir = Path(args.analysis_dir)
    figures_dir = Path(args.figures_dir)
    json_base_path = analysis_dir / "trajflow_robustness_bounds.json"
    json_paths = [_with_alpha_suffix(json_base_path, alpha) for alpha in alpha_values]

    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif any(p.exists() for p in json_paths):
        do_regenerate = _ask_yes_no("Found existing robustness data.\nRegenerate? [y/n]: ")
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
    else:
        available_alpha_values: list[float] = []
        missing_alpha_values: list[float] = []
        for alpha, json_path in zip(alpha_values, json_paths):
            if json_path.exists():
                available_alpha_values.append(float(alpha))
            else:
                missing_alpha_values.append(float(alpha))

        if not available_alpha_values:
            missing_text = ", ".join(f"{a:g}" for a in missing_alpha_values)
            raise FileNotFoundError(
                "--no-regenerate was requested, but no robustness JSON files were found. "
                f"Missing alphas: [{missing_text}]"
            )
        if missing_alpha_values:
            missing_text = ", ".join(f"{a:g}" for a in missing_alpha_values)
            print(f"Skipping alphas without JSON in --no-regenerate mode: [{missing_text}]")
        alpha_values = available_alpha_values

    # Collect per-alpha results into a dict for the bounds plot.
    bounds_by_alpha: dict[float, list[dict[str, Any]]] = {}

    for alpha in alpha_values:
        alpha_cfg = replace(cfg, alpha=float(alpha))
        json_path = _with_alpha_suffix(json_base_path, alpha)

        if do_regenerate:
            all_rows: list[dict[str, Any]] = []
            num_batches = (len(selected_indices) + cfg.input_batch_size - 1) // cfg.input_batch_size
            for batch_id, start in enumerate(range(0, len(selected_indices), cfg.input_batch_size), start=1):
                t0 = time.perf_counter()
                batch_indices = selected_indices[start : start + cfg.input_batch_size]
                print(
                    f"alpha={alpha:.3f}: processing batch {batch_id}/{num_batches} "
                    f"with {len(batch_indices)} dataset indices"
                )
                batch_rows = _compute_batch_certification_rows(
                    cfg_inst=alpha_cfg,
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
                "sigma": cfg.sigma,
                "smoothing_N": cfg.smoothing_N,
                "alpha": float(alpha),
                "coverage": cfg.coverage,
                "cluster_strategy": cfg.cluster_strategy,
                "behavior_stay_radius": cfg.behavior_stay_radius,
                "confidence_num_samples": cfg.confidence_num_samples,
                "input_batch_size": cfg.input_batch_size,
                "seed": cfg.seed,
            }
            _save_json(json_path, {"config": row_config, "rows": all_rows})
            print(f"alpha={alpha:.3f}: saved {json_path}")
        else:
            payload = _load_json(json_path)
            all_rows = payload.get("rows", [])
            print(f"alpha={alpha:.3f}: loaded {json_path}")

        # Aggregate by radius for the plot: compute per-input difference (k=1 − k=2).
        bounds_for_alpha: list[dict[str, Any]] = []
        for radius in radii:
            subset = [r for r in all_rows if float(r["radius"]) == float(radius)]
            if not subset:
                continue
            cert_k1_vals = np.array([float(r["cert_k1"]) for r in subset])
            cert_k2_vals = np.array([float(r["cert_k2"]) for r in subset])
            diff_vals = cert_k1_vals - cert_k2_vals
            bounds_for_alpha.append(
                {
                    "radius": float(radius),
                    "diff_mean": float(diff_vals.mean()),
                    "diff_std": float(diff_vals.std(ddof=0)),
                }
            )
        bounds_by_alpha[float(alpha)] = bounds_for_alpha

    fig_path = figures_dir / args.plot_name
    plot_robustness_bounds_only(bounds_by_alpha=bounds_by_alpha, output_path=fig_path)
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
