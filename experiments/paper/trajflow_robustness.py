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


def _ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter y or n.")


def _ask_nonnegative_float(prompt: str) -> float:
    while True:
        answer = input(prompt).strip()
        try:
            value = float(answer)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if value < 0.0:
            print("Please enter a non-negative number.")
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


def _behavior_probabilities(
    samples: torch.Tensor,
    anchor: torch.Tensor,
    behavior_stay_radius: float,
    behavior_min_down: float,
    behavior_min_left: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # samples: (nx, num_samples, 2), anchor: (nx, 2)
    delta = samples - anchor.unsqueeze(1)
    stay_mask = torch.linalg.norm(delta, dim=-1) <= behavior_stay_radius
    turn_mask = (
        (delta[..., 1] <= -behavior_min_down)
        & (delta[..., 0] <= -behavior_min_left)
        & (~stay_mask)
    )
    stay_prob = stay_mask.to(dtype=samples.dtype).mean(dim=1)
    turn_prob = turn_mask.to(dtype=samples.dtype).mean(dim=1)
    return stay_prob, turn_prob


def _find_variance_maximizing_delta(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    model_input: torch.Tensor,
    helper: list[torch.Tensor],
    cluster_anchor: torch.Tensor,
    sigma: float,
    radius: float,
    num_candidates: int,
    num_samples: int,
    attack_objective: str,
    behavior_stay_radius: float,
    behavior_min_down: float,
    behavior_min_left: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if radius <= 0.0:
        return torch.zeros_like(model_input), torch.zeros(
            (model_input.shape[0],),
            dtype=model_input.dtype,
            device=model_input.device,
        )
    if num_candidates <= 0:
        raise ValueError("attack-num-candidates must be positive")
    if num_samples <= 0:
        raise ValueError("attack-num-samples must be positive")
    if attack_objective not in ("variance", "behavior_flip"):
        raise ValueError(f"Unsupported attack objective: {attack_objective}")

    dimension = model_input.shape[-1]
    base_noise = MultivariateGaussian(
        mean=torch.zeros(dimension, dtype=model_input.dtype, device=model_input.device),
        cov=(sigma ** 2) * torch.eye(dimension, dtype=model_input.dtype, device=model_input.device),
    )

    delta = torch.zeros_like(model_input)
    best_scores = torch.empty((model_input.shape[0],), dtype=model_input.dtype, device=model_input.device)
    for i in range(model_input.shape[0]):
        x_i = model_input[i: i + 1]
        h_i = _slice_helper(helper, i)
        anchor_i = cluster_anchor[i: i + 1].to(device=model_input.device, dtype=model_input.dtype)

        target_is_turn: bool | None = None
        if attack_objective == "behavior_flip":
            clean_samples = stochastic_fn.sample_from_noisy_inputs(
                x=x_i,
                smoothing_distribution=base_noise,
                num_samples=num_samples,
                h=h_i,
            )
            clean_stay_prob, clean_turn_prob = _behavior_probabilities(
                samples=clean_samples,
                anchor=anchor_i,
                behavior_stay_radius=behavior_stay_radius,
                behavior_min_down=behavior_min_down,
                behavior_min_left=behavior_min_left,
            )
            # Flip away from dominant clean behavior.
            target_is_turn = bool(clean_turn_prob[0].item() < clean_stay_prob[0].item())

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
            if attack_objective == "variance":
                # Score = trace of empirical covariance of noisy outputs.
                score = float(samples.var(dim=1, unbiased=False).sum().item())
            else:
                adv_stay_prob, adv_turn_prob = _behavior_probabilities(
                    samples=samples,
                    anchor=anchor_i,
                    behavior_stay_radius=behavior_stay_radius,
                    behavior_min_down=behavior_min_down,
                    behavior_min_left=behavior_min_left,
                )
                if target_is_turn:
                    score = float(adv_turn_prob[0].item())
                else:
                    score = float(adv_stay_prob[0].item())
            if score > best_score:
                best_score = score
                best_delta = delta_c.squeeze(0)

        delta[i] = best_delta
        best_scores[i] = best_score

    return delta, best_scores


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


def _radius_suffix(radius: float) -> str:
    # Keep filenames portable and compact, e.g. r0p1, r1p25, r0.
    text = f"{radius:.10g}".replace("-", "m").replace(".", "p")
    return f"r{text}"


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


def _save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    header: list[str],
    keys: list[str],
    title_row: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if title_row is not None:
            writer.writerow(title_row)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_format_cell(row[k]) for k in keys])


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.2f}"


def _format_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return _format_float(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _print_table(
    rows: list[dict[str, Any]],
    header: list[str],
    keys: list[str],
    title_row: list[str] | None = None,
) -> None:
    if not rows:
        print("No rows to print.")
        return

    body = []
    for row in rows:
        values = [_format_cell(row[key]) for key in keys]
        body.append(values)

    widths = [
        max(
            len(header[i]),
            len(title_row[i]) if title_row is not None and i < len(title_row) else 0,
            *(len(r[i]) for r in body),
        )
        for i in range(len(header))
    ]

    def _fmt(values: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    if title_row is not None:
        print(_fmt(title_row))
    print(_fmt(header))
    print("-+-".join("-" * w for w in widths))
    for values in body:
        print(_fmt(values))


def _merged_table_schema() -> tuple[list[str], list[str], list[str], list[str]]:
    title_row = ["", "", r"Clustered $\alpha$-smoothing", "", "", r"$\alpha$-smoothing"]
    header = [
        r"Index $i$",
        "Statistic",
        r"$\mathcal{R}_1$",
        r"$\mathcal{R}_2$",
        r"$\mathcal{R} = \mathcal{R}_1 \cup \mathcal{R}_2$",
        r"$\mathcal{R}$",
    ]
    keys = [
        "x_index",
        "quantity",
        "region_1",
        "region_2",
        "region_union",
        "region_alpha",
    ]
    return title_row, header, keys, [
        r"Proportion $h_w(x+\varepsilon) \in \mathcal{R}_j$",
        "Area",
        "Bounds Thm 3.1.",
        r"Proportion $H(x) \in \mathcal{R}_j$",
    ]


def _merged_table_rows(per_x_rows: list[dict[str, Any]], quantity_order: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for per_x in per_x_rows:
        x_index = int(per_x["x_index"])
        rows.extend(
            [
                {
                    "x_index": x_index,
                    "quantity": quantity_order[0],
                    "region_1": per_x["set_containment_region_1_k2"],
                    "region_2": per_x["set_containment_region_2_k2"],
                    "region_union": per_x["set_containment_union_k2"],
                    "region_alpha": per_x["set_containment_k1"],
                },
                {
                    "x_index": x_index,
                    "quantity": quantity_order[1],
                    "region_1": per_x["area_region_1_k2"],
                    "region_2": per_x["area_region_2_k2"],
                    "region_union": per_x["area_union_k2"],
                    "region_alpha": per_x["area_k1"],
                },
                {
                    "x_index": x_index,
                    "quantity": quantity_order[2],
                    "region_1": per_x["bound_region_1_k2"],
                    "region_2": per_x["bound_region_2_k2"],
                    "region_union": per_x["bound_union_k2"],
                    "region_alpha": per_x["bound_k1"],
                },
                {
                    "x_index": x_index,
                    "quantity": quantity_order[3],
                    "region_1": per_x["mc_containment_region_1_k2"],
                    "region_2": per_x["mc_containment_region_2_k2"],
                    "region_union": per_x["mc_containment_union_k2"],
                    "region_alpha": per_x["mc_containment_k1"],
                },
            ]
        )
    return rows


def _load_per_x_rows_from_json(json_path: Path) -> list[dict[str, Any]]:
    payload = _load_json(json_path)
    rows = payload.get("rows", None)
    if not isinstance(rows, list):
        raise ValueError(f"Expected 'rows' list in {json_path}")
    # The JSON stores merged display rows (4 per x_index). Reconstruct per-x dicts.
    quantity_labels = [
        r"Proportion $h_w(x+\varepsilon) \in \mathcal{R}_j$",
        "Area",
        "Bounds Thm 3.1.",
        r"Proportion $H(x) \in \mathcal{R}_j$",
    ]
    key_map = {
        quantity_labels[0]: (
            "set_containment_region_1_k2", "set_containment_region_2_k2",
            "set_containment_union_k2", "set_containment_k1",
        ),
        quantity_labels[1]: (
            "area_region_1_k2", "area_region_2_k2",
            "area_union_k2", "area_k1",
        ),
        quantity_labels[2]: (
            "bound_region_1_k2", "bound_region_2_k2",
            "bound_union_k2", "bound_k1",
        ),
        quantity_labels[3]: (
            "mc_containment_region_1_k2", "mc_containment_region_2_k2",
            "mc_containment_union_k2", "mc_containment_k1",
        ),
    }
    # Group rows by x_index, preserving order.
    per_x: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for r in rows:
        xi = int(r["x_index"])
        if xi not in per_x:
            per_x[xi] = {"x_index": xi}
            order.append(xi)
        qty = r.get("quantity", "")
        if qty in key_map:
            k1, k2, ku, ka = key_map[qty]
            def _to_float(v: Any) -> float:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            per_x[xi][k1] = _to_float(r.get("region_1", float("nan")))
            per_x[xi][k2] = _to_float(r.get("region_2", float("nan")))
            per_x[xi][ku] = _to_float(r.get("region_union", float("nan")))
            per_x[xi][ka] = _to_float(r.get("region_alpha", float("nan")))
    return [per_x[xi] for xi in order]


def _compute_statistics_table(
    per_x_rows: list[dict[str, Any]],
    mc_samples: int = 1000,
) -> list[dict[str, Any]]:
    n = len(per_x_rows)
    if n == 0:
        return []

    # ── Row 1: proportion of x where union containment is strictly higher ──
    clustered_wins = [
        r["set_containment_union_k2"] > r["set_containment_k1"]
        for r in per_x_rows
    ]
    alpha_wins = [
        r["set_containment_k1"] > r["set_containment_union_k2"]
        for r in per_x_rows
    ]
    prop_clustered_wins = float(np.mean(clustered_wins))
    prop_alpha_wins = float(np.mean(alpha_wins))

    # ── Row 2: average margin when each method wins ──
    clustered_margins = [
        r["set_containment_union_k2"] - r["set_containment_k1"]
        for r, w in zip(per_x_rows, clustered_wins) if w
    ]
    alpha_margins = [
        r["set_containment_k1"] - r["set_containment_union_k2"]
        for r, w in zip(per_x_rows, alpha_wins) if w
    ]
    avg_margin_clustered = float(np.mean(clustered_margins)) if clustered_margins else float("nan")
    avg_margin_alpha = float(np.mean(alpha_margins)) if alpha_margins else float("nan")

    # ── Row 3: average total area ──
    avg_area_clustered = float(np.mean([r["area_union_k2"] for r in per_x_rows]))
    avg_area_alpha = float(np.mean([r["area_k1"] for r in per_x_rows]))

    # ── Row 4: average minimum area (over individual regions) ──
    def _min_area_k2(r: dict[str, Any]) -> float:
        a1 = r["area_region_1_k2"]
        a2 = r["area_region_2_k2"]
        candidates = [v for v in (a1, a2) if not np.isnan(v)]
        return float(min(candidates)) if candidates else float("nan")

    avg_min_area_clustered = float(np.nanmean([_min_area_k2(r) for r in per_x_rows]))
    # Alpha-smoothing has only one region, so min == total area.
    avg_min_area_alpha = float(np.mean([r["area_k1"] for r in per_x_rows]))

    # ── Row 5: proportion of (x, region) pairs where bound <= mc_containment ──
    # Skip pairs where the bound is below 1/mc_samples: these are unverifiable
    # because the expected number of MC hits is < 1, so the MC estimate is
    # almost surely 0 regardless of whether the bound truly holds.
    min_observable = 1.0 / mc_samples
    # Clustered: check region 1 and region 2 independently (skip NaN bounds).
    k2_checks: list[bool] = []
    for r in per_x_rows:
        for bk, mck in [
            ("bound_region_1_k2", "mc_containment_region_1_k2"),
            ("bound_region_2_k2", "mc_containment_region_2_k2"),
        ]:
            b, mc = r[bk], r[mck]
            if np.isnan(b) or np.isnan(mc):
                continue
            if b < min_observable:
                continue  # too small to verify with MC
            k2_checks.append(bool(b <= mc))
    prop_bound_valid_clustered = float(np.mean(k2_checks)) if k2_checks else float("nan")
    # Alpha: single region (same filter).
    k1_checks = [
        bool(r["bound_k1"] <= r["mc_containment_k1"])
        for r in per_x_rows
        if r["bound_k1"] >= min_observable
    ]
    prop_bound_valid_alpha = float(np.mean(k1_checks)) if k1_checks else float("nan")

    stat_rows = [
        {
            "statistic": r"Proportion $h_w(x+\varepsilon) \in \mathcal{R}$ clustered wins",
            "clustered": prop_clustered_wins,
            "alpha": prop_alpha_wins,
        },
        {
            "statistic": "Avg margin (containment) when winning",
            "clustered": avg_margin_clustered,
            "alpha": avg_margin_alpha,
        },
        {
            "statistic": "Avg total area",
            "clustered": avg_area_clustered,
            "alpha": avg_area_alpha,
        },
        {
            "statistic": "Avg minimum area (per region)",
            "clustered": avg_min_area_clustered,
            "alpha": avg_min_area_alpha,
        },
        {
            "statistic": r"Proportion Bound Thm 3.1. $\leq$ Proportion $H(x) \in \mathcal{R}$ (per region)",
            "clustered": prop_bound_valid_clustered,
            "alpha": prop_bound_valid_alpha,
        },
    ]
    return stat_rows


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
    attack_scores = torch.zeros(
        (model_input.shape[0],),
        dtype=model_input.dtype,
        device=model_input.device,
    )
    if args.certification_radius > 0.0:
        delta, attack_scores = _find_variance_maximizing_delta(
            stochastic_fn=stochastic_fn,
            model_input=model_input,
            helper=helper,
            cluster_anchor=cluster_anchor,
            sigma=args.sigma,
            radius=args.certification_radius,
            num_candidates=args.attack_num_candidates,
            num_samples=args.attack_num_samples,
            attack_objective=args.attack_objective,
            behavior_stay_radius=args.behavior_stay_radius,
            behavior_min_down=args.behavior_min_down,
            behavior_min_left=args.behavior_min_left,
        )
        if args.attack_objective == "behavior_flip":
            print(
                "attack behavior-flip score "
                f"mean={attack_scores.mean().item():.3f}, "
                f"min={attack_scores.min().item():.3f}, "
                f"max={attack_scores.max().item():.3f}"
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
    noisy_region_inside_k2 = smoother_k2.regions.contains(noisy_samples)
    noisy_inside_k2 = noisy_region_inside_k2.any(dim=-1)
    noisy_inside_k1 = smoother_k1.regions.contains(noisy_samples).any(dim=-1)
    containment_region_k2 = noisy_region_inside_k2.float().mean(dim=1)
    containment_k1 = noisy_inside_k1.float().mean(dim=1)

    mc_samples_k2 = smoother_k2.monte_carlo(num_simulations=args.mc_containment_samples, z=model_input_eval)
    mc_samples_k1 = smoother_k1.monte_carlo(num_simulations=args.mc_containment_samples, z=model_input_eval)
    mc_region_inside_k2 = smoother_k2.regions.contains(mc_samples_k2)
    mc_inside_k2 = mc_region_inside_k2.any(dim=-1)
    mc_inside_k1 = smoother_k1.regions.contains(mc_samples_k1).any(dim=-1)
    mc_containment_region_k2 = mc_region_inside_k2.float().mean(dim=1)
    mc_containment_k2 = mc_inside_k2.float().mean(dim=1)
    mc_containment_k1 = mc_inside_k1.float().mean(dim=1)

    cert_k2_probs = cert_k2.probs.detach().cpu()
    cert_k2_total = cert_k2.total_probs.detach().cpu()
    cert_k1_total = cert_k1.total_probs.detach().cpu()
    k2_used = smoother_k2.cluster.k_per_row.detach().cpu().to(torch.int64)
    k1_used = smoother_k1.cluster.k_per_row.detach().cpu().to(torch.int64)

    k2_volume = smoother_k2.regions.volume.detach().cpu().to(torch.float64)
    k1_volume = smoother_k1.regions.volume.detach().cpu().to(torch.float64)
    k2_area_region_1 = torch.full((k2_volume.shape[0],), float("nan"), dtype=torch.float64)
    k2_area_region_2 = torch.full((k2_volume.shape[0],), float("nan"), dtype=torch.float64)
    k2_area_sum = torch.zeros(k2_volume.shape[0], dtype=torch.float64)
    k1_total_area = torch.zeros(k1_volume.shape[0], dtype=torch.float64)
    for i in range(k2_volume.shape[0]):
        used = int(k2_used[i].item())
        k2_area_sum[i] = k2_volume[i, :used].sum()
        if used >= 1 and k2_volume.shape[1] >= 1:
            k2_area_region_1[i] = k2_volume[i, 0]
        if used >= 2 and k2_volume.shape[1] >= 2:
            k2_area_region_2[i] = k2_volume[i, 1]
        k1_total_area[i] = k1_volume[i, : int(k1_used[i].item())].sum()

    rows: list[dict[str, Any]] = []
    containment_region_k2_cpu = containment_region_k2.detach().cpu().numpy()
    containment_k1_cpu = containment_k1.detach().cpu().numpy()
    mc_containment_region_k2_cpu = mc_containment_region_k2.detach().cpu().numpy()
    mc_containment_k2_cpu = mc_containment_k2.detach().cpu().numpy()
    mc_containment_k1_cpu = mc_containment_k1.detach().cpu().numpy()
    k2_area_region_1_cpu = k2_area_region_1.numpy()
    k2_area_region_2_cpu = k2_area_region_2.numpy()
    k2_area_sum_cpu = k2_area_sum.numpy()
    k1_total_area_cpu = k1_total_area.numpy()

    for i, dataset_idx in enumerate(batch_indices.tolist()):
        used = int(k2_used[i].item())
        contain_r1 = float("nan")
        contain_r2 = float("nan")
        contain_union = float(noisy_inside_k2[i].float().mean().item())
        mc_contain_r1 = float("nan")
        mc_contain_r2 = float("nan")
        if used >= 1 and containment_region_k2_cpu.shape[1] >= 1:
            contain_r1 = float(containment_region_k2_cpu[i, 0])
            mc_contain_r1 = float(mc_containment_region_k2_cpu[i, 0])
        if used >= 2 and containment_region_k2_cpu.shape[1] >= 2:
            contain_r2 = float(containment_region_k2_cpu[i, 1])
            mc_contain_r2 = float(mc_containment_region_k2_cpu[i, 1])

        row: dict[str, Any] = {
            "x_index": int(dataset_idx),
            "set_containment_region_1_k2": contain_r1,
            "set_containment_region_2_k2": contain_r2,
            "set_containment_union_k2": contain_union,
            "area_region_1_k2": float(k2_area_region_1_cpu[i]),
            "area_region_2_k2": float(k2_area_region_2_cpu[i]),
            "area_union_k2": float(k2_area_sum_cpu[i]),
            "bound_region_1_k2": float("nan"),
            "bound_region_2_k2": float("nan"),
            "bound_union_k2": float(cert_k2_total[i].item()),
            "mc_containment_region_1_k2": mc_contain_r1,
            "mc_containment_region_2_k2": mc_contain_r2,
            "mc_containment_union_k2": float(mc_containment_k2_cpu[i]),
            "set_containment_k1": float(containment_k1_cpu[i]),
            "area_k1": float(k1_total_area_cpu[i]),
            "bound_k1": float(cert_k1_total[i].item()),
            "mc_containment_k1": float(mc_containment_k1_cpu[i]),
        }
        if used >= 1 and cert_k2_probs.shape[1] >= 1:
            row["bound_region_1_k2"] = float(cert_k2_probs[i, 0].item())
        if used >= 2 and cert_k2_probs.shape[1] >= 2:
            row["bound_region_2_k2"] = float(cert_k2_probs[i, 1].item())
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-input TrajFlow containment and guarantees tables "
            "for k=1 and k=2 smoothers."
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
        "--attack-objective",
        type=str,
        choices=("variance", "behavior_flip"),
        default="behavior_flip",
        help=(
            "Attack objective. 'variance' maximizes output variance; "
            "'behavior_flip' maximizes probability of flipping stay/turn behavior."
        ),
    )
    parser.add_argument(
        "--certification-radius",
        type=float,
        default=None,
        help=(
            "Input perturbation radius used in robust certification. "
            "If omitted, the script asks interactively."
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
    parser.add_argument(
        "--regenerate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If set, recomputes data even when a saved JSON already exists. "
            "If --no-regenerate is set, loads from the existing JSON without recomputing. "
            "If omitted, the script asks interactively when the JSON already exists."
        ),
    )
    args = parser.parse_args()

    if args.certification_radius is None:
        args.certification_radius = _ask_nonnegative_float(
            "Certification radius for robustness/attack (>= 0): "
        )

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

    analysis_dir = Path(args.analysis_dir)
    if args.certification_radius is None:
        args.certification_radius = _ask_nonnegative_float(
            "Certification radius for robustness/attack (>= 0): "
        )

    if args.num_inputs is None and not (args.selected_indices or "").strip():
        n_available = int(np.load(EXPERIMENTS / "TrajFlowExperiment" / "Data" / "Outputs_gt.npy").shape[0])
        print(f"Available inputs in dataset: {n_available}")
        args.num_inputs = _ask_num_indices(
            "How many xs should be analyzed? (0 for all) ",
            n_available,
        )
    if args.num_inputs is None:
        args.num_inputs = 0
    radius_tag = _radius_suffix(args.certification_radius)
    merged_json_path = analysis_dir / f"robustness_table_{radius_tag}.json"

    # --- Decide whether to regenerate or load from existing JSON ---
    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif merged_json_path.exists():
        do_regenerate = _ask_yes_no(
            f"Found existing data at {merged_json_path}.\n"
            "Regenerate? [y/n]: "
        )
    else:
        do_regenerate = True

    merged_csv_path = analysis_dir / f"robustness_table_{radius_tag}.csv"
    stats_json_path = analysis_dir / f"statistics_data_{radius_tag}.json"
    stats_csv_path = analysis_dir / f"statistics_data_{radius_tag}.csv"

    merged_title_row, merged_header, merged_keys, quantity_order = _merged_table_schema()

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

        if args.num_inputs is None:
            args.num_inputs = 0

        selected_indices = _choose_indices(
            fixed_random_order=fixed_random_order,
            num_inputs=args.num_inputs,
            selected_indices_text=args.selected_indices,
        )

        per_x_rows: list[dict[str, Any]] = []
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
            per_x_rows.extend(rows)
            print(f"batch {batch_id}/{num_batches} completed in {time.perf_counter() - batch_t0:.1f}s")

        merged_rows = _merged_table_rows(per_x_rows, quantity_order=quantity_order)

        base_config = {
            "selected_indices": selected_indices.tolist(),
            "containment_samples": args.containment_samples,
            "mc_containment_samples": args.mc_containment_samples,
            "attack_num_candidates": args.attack_num_candidates,
            "attack_num_samples": args.attack_num_samples,
            "attack_objective": args.attack_objective,
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
        }
        merged_payload = {
            "config": base_config,
            "title_row": merged_title_row,
            "columns": merged_keys,
            "rows": merged_rows,
        }
        _save_json(merged_json_path, merged_payload)
        _save_csv(
            merged_csv_path,
            merged_rows,
            header=merged_header,
            keys=merged_keys,
            title_row=merged_title_row,
        )
        print("=== TrajFlow Robustness Table ===")
        print(f"selected_indices: {selected_indices.tolist()}")
        print(f"saved: {merged_json_path}")
        print(f"saved: {merged_csv_path}")
        _print_table(
            merged_rows,
            header=merged_header,
            keys=merged_keys,
            title_row=merged_title_row,
        )
    else:
        print(f"Loading data from {merged_json_path}")
        per_x_rows = _load_per_x_rows_from_json(merged_json_path)

    # --- Statistics table (always computed from per_x_rows) ---
    stat_rows = _compute_statistics_table(per_x_rows, mc_samples=args.mc_containment_samples)
    stats_title_row = [
        "",
        r"Clustered $\alpha$-smoothing",
        r"$\alpha$-smoothing",
    ]
    stats_header = ["Statistic", "Clustered", "Alpha"]
    stats_keys = ["statistic", "clustered", "alpha"]

    _save_json(stats_json_path, {"rows": stat_rows})
    _save_csv(
        stats_csv_path,
        stat_rows,
        header=stats_header,
        keys=stats_keys,
        title_row=stats_title_row,
    )
    print("=== TrajFlow Statistics Table ===")
    print(f"saved: {stats_json_path}")
    print(f"saved: {stats_csv_path}")
    _print_table(
        stat_rows,
        header=stats_header,
        keys=stats_keys,
        title_row=stats_title_row,
    )


if __name__ == "__main__":
    main()
