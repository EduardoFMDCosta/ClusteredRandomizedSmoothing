"""trajflow_safety.py — Decision-making safety analysis for TrajFlow predictions.

This experiment asks: if the ego vehicle uses the predicted probability that the
other vehicle *stays* to decide whether it is safe to *go*, how often does that
lead to a risky situation (both vehicles go at the same time)?

Alpha-smoothing (k=1) tends to collapse the bimodal stay/go distribution to a
single mode near the mean. In many L-gap scenarios this mean sits closer to the
stay region, causing the smoother to over-predict P(other stays) — which in turn
makes the ego vehicle falsely believe it is safe to proceed.

Clustered alpha-smoothing (k=2) explicitly separates the two modes and returns a
well-calibrated P(other stays) that leads to fewer risky decisions.

Analysis pipeline
-----------------
1.  For each selected scenario, run both smoothers and draw MC samples.
2.  Compute P(other stays) from each set of MC samples by counting the fraction
    of samples whose *endpoint displacement* ||y - y_init|| <= stay_radius.
3.  Determine the ego decision: go if P(other stays) >= threshold.
4.  Determine the GT label: other stays if ||gt_endpoint - initial|| <= stay_radius.
5.  Flag as risky if ego decides to go AND other actually goes.
6.  Aggregate and plot.
"""
from __future__ import annotations

import argparse
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
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from stochastic_functions import TrajFlowMeszarosStochasticFunction
from plotting import plot_safety_analysis
from experiment_configs import SAFETY as cfg
from trajflow_robustness import (
    _ask_num_indices,
    _ask_yes_no,
    _choose_indices,
    _extract_target_initial_position,
    _finite_target_mask,
    _load_inputs_and_outputs,
    _load_json,
    _load_or_create_fixed_random_order,
    _prepare_inputs_from_arrays,
    _save_json,
)


# ---------------------------------------------------------------------------
# Parameter builder
# ---------------------------------------------------------------------------

def _build_parameters(max_num_clusters: int) -> Parameters:
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
            max_num_clusters=2,
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


# ---------------------------------------------------------------------------
# P(stays) computation helpers
# ---------------------------------------------------------------------------

def _p_stay_from_mc(
    mc_samples: torch.Tensor,
    anchor: torch.Tensor,
    stay_radius: float,
) -> torch.Tensor:
    """Compute P(stay) from Monte Carlo samples as fraction with ||endpoint - anchor|| <= stay_radius.

    Args:
        mc_samples: Shape (nx, num_samples, output_dim). Last two dimensions are
            the final 2-D position prediction (endpoint).
        anchor: Shape (nx, 2). Initial position of the target vehicle.
        stay_radius: Scalar threshold in metres.

    Returns:
        Tensor of shape (nx,) with P(stay) in [0, 1].
    """
    # Extract final 2-D position from flattened output.
    endpoints = mc_samples[..., -2:]  # (nx, num_samples, 2)
    anchor_exp = anchor.to(device=mc_samples.device, dtype=mc_samples.dtype).unsqueeze(1)
    displacement = torch.linalg.norm(endpoints - anchor_exp, dim=-1)  # (nx, num_samples)
    return (displacement <= stay_radius).float().mean(dim=1)  # (nx,)


def _gt_stays_label(
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
    selected_indices: np.ndarray,
    stay_radius: float,
) -> np.ndarray:
    """Compute ground-truth stay/go labels for selected dataset indices.

    A scenario is labelled *stays* if the target vehicle's final GT position
    is within ``stay_radius`` metres of its initial position.

    Returns:
        Boolean array of shape (len(selected_indices),).
    """
    # Raw TrajFlow inputs: target vehicle is at agent slot 1 in the original data.
    target_initial = inputs[selected_indices, 1, 0, :2]    # (n, 2)
    target_final = outputs_gt[selected_indices, 0, -1, :]  # (n, 2)
    displacement = np.linalg.norm(target_final - target_initial, axis=1)
    return displacement <= stay_radius


# ---------------------------------------------------------------------------
# Per-batch processing
# ---------------------------------------------------------------------------

def _compute_batch_safety_rows(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    batch_indices: np.ndarray,
    inputs: np.ndarray,
    outputs_gt: np.ndarray,
) -> list[dict[str, Any]]:
    """Run both smoothers on one batch and collect P(other stays) for each input."""
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

    smoother_k1 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=_build_parameters(max_num_clusters=1),
        h=helper,
        cluster_anchor=None,
    )
    smoother_k2 = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=_build_parameters(max_num_clusters=2),
        h=helper,
        cluster_anchor=cluster_anchor,
    )

    mc_k1 = smoother_k1.monte_carlo(num_simulations=cfg.mc_samples)
    mc_k2 = smoother_k2.monte_carlo(num_simulations=cfg.mc_samples)

    p_stay_k1 = _p_stay_from_mc(mc_k1, cluster_anchor, stay_radius=cfg.behavior_stay_radius)
    p_stay_k2 = _p_stay_from_mc(mc_k2, cluster_anchor, stay_radius=cfg.behavior_stay_radius)

    # GT labels for this batch.
    gt_stays_batch = _gt_stays_label(
        inputs=inputs,
        outputs_gt=outputs_gt,
        selected_indices=batch_indices,
        stay_radius=cfg.behavior_stay_radius,
    )

    rows: list[dict[str, Any]] = []
    for i, dataset_idx in enumerate(batch_indices.tolist()):
        rows.append(
            {
                "dataset_index": int(dataset_idx),
                "p_stay_k1": float(p_stay_k1[i].item()),
                "p_stay_k2": float(p_stay_k2[i].item()),
                "gt_stays": bool(gt_stays_batch[i]),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """All smoothing/model parameters are in experiment_configs.py (SAFETY)."""
    parser = argparse.ArgumentParser(
        description=(
            "TrajFlow safety analysis — compare risk of ego-vehicle decisions\n"
            "under alpha-smoothing (k=1) vs clustered alpha-smoothing (k=2).\n"
            "Edit experiment_configs.SAFETY to change model/smoothing parameters."
        )
    )
    parser.add_argument("--num-inputs", type=int, default=0,
                        help="How many valid inputs to evaluate (0 = all). Ignored if --selected-indices is set.")
    parser.add_argument("--selected-indices", type=str, default="",
                        help="Comma-separated dataset indices. Overrides --num-inputs.")
    parser.add_argument("--regenerate", action=argparse.BooleanOptionalAction, default=None,
                        help="Re-run even when a saved JSON already exists.")
    parser.add_argument("--analysis-dir", default=str(EXPERIMENTS / "data" / "analysis" / "safety"))
    parser.add_argument("--figures-dir", default=str(EXPERIMENTS / "data" / "figures"))
    parser.add_argument("--figure-name", default="trajflow_safety_analysis.pdf")
    parser.add_argument("--fixed-index-order-file", default="")
    args = parser.parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    analysis_dir = Path(args.analysis_dir)
    json_path = analysis_dir / "trajflow_safety.json"

    do_regenerate: bool
    if args.regenerate is not None:
        do_regenerate = bool(args.regenerate)
    elif json_path.exists():
        do_regenerate = _ask_yes_no("Found existing safety data JSON.\nRegenerate? [y/n]: ")
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
            seed=cfg.seed,
        )

        num_inputs = args.num_inputs
        if num_inputs == 0 and not (args.selected_indices or "").strip():
            print(f"Available valid indices in dataset: {len(valid_indices)}")
            num_inputs = _ask_num_indices(
                "How many xs should be analyzed? (0 for all) ",
                len(valid_indices),
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
            batch_rows = _compute_batch_safety_rows(
                stochastic_fn=stochastic_fn,
                batch_indices=batch_indices,
                inputs=all_inputs,
                outputs_gt=outputs_gt,
            )
            all_rows.extend(batch_rows)
            print(f"batch {batch_id}/{num_batches} done in {time.perf_counter() - t0:.1f}s")

        _save_json(
            json_path,
            {
                "config": {
                    "selected_indices": selected_indices.tolist(),
                    "mc_samples": cfg.mc_samples,
                    "decision_threshold": cfg.decision_threshold,
                    "sigma": cfg.sigma,
                    "smoothing_N": cfg.smoothing_N,
                    "alpha": cfg.alpha,
                    "coverage": cfg.coverage,
                    "cluster_strategy": cfg.cluster_strategy,
                    "behavior_stay_radius": cfg.behavior_stay_radius,
                    "confidence_num_samples": cfg.confidence_num_samples,
                    "seed": cfg.seed,
                },
                "per_input": all_rows,
            },
        )
        print(f"saved: {json_path}")
    else:
        payload = _load_json(json_path)
        all_rows = payload.get("per_input", [])
        print(f"loaded: {json_path}")

    if len(all_rows) == 0:
        raise ValueError("No safety rows found to plot.")

    p_stay_k1 = np.array([float(r["p_stay_k1"]) for r in all_rows], dtype=float)
    p_stay_k2 = np.array([float(r["p_stay_k2"]) for r in all_rows], dtype=float)
    gt_stays = np.array([bool(r["gt_stays"]) for r in all_rows], dtype=bool)

    gt_goes = ~gt_stays
    risk_k1 = float(np.mean((p_stay_k1 >= cfg.decision_threshold) & gt_goes))
    risk_k2 = float(np.mean((p_stay_k2 >= cfg.decision_threshold) & gt_goes))

    print("\n=== TrajFlow Safety Analysis ===")
    print(f"N scenarios:           {len(all_rows)}")
    print(f"GT go rate:            {float(np.mean(gt_goes)):.3f}")
    print(f"Decision threshold:    {cfg.decision_threshold}")
    print(f"Risk rate  k=1:        {risk_k1:.3f}")
    print(f"Risk rate  k=2:        {risk_k2:.3f}")
    print(f"Mean P(stay)  k=1:     {float(np.mean(p_stay_k1)):.3f}")
    print(f"Mean P(stay)  k=2:     {float(np.mean(p_stay_k2)):.3f}")

    figure_path = Path(args.figures_dir) / args.figure_name
    plot_safety_analysis(
        p_stay_k1=p_stay_k1,
        p_stay_k2=p_stay_k2,
        gt_stays=gt_stays,
        threshold=cfg.decision_threshold,
        output_path=figure_path,
    )
    print(f"saved: {figure_path}")


if __name__ == "__main__":
    main()
