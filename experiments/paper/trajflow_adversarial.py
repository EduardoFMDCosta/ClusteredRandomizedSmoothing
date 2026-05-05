from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import ot
except ImportError as exc:  # pragma: no cover - import guard for optional runtime dependency
    raise ImportError(
        "POT is required for trajflow_adversarial. Install with 'pip install pot'."
    ) from exc


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
from trajflow_robustness import (
    _ask_num_indices,
    _choose_indices,
    _extract_target_initial_position,
    _finite_target_mask,
    _load_inputs_and_outputs,
    _load_or_create_fixed_random_order,
    _prepare_inputs_from_arrays,
)


def _parse_radius_values(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("radius-values must contain at least one value")
    radii = [float(v) for v in values]
    if any(r < 0.0 for r in radii):
        raise ValueError("all radius values must be non-negative")
    return radii


def _compute_sinkhorn_wasserstein_2_distance(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    reg: float,
    ot_method: str,
) -> float:
    """Compute empirical 2-Wasserstein distance using a POT solver.
    
    Uses sample-based approximation with stable entropic regularization methods.
    Aligns with trajflow_wasserstein reference implementation.
    """
    if x_samples.ndim != 2 or y_samples.ndim != 2:
        raise ValueError("Both sample sets must be 2-D arrays of shape (n, d).")
    if x_samples.shape[1] != y_samples.shape[1]:
        raise ValueError(
            "Both sample sets must have the same feature dimension; "
            f"got {x_samples.shape[1]} and {y_samples.shape[1]}"
        )
    if reg <= 0.0 and ot_method != "emd":
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
    if radius == 0.0:
        return torch.zeros((1, dimension), device=device, dtype=dtype)
    if num_points <= 1:
        return torch.zeros((1, dimension), device=device, dtype=dtype)

    # Include the clean point (delta=0) and sample the remaining candidates
    # uniformly in the closed ball to search over ||delta|| <= radius.
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


def _find_best_attack_for_smoother(
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    smoother: GaussianAlphaTrimmingSmoothing,
    x_single: torch.Tensor,
    helper_single: list[torch.Tensor],
    sigma: float,
    radius: float,
    num_candidates: int,
    num_distribution_samples: int,
    attack_distribution_samples: int,
    sinkhorn_reg: float,
    ot_method: str,
    noise_sub_batch_size: int,
    attack_search_strategy: str,
    attack_screening_samples: int,
    attack_topk_w2: int,
    warm_start_delta: torch.Tensor | None = None,
    noisy_reference_samples: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Find the attack delta on the ball ||delta||<=radius maximizing W2.

    Objective: max_{||delta||<=radius} W2(h_w(x+eps), smoother(x+delta))

    The reference distribution h_w(x+eps) is fixed at the clean input x and
    sampled once. Only smoother.monte_carlo(z=x+delta) varies per candidate,
    so noisy sampling cost is paid exactly once regardless of candidate count.
    """
    dimension = x_single.shape[-1]
    base_noise = MultivariateGaussian(
        mean=torch.zeros(dimension, dtype=x_single.dtype, device=x_single.device),
        cov=(sigma**2) * torch.eye(dimension, dtype=x_single.dtype, device=x_single.device),
    )

    candidates = _random_ball_points(
        num_points=max(num_candidates, 1),
        dimension=dimension,
        radius=radius,
        device=x_single.device,
        dtype=x_single.dtype,
    )
    if warm_start_delta is not None:
        warm = warm_start_delta.to(device=x_single.device, dtype=x_single.dtype).view(1, dimension)
        if float(torch.linalg.norm(warm).item()) <= radius + 1e-12:
            candidates = torch.cat((candidates, warm), dim=0)

    # Reference distribution h_w(x+eps) can be provided externally so the same
    # samples are reused across radii; this reduces objective noise and makes
    # radius trends more stable.
    if noisy_reference_samples is None:
        noisy_full = stochastic_fn.sample_from_noisy_inputs(
            x=x_single,
            smoothing_distribution=base_noise,
            num_samples=num_distribution_samples,
            h=helper_single,
            noise_sub_batch_size=noise_sub_batch_size,
        )
        noisy_ref = noisy_full[0]
    else:
        noisy_ref = noisy_reference_samples

    if noisy_ref.shape[0] < num_distribution_samples:
        raise ValueError(
            "noisy_reference_samples must contain at least num_distribution_samples rows"
        )

    noisy_np_full = noisy_ref[:num_distribution_samples].detach().cpu().numpy()

    # Precompute noisy moments for the cheap screening proxy.
    n_screen = max(1, min(attack_screening_samples, num_distribution_samples))
    noisy_screen = noisy_ref[:n_screen]
    noisy_screen_mean = noisy_screen.mean(dim=0)
    noisy_screen_var = noisy_screen.var(dim=0, unbiased=False)

    def _w2_against_noisy(x_adv: torch.Tensor, n_samples: int) -> float:
        smooth_np = smoother.monte_carlo(num_simulations=n_samples, z=x_adv)[0].detach().cpu().numpy()
        return _compute_sinkhorn_wasserstein_2_distance(
            noisy_np_full[:n_samples], smooth_np, reg=sinkhorn_reg, ot_method=ot_method,
        )

    def _screen_score(x_adv: torch.Tensor) -> float:
        smooth = smoother.monte_carlo(num_simulations=n_screen, z=x_adv)[0]
        mean_gap = torch.linalg.norm(noisy_screen_mean - smooth.mean(dim=0), ord=2)
        var_gap = torch.linalg.norm(noisy_screen_var - smooth.var(dim=0, unbiased=False), ord=2)
        return float((mean_gap + 0.25 * var_gap).item())

    if attack_search_strategy == "full" or candidates.shape[0] == 1:
        candidate_indices = list(range(candidates.shape[0]))
    else:
        proxy_scores = [_screen_score(x_single + candidates[c : c + 1]) for c in range(candidates.shape[0])]
        topk = max(1, min(attack_topk_w2, candidates.shape[0]))
        candidate_indices = [
            int(i) for i in np.argsort(np.asarray(proxy_scores, dtype=np.float64))[-topk:]
        ]

    search_samples = max(1, min(attack_distribution_samples, num_distribution_samples))
    best_w2_search = float("-inf")
    best_delta = candidates[0].clone()
    for c in candidate_indices:
        w2 = _w2_against_noisy(x_single + candidates[c : c + 1], search_samples)
        if w2 > best_w2_search:
            best_w2_search = w2
            best_delta = candidates[c].clone()

    # Re-evaluate the winner with the full reporting budget (uses all of noisy_np_full).
    final_w2 = _w2_against_noisy(x_single + best_delta.unsqueeze(0), num_distribution_samples)
    return best_delta, float(final_w2)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _plot_wasserstein_certification_tradeoff(
    radius_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not radius_rows:
        raise ValueError("radius_rows must be non-empty")

    cert_k1 = np.array([float(r["mean_certified_prob_k1"]) for r in radius_rows], dtype=float)
    cert_k2 = np.array([float(r["mean_certified_prob_k2"]) for r in radius_rows], dtype=float)
    w2_k1 = np.array([float(r["mean_max_w2_k1"]) for r in radius_rows], dtype=float)
    w2_k2 = np.array([float(r["mean_max_w2_k2"]) for r in radius_rows], dtype=float)
    
    std_w2_k1 = np.array([float(r["std_max_w2_k1"]) for r in radius_rows], dtype=float)
    std_w2_k2 = np.array([float(r["std_max_w2_k2"]) for r in radius_rows], dtype=float)

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
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        
        # Plot k1 smoother with uncertainty band
        ax.fill_between(
            cert_k1,
            w2_k1 - std_w2_k1,
            w2_k1 + std_w2_k1,
            alpha=0.2,
            color="#4C78A8",
        )
        ax.plot(
            cert_k1,
            w2_k1,
            marker="o",
            linewidth=1.4,
            color="#4C78A8",
            label=r"$\alpha$-smoothing",
        )
        
        # Plot k2 smoother with uncertainty band
        ax.fill_between(
            cert_k2,
            w2_k2 - std_w2_k2,
            w2_k2 + std_w2_k2,
            alpha=0.2,
            color="#2A9D8F",
        )
        ax.plot(
            cert_k2,
            w2_k2,
            marker="s",
            linewidth=1.4,
            color="#2A9D8F",
            label=r"Clustered $\alpha$-smoothing",
        )

        ax.set_xlabel(r"Mean certified lower bound probability")
        ax.set_ylabel(r"Mean Wasserstein")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adversarial TrajFlow experiment: for each radius and each x_i, find specialized "
            "attacks maximizing Wasserstein distance between attacked noisy outputs and "
            "attacked-center smoothers, then compare to certified lower bounds."
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
        default="0.0, 0.1, 0.3, 0.5",
        help="Comma-separated attack radii.",
    )
    parser.add_argument(
        "--attack-num-candidates",
        type=int,
        default=64,
        help="Random points sampled in the attack ball ||delta||<=radius for each smoother and input.",
    )
    parser.add_argument(
        "--attack-search-strategy",
        type=str,
        choices=("screened", "full"),
        default="screened",
        help=(
            "Attack search strategy. 'screened' ranks all candidates using a cheap "
            "moment-mismatch proxy and computes exact Wasserstein only for top-k; "
            "'full' computes exact Wasserstein for all candidates."
        ),
    )
    parser.add_argument(
        "--attack-screening-samples",
        type=int,
        default=14,
        help="Samples per candidate used by the cheap screening proxy in screened mode.",
    )
    parser.add_argument(
        "--attack-topk-w2",
        type=int,
        default=8,
        help="Number of top-ranked candidates promoted to exact Wasserstein evaluation.",
    )
    parser.add_argument(
        "--attack-distribution-samples",
        type=int,
        default=60,
        help=(
            "Samples used to estimate W2 during the attack candidate ranking step. "
            "Smaller than --distribution-samples to keep the search fast. "
            "The winning delta is always re-evaluated with the full --distribution-samples."
        ),
    )
    parser.add_argument(
        "--distribution-samples",
        type=int,
        default=300,
        help="Distribution samples used for the final reported Wasserstein (winner re-evaluation).",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=("control", "position"),
        default="control",
        help="Input representation for smoothing and perturbations.",
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
        default=0.35,
        help="Trimming parameter alpha.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.90,
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
        default=35,
        help="DBSCAN min_samples for clustering inside smoothing.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=2000,
        help="num_samples for confidence parameter used in smoothing setup.",
    )
    parser.add_argument(
        "--noise-sub-batch-size",
        type=int,
        default=300,
        help="Sub-batch size along noise/sample dimension for stochastic sampling.",
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
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis" / "adversarial"),
        help="Directory where adversarial JSON is saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where adversarial figure is saved.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="trajflow_adversarial_tradeoff.pdf",
        help="Filename for Wasserstein-vs-certification plot.",
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

    if args.attack_num_candidates <= 0:
        raise ValueError("attack-num-candidates must be positive")
    if args.attack_screening_samples <= 0:
        raise ValueError("attack-screening-samples must be positive")
    if args.attack_topk_w2 <= 0:
        raise ValueError("attack-topk-w2 must be positive")
    if args.attack_distribution_samples <= 0:
        raise ValueError("attack-distribution-samples must be positive")
    if args.distribution_samples <= 0:
        raise ValueError("distribution-samples must be positive")
    if args.noise_sub_batch_size <= 0:
        raise ValueError("noise-sub-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if args.ot_method != "emd" and args.sinkhorn_reg <= 0.0:
        raise ValueError("sinkhorn-reg must be positive")

    radii = _parse_radius_values(args.radius_values)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    x_position = _prepare_inputs_from_arrays(
        selected_indices=selected_indices,
        inputs=all_inputs,
        outputs_gt=outputs_gt,
    )
    cluster_anchor = _extract_target_initial_position(x_position)

    if args.input_mode == "position":
        model_input = x_position
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x_position)

    params_k1 = _build_parameters(args, max_num_clusters=1)
    params_k2 = _build_parameters(args, max_num_clusters=2)

    # Pre-build clean smoothers once per input — they don't depend on radius.
    # Only robust_certification(radius) varies; building them inside the radius
    # loop would multiply construction cost by len(radii) unnecessarily.
    print(f"pre-building clean smoothers for {len(selected_indices)} inputs ...")
    t_build = time.perf_counter()
    clean_smoothers: list[tuple[GaussianAlphaTrimmingSmoothing, GaussianAlphaTrimmingSmoothing]] = []
    input_meta: list[tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]] = []
    fixed_noisy_reference: list[torch.Tensor] = []
    for i in range(len(selected_indices)):
        x_i = model_input[i : i + 1]
        h_i = _slice_helper(helper, i)
        anchor_i = cluster_anchor[i : i + 1].to(device=x_i.device, dtype=x_i.dtype)
        s_k1 = GaussianAlphaTrimmingSmoothing(
            stochastic_function=stochastic_fn,
            x=x_i,
            sigma=args.sigma,
            parameters=params_k1,
            h=h_i,
            cluster_anchor=anchor_i,
        )
        s_k2 = GaussianAlphaTrimmingSmoothing(
            stochastic_function=stochastic_fn,
            x=x_i,
            sigma=args.sigma,
            parameters=params_k2,
            h=h_i,
            cluster_anchor=anchor_i,
        )
        clean_smoothers.append((s_k1, s_k2))
        input_meta.append((x_i, h_i, anchor_i))

        dimension = x_i.shape[-1]
        base_noise_i = MultivariateGaussian(
            mean=torch.zeros(dimension, dtype=x_i.dtype, device=x_i.device),
            cov=(args.sigma**2) * torch.eye(dimension, dtype=x_i.dtype, device=x_i.device),
        )
        noisy_ref_i = stochastic_fn.sample_from_noisy_inputs(
            x=x_i,
            smoothing_distribution=base_noise_i,
            num_samples=args.distribution_samples,
            h=h_i,
            noise_sub_batch_size=args.noise_sub_batch_size,
        )[0]
        fixed_noisy_reference.append(noisy_ref_i)
    print(f"smoothers built in {time.perf_counter() - t_build:.1f}s")

    per_radius_rows: list[dict[str, Any]] = []
    prev_best_delta_k1: list[torch.Tensor | None] = [None] * len(selected_indices)
    prev_best_delta_k2: list[torch.Tensor | None] = [None] * len(selected_indices)

    for radius in radii:
        t0 = time.perf_counter()
        print(f"processing radius r={radius:.4f} over {len(selected_indices)} inputs")

        per_input_rows: list[dict[str, Any]] = []

        for i, dataset_idx in enumerate(selected_indices.tolist()):
            x_i, h_i, anchor_i = input_meta[i]
            smoother_k1_clean, smoother_k2_clean = clean_smoothers[i]

            cert_radius = torch.full((1, 1), radius, dtype=x_i.dtype, device=x_i.device)
            cert_prob_k1 = float(smoother_k1_clean.robust_certification(radius=cert_radius).total_probs[0].item())
            cert_prob_k2 = float(smoother_k2_clean.robust_certification(radius=cert_radius).total_probs[0].item())

            delta_k1, best_w2_k1 = _find_best_attack_for_smoother(
                stochastic_fn=stochastic_fn,
                smoother=smoother_k1_clean,
                x_single=x_i,
                helper_single=h_i,
                sigma=args.sigma,
                radius=radius,
                num_candidates=args.attack_num_candidates,
                num_distribution_samples=args.distribution_samples,
                attack_distribution_samples=args.attack_distribution_samples,
                sinkhorn_reg=args.sinkhorn_reg,
                ot_method=args.ot_method,
                noise_sub_batch_size=args.noise_sub_batch_size,
                attack_search_strategy=args.attack_search_strategy,
                attack_screening_samples=args.attack_screening_samples,
                attack_topk_w2=args.attack_topk_w2,
                warm_start_delta=prev_best_delta_k1[i],
                noisy_reference_samples=fixed_noisy_reference[i],
            )
            delta_k2, best_w2_k2 = _find_best_attack_for_smoother(
                stochastic_fn=stochastic_fn,
                smoother=smoother_k2_clean,
                x_single=x_i,
                helper_single=h_i,
                sigma=args.sigma,
                radius=radius,
                num_candidates=args.attack_num_candidates,
                num_distribution_samples=args.distribution_samples,
                attack_distribution_samples=args.attack_distribution_samples,
                sinkhorn_reg=args.sinkhorn_reg,
                ot_method=args.ot_method,
                noise_sub_batch_size=args.noise_sub_batch_size,
                attack_search_strategy=args.attack_search_strategy,
                attack_screening_samples=args.attack_screening_samples,
                attack_topk_w2=args.attack_topk_w2,
                warm_start_delta=prev_best_delta_k2[i],
                noisy_reference_samples=fixed_noisy_reference[i],
            )
            prev_best_delta_k1[i] = delta_k1.detach().clone()
            prev_best_delta_k2[i] = delta_k2.detach().clone()

            per_input_rows.append(
                {
                    "dataset_index": int(dataset_idx),
                    "certified_prob_k1": cert_prob_k1,
                    "certified_prob_k2": cert_prob_k2,
                    "max_w2_k1": float(best_w2_k1),
                    "max_w2_k2": float(best_w2_k2),
                    "attack_norm_k1": float(torch.linalg.norm(delta_k1).item()),
                    "attack_norm_k2": float(torch.linalg.norm(delta_k2).item()),
                    "attack_delta_k1": delta_k1.detach().cpu().tolist(),
                    "attack_delta_k2": delta_k2.detach().cpu().tolist(),
                }
            )

        w2_k1 = np.array([float(row["max_w2_k1"]) for row in per_input_rows], dtype=float)
        w2_k2 = np.array([float(row["max_w2_k2"]) for row in per_input_rows], dtype=float)
        cert_k1 = np.array([float(row["certified_prob_k1"]) for row in per_input_rows], dtype=float)
        cert_k2 = np.array([float(row["certified_prob_k2"]) for row in per_input_rows], dtype=float)

        row = {
            "radius": float(radius),
            "mean_max_w2_k1": float(np.mean(w2_k1)),
            "std_max_w2_k1": float(np.std(w2_k1)),
            "mean_max_w2_k2": float(np.mean(w2_k2)),
            "std_max_w2_k2": float(np.std(w2_k2)),
            "mean_certified_prob_k1": float(np.mean(cert_k1)),
            "std_certified_prob_k1": float(np.std(cert_k1)),
            "mean_certified_prob_k2": float(np.mean(cert_k2)),
            "std_certified_prob_k2": float(np.std(cert_k2)),
            "per_input": per_input_rows,
        }
        per_radius_rows.append(row)

        elapsed = time.perf_counter() - t0
        print(
            f"radius r={radius:.4f} done in {elapsed:.1f}s | "
            f"mean_w2(k1,k2)=({row['mean_max_w2_k1']:.4f},{row['mean_max_w2_k2']:.4f}) | "
            f"mean_cert(k1,k2)=({row['mean_certified_prob_k1']:.4f},{row['mean_certified_prob_k2']:.4f})"
        )

    analysis_dir = Path(args.analysis_dir)
    figure_path = Path(args.figures_dir) / args.figure_name
    _plot_wasserstein_certification_tradeoff(
        radius_rows=per_radius_rows,
        output_path=figure_path,
    )

    summary = {
        "config": {
            "selected_indices": selected_indices.tolist(),
            "radius_values": [float(r) for r in radii],
            "attack_num_candidates": int(args.attack_num_candidates),
            "attack_search_strategy": args.attack_search_strategy,
            "attack_screening_samples": int(args.attack_screening_samples),
            "attack_topk_w2": int(args.attack_topk_w2),
            "attack_distribution_samples": int(args.attack_distribution_samples),
            "distribution_samples": int(args.distribution_samples),
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
            "confidence_num_samples": int(args.confidence_num_samples),
            "noise_sub_batch_size": int(args.noise_sub_batch_size),
            "sinkhorn_reg": float(args.sinkhorn_reg),
            "ot_method": args.ot_method,
            "seed": int(args.seed),
            "fixed_index_order_file": str(fixed_order_path),
            "figure_path": str(figure_path),
        },
        "per_radius": per_radius_rows,
    }

    json_path = analysis_dir / "trajflow_adversarial.json"
    _save_json(json_path, summary)

    print("=== TrajFlow Adversarial Experiment ===")
    print(f"selected_indices: {selected_indices.tolist()}")
    print(f"saved: {json_path}")
    print(f"saved: {figure_path}")


if __name__ == "__main__":
    main()
