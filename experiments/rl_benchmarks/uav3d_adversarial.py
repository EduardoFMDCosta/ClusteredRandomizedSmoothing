#!/usr/bin/env python3
"""Empirical-attack sweep for the UAV3D task-directed state attack.

Mirrors ``experiments/paper/trajflow_adversarial.py``'s structure — sweep an
attack radius and find the worst-case perturbation at each radius — but for
the UAV3D controller, with the task-directed attack objective from
``uav3d_attacks.py`` instead of black-box Wasserstein maximization, and three
branches instead of two: the raw policy, single-cluster alpha-smoothing, and
clustered robust smoothing.

For each representative state and each swept radius, the task-directed search
finds the worst-case ``delta`` independently per branch (against its own
smoother/policy), and the reported quantity is the Wasserstein-2 distance
between the clean and attacked *action* distributions — for the smoothed
branches this is computed via ``smoother.monte_carlo(z=...)`` reusing the
*same* clean-anchored cluster the branch was built with (not a freshly
re-clustered smoother), and analogously for the raw branch via repeated
``policy.sample()`` draws.

``robust_certification(radius)`` for the clustered branch is still computed
and stored in the output JSON, but is not plotted — the figure only compares
the three branches' empirical action-shift under attack.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from collect_uav3d_rollouts import _load_uav3d_policy
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import VariationalPolicy, normalize_state
from uav3d_attacks import (
    _compute_sinkhorn_wasserstein_2_distance,
    _find_best_delta_for_smoother,
    _find_best_delta_raw,
)

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RuntimeHeuristics,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

_BRANCH_STYLE = {
    "raw": {"color": "#CC3311", "marker": "o", "label": "Raw policy"},
    "alpha": {"color": "#EE7733", "marker": "^", "label": "Alpha-smoothing"},
    "robust": {"color": "#0077BB", "marker": "s", "label": "Clustered robust smoothing"},
}


def _parse_radius_values(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("radius-values must contain at least one value")
    radii = [float(v) for v in values]
    if any(r < 0.0 for r in radii):
        raise ValueError("all radius values must be non-negative")
    return radii


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sample_representative_states(
    env: UAV3D,
    policy: VariationalPolicy,
    num_states: int,
    num_seed_rollouts: int,
    horizon: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Pool states visited across a few clean, deterministic rollouts from
    ``BIMODAL_START_STATE``, then subsample ``num_states`` of them with a
    fixed RNG. Gives a handful of "typical, in-flight" states to attack,
    without needing a persistent dataset-index cache like the TrajFlow
    precedent (UAV3D states are cheap to regenerate on demand).
    """
    rng = np.random.default_rng(seed)
    visited: list[np.ndarray] = []
    for i in range(num_seed_rollouts):
        np.random.seed(seed + 1000 + i)
        torch.manual_seed(seed + 1000 + i)
        env._state = BIMODAL_START_STATE.copy()
        env._step_count = 0
        env._done = False
        env._terminal_reason = None
        pos = np.array([env._state[0], env._state[2], env._state[4]], dtype=np.float32)
        env._prev_dist = float(np.linalg.norm(pos - env.goal_center))

        state = env._state.copy()
        steps = 0
        while not env.is_done() and steps < horizon:
            state_t = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.deterministic(state_t)
            a = action.squeeze(0).cpu().numpy().astype(np.float32)
            _, _, state = env.step(a)
            visited.append(state.copy())
            steps += 1

    if not visited:
        raise ValueError("No visited states collected — check --state-horizon/--num-seed-rollouts.")
    visited_arr = np.stack(visited, axis=0)
    num_states = min(num_states, visited_arr.shape[0])
    idx = rng.choice(visited_arr.shape[0], size=num_states, replace=False)
    return visited_arr[idx]


def _plot_radius_tradeoff(
    radius_rows: list[dict[str, Any]],
    output_path: Path,
    branches: list[str],
) -> None:
    if not radius_rows:
        raise ValueError("radius_rows must be non-empty")

    radii = np.array([float(r["radius"]) for r in radius_rows], dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for branch in branches:
        style = _BRANCH_STYLE[branch]
        mean = np.array([float(r[f"mean_w2_{branch}"]) for r in radius_rows], dtype=float)
        std = np.array([float(r[f"std_w2_{branch}"]) for r in radius_rows], dtype=float)
        ax.fill_between(radii, mean - std, mean + std, alpha=0.2, color=style["color"])
        ax.plot(radii, mean, marker=style["marker"], linewidth=1.4, color=style["color"], label=style["label"])

    ax.set_xlabel("Attack radius ε (normalized state-space L2)")
    ax.set_ylabel("Mean Wasserstein-2 (attacked vs. clean action)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--num-states", type=int, default=8,
                        help="Number of representative in-flight states to attack.")
    parser.add_argument("--num-seed-rollouts", type=int, default=20,
                        help="Clean deterministic rollouts used to pool candidate states.")
    parser.add_argument("--state-horizon", type=int, default=64,
                        help="Max steps per seed rollout used only to sample states.")
    parser.add_argument("--radius-values", type=str, default="0.0,0.02,0.05,0.08,0.12,0.16")
    parser.add_argument("--attack-num-candidates", type=int, default=32)
    parser.add_argument("--attack-screen-samples", type=int, default=6)
    parser.add_argument("--attack-obstacle-margin-weight", type=float, default=0.0)
    parser.add_argument("--distribution-samples", type=int, default=300,
                        help="Samples used for the final reported Wasserstein distance.")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--smoothing-N", type=int, default=30)
    parser.add_argument("--confidence-N", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.45)
    parser.add_argument("--dbscan-min-samples", type=int, default=50)
    parser.add_argument("--noise-sub-batch-size", type=int, default=300)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1)
    parser.add_argument("--ot-method", type=str,
                        choices=("sinkhorn_log", "sinkhorn_stabilized", "sinkhorn", "emd"),
                        default="sinkhorn_log")
    parser.add_argument("--no-alpha", action="store_true",
                        help="Skip the single-cluster alpha-smoothing branch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-dir", type=str,
                        default="experiments/uav_3d/analysis/adversarial")
    parser.add_argument("--figures-dir", type=str,
                        default="experiments/uav_3d/adversarial_figures")
    parser.add_argument("--figure-name", type=str, default="uav3d_adversarial_tradeoff.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    radii = _parse_radius_values(args.radius_values)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    policy.to(device)
    print(f"Loaded VI-BNN policy from {checkpoint_path} (action_scale={action_scale}) onto {device}.")

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
    env = UAV3D(gaussian_noise=True, noise_std=0.05, max_steps=None)

    print(f"Sampling {args.num_states} representative states from {args.num_seed_rollouts} seed rollouts...")
    states = _sample_representative_states(
        env=env, policy=policy, num_states=args.num_states,
        num_seed_rollouts=args.num_seed_rollouts, horizon=args.state_horizon,
        seed=args.seed, device=device,
    )

    def _make_parameters(max_clusters: int) -> Parameters:
        return Parameters(
            smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
            clustering=ClusteringParameters(
                max_num_clusters=max_clusters, coverage=args.coverage,
                dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
            ),
            confidence=ConfidenceParameters(num_samples=args.confidence_N, beta=1e-2),
            runtime=RuntimeHeuristics(noise_sub_batch_size=args.noise_sub_batch_size),
        )

    smoothed_branches = ["alpha", "robust"] if not args.no_alpha else ["robust"]
    parameters_by_branch = {
        "robust": _make_parameters(args.max_clusters),
        "alpha": _make_parameters(1),
    }
    all_branches = ["raw"] + smoothed_branches

    # Pre-build one smoother per (state, smoothed branch) — reused across all
    # radii, only robust_certification(radius) and the attack search vary per
    # radius.
    print(f"Pre-building smoothers for {len(states)} states x {len(smoothed_branches)} branch(es)...")
    t_build = time.perf_counter()
    xs: list[torch.Tensor] = []
    smoothers: dict[str, list[GaussianAlphaTrimmingSmoothing]] = {b: [] for b in smoothed_branches}
    clean_samples: dict[str, list[np.ndarray]] = {b: [] for b in all_branches}
    for state in states:
        x_i = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        xs.append(x_i)
        for branch in smoothed_branches:
            smoother = GaussianAlphaTrimmingSmoothing(
                stochastic_function=stochastic_fn, x=x_i, sigma=args.sigma,
                parameters=parameters_by_branch[branch],
            )
            smoothers[branch].append(smoother)
            clean_samples[branch].append(
                smoother.monte_carlo(num_simulations=args.distribution_samples, z=x_i)[0].detach().cpu().numpy()
            )
        with torch.no_grad():
            x_rep = x_i.expand(args.distribution_samples, -1)
            raw_actions, _, _ = policy.sample(x_rep)
        clean_samples["raw"].append(raw_actions.detach().cpu().numpy())
    print(f"smoothers built in {time.perf_counter() - t_build:.1f}s")

    per_radius_rows: list[dict[str, Any]] = []
    for radius in radii:
        t0 = time.perf_counter()
        per_state_rows: list[dict[str, Any]] = []

        for i, state in enumerate(states):
            x_i = xs[i]
            state_row: dict[str, Any] = {"state_index": i}

            if "robust" in smoothed_branches:
                cert_radius = torch.full((1, 1), radius, dtype=x_i.dtype, device=x_i.device)
                cert_prob_robust = float(
                    smoothers["robust"][i].robust_certification(radius=cert_radius).total_probs[0].item()
                )
                state_row["cert_prob_robust"] = cert_prob_robust

            for branch in smoothed_branches:
                smoother = smoothers[branch][i]
                delta = _find_best_delta_for_smoother(
                    smoother=smoother, env=env, true_state=state, radius=radius,
                    num_candidates=args.attack_num_candidates,
                    screen_samples=args.attack_screen_samples,
                    obstacle_margin_weight=args.attack_obstacle_margin_weight,
                )
                attacked = smoother.monte_carlo(
                    num_simulations=args.distribution_samples, z=x_i + delta.unsqueeze(0)
                )[0].detach().cpu().numpy()
                w2 = _compute_sinkhorn_wasserstein_2_distance(
                    clean_samples[branch][i], attacked, reg=args.sinkhorn_reg, ot_method=args.ot_method,
                )
                state_row[f"w2_{branch}"] = w2
                state_row[f"attack_norm_{branch}"] = float(torch.linalg.norm(delta).item())

            delta_raw = _find_best_delta_raw(
                policy=policy, env=env, true_state=state, x=x_i, radius=radius,
                num_candidates=args.attack_num_candidates,
                obstacle_margin_weight=args.attack_obstacle_margin_weight,
            )
            with torch.no_grad():
                x_adv_rep = (x_i + delta_raw.unsqueeze(0)).expand(args.distribution_samples, -1)
                raw_attacked_actions, _, _ = policy.sample(x_adv_rep)
            w2_raw = _compute_sinkhorn_wasserstein_2_distance(
                clean_samples["raw"][i], raw_attacked_actions.detach().cpu().numpy(),
                reg=args.sinkhorn_reg, ot_method=args.ot_method,
            )
            state_row["w2_raw"] = w2_raw
            state_row["attack_norm_raw"] = float(torch.linalg.norm(delta_raw).item())

            per_state_rows.append(state_row)

        row: dict[str, Any] = {"radius": float(radius)}
        for branch in all_branches:
            arr = np.array([r[f"w2_{branch}"] for r in per_state_rows], dtype=float)
            row[f"mean_w2_{branch}"] = float(arr.mean())
            row[f"std_w2_{branch}"] = float(arr.std())
        if "robust" in smoothed_branches:
            cert_arr = np.array([r["cert_prob_robust"] for r in per_state_rows], dtype=float)
            row["mean_cert_prob_robust"] = float(cert_arr.mean())
            row["std_cert_prob_robust"] = float(cert_arr.std())
        row["per_state"] = per_state_rows
        per_radius_rows.append(row)

        elapsed = time.perf_counter() - t0
        w2_summary = ", ".join(f"{b}={row[f'mean_w2_{b}']:.4f}" for b in all_branches)
        print(f"radius r={radius:.4f} done in {elapsed:.1f}s | mean_w2({w2_summary})")

    analysis_dir = Path(args.analysis_dir)
    figure_path = Path(args.figures_dir) / args.figure_name
    _plot_radius_tradeoff(per_radius_rows, figure_path, branches=all_branches)

    summary = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "num_states": int(args.num_states),
            "num_seed_rollouts": int(args.num_seed_rollouts),
            "state_horizon": int(args.state_horizon),
            "radius_values": [float(r) for r in radii],
            "attack_num_candidates": int(args.attack_num_candidates),
            "attack_screen_samples": int(args.attack_screen_samples),
            "attack_obstacle_margin_weight": float(args.attack_obstacle_margin_weight),
            "distribution_samples": int(args.distribution_samples),
            "sigma": float(args.sigma),
            "smoothing_N": int(args.smoothing_N),
            "confidence_N": int(args.confidence_N),
            "alpha": float(args.alpha),
            "coverage": float(args.coverage),
            "max_clusters": int(args.max_clusters),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "noise_sub_batch_size": int(args.noise_sub_batch_size),
            "sinkhorn_reg": float(args.sinkhorn_reg),
            "ot_method": args.ot_method,
            "branches": all_branches,
            "seed": int(args.seed),
            "figure_path": str(figure_path),
        },
        "per_radius": per_radius_rows,
    }

    json_path = analysis_dir / "uav3d_adversarial.json"
    _save_json(json_path, summary)

    print("=== UAV3D Adversarial Experiment ===")
    print(f"saved: {json_path}")
    print(f"saved: {figure_path}")


if __name__ == "__main__":
    main()
