#!/usr/bin/env python3
"""Roll out UAV3D trajectories under a task-directed adversarial attack on the
policy's **state** input, for the raw, alpha-smoothing, and robustified
(state-space smoothed) controllers side by side.

At every environment step, before the policy sees the (normalized) state, an
attacker searches a small L2 ball ``||delta|| <= --attack-radius`` around it
for the perturbation that most hurts the mission (reduces goal progress /
steers toward an obstacle — see ``uav3d_attacks.py``), then the resulting
``state + delta`` is fed to the policy:

  * raw branch: ``policy.sample(state + delta)`` — no defense.
  * alpha branch: the same state-space ``GaussianAlphaTrimmingSmoothing``
    defense as the robust branch, but with ``max_clusters=1`` (pure
    alpha-trimming, no cluster splitting) — the baseline compared against
    ``collect_uav3d_rollouts.py``'s clean-run alpha branch.
  * robust branch: the existing state-space ``GaussianAlphaTrimmingSmoothing``
    defense from ``collect_uav3d_rollouts.py`` (``--max-clusters``, default
    3), but centered on the attacked observation ``state + delta`` (a
    deployed system only ever sees the attacked reading, never the clean
    one) — same centering applies to the alpha branch.

This is the companion, attacked-input counterpart to ``collect_uav3d_rollouts.py``
— it writes a ``RolloutBundle`` (see ``rollout_io.py``) with the same schema,
so the existing ``visualize_uav3d_robustified_pyvista.py`` renders it
unmodified and its figures are directly comparable to the clean-run figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from collect_uav3d_rollouts import (
    _collect_trajectories,
    _load_uav3d_policy,
    _robustified_action_from_x,
)
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE, PolicyRolloutData, RolloutBundle, save_rollouts
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import VariationalPolicy, normalize_state
from uav3d_attacks import _find_best_delta_for_smoother, _find_best_delta_raw

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing


def _attacked_raw_action_fn(
    policy: VariationalPolicy,
    env: UAV3D,
    radius: float,
    num_candidates: int,
    device: torch.device,
    obstacle_margin_weight: float,
    attack_prob: float,
):
    def f(state: np.ndarray) -> np.ndarray:
        x = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        attacked = attack_prob >= 1.0 or np.random.rand() < attack_prob
        if attacked:
            delta = _find_best_delta_raw(
                policy=policy, env=env, true_state=state, x=x,
                radius=radius, num_candidates=num_candidates,
                obstacle_margin_weight=obstacle_margin_weight,
            )
            x = x + delta.unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = policy.sample(x)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)
    return f


def _attacked_robust_action_fn(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    env: UAV3D,
    parameters: Parameters,
    sigma: float,
    radius: float,
    num_candidates: int,
    screen_samples: int,
    device: torch.device,
    obstacle_margin_weight: float,
    attack_prob: float,
):
    def f(state: np.ndarray) -> np.ndarray:
        x = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        attacked = attack_prob >= 1.0 or np.random.rand() < attack_prob
        if not attacked:
            return _robustified_action_from_x(stochastic_fn, x, parameters, sigma)

        # Cheap search: reuse a smoother anchored at the clean state as a
        # proxy for "what would the robustified pipeline do near x+delta"
        # (exploits AlphaTrimmingSmoothing.monte_carlo's cluster reuse), so
        # scoring num_candidates deltas costs only screen_samples policy
        # calls each, not a full re-clustering per candidate.
        search_smoother = GaussianAlphaTrimmingSmoothing(
            stochastic_function=stochastic_fn, x=x, sigma=sigma, parameters=parameters,
        )
        delta = _find_best_delta_for_smoother(
            smoother=search_smoother, env=env, true_state=state,
            radius=radius, num_candidates=num_candidates, screen_samples=screen_samples,
            obstacle_margin_weight=obstacle_margin_weight,
        )
        x_adv = x + delta.unsqueeze(0)

        # Actually-executed action: a fresh smoother anchored at the attacked
        # observation, matching real deployment (the controller only ever
        # sees x_adv, never the clean x).
        return _robustified_action_from_x(stochastic_fn, x_adv, parameters, sigma)
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--smoothing-N", type=int, default=30,
                        help="Noisy-state samples used inside smooth() per env step.")
    parser.add_argument("--confidence-N", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.45)
    parser.add_argument("--dbscan-min-samples", type=int, default=50)
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="UAV3D process-noise std used when rolling out "
                             "(unrelated to the adversarial attack).")
    parser.add_argument("--attack-radius", type=float, default=0.05,
                        help="Epsilon: L2 ball radius for the state-space attack, "
                             "in the same (normalized-state) units as --sigma.")
    parser.add_argument("--attack-num-candidates", type=int, default=24,
                        help="Candidate perturbations sampled per attack search.")
    parser.add_argument("--attack-screen-samples", type=int, default=6,
                        help="Monte Carlo draws per candidate when scoring "
                             "against the robust branch's smoother.")
    parser.add_argument("--attack-obstacle-margin-weight", type=float, default=0.0,
                        help="Soft bias in the attack's scoring function toward "
                             "obstacle proximity, even short of an outright "
                             "one-step collision.")
    parser.add_argument("--attack-prob", type=float, default=1.0,
                        help="Probability the attack is applied at each step "
                             "(1.0 = every step).")
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip rolling out the attacked raw policy.")
    parser.add_argument("--no-alpha", action="store_true",
                        help="Skip rolling out the attacked alpha-smoothing "
                             "baseline (``max_clusters=1``).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--out", type=str,
                        default="experiments/uav_3d/attacked_rollouts.pkl",
                        help="Where to write the rollout bundle consumed by "
                             "visualize_uav3d_robustified_pyvista.py.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    policy.to(device)
    print(f"Loaded VI-BNN policy from {checkpoint_path} (action_scale={action_scale}) onto {device}.")

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)

    def _make_parameters(max_clusters: int) -> Parameters:
        return Parameters(
            smoothing=SmoothingParameters(N=args.smoothing_N, alpha=args.alpha),
            clustering=ClusteringParameters(
                max_num_clusters=max_clusters,
                coverage=args.coverage,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
            ),
            confidence=ConfidenceParameters(num_samples=args.confidence_N, beta=1e-2),
        )

    parameters_robust = _make_parameters(args.max_clusters)
    parameters_alpha = _make_parameters(1)

    # max_steps=None: rollouts run to natural termination (goal / collision
    # / corridor exit). args.horizon is a safety cap inside _rollout.
    env = UAV3D(gaussian_noise=True, noise_std=args.noise_std, max_steps=None)

    policy_names = (["raw"] if not args.no_raw else []) + \
                   (["alpha"] if not args.no_alpha else []) + ["robust"]
    num_seeds = max(1, args.num_seeds)
    seeds = [args.seed + i for i in range(num_seeds)]

    per_seed_results: dict[str, list] = {name: [] for name in policy_names}
    primary: dict[str, tuple] = {}

    for seed_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        is_primary = seed_idx == 0
        if num_seeds > 1:
            print(f"\n=== Seed {seed} ({seed_idx + 1}/{num_seeds}) ===")

        if not args.no_raw:
            print("Rolling out attacked raw policy trajectories...")
            result = _collect_trajectories(
                env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
                num_trajectories=args.num_trajectories,
                action_fn=_attacked_raw_action_fn(
                    policy=policy, env=env, radius=args.attack_radius,
                    num_candidates=args.attack_num_candidates, device=device,
                    obstacle_margin_weight=args.attack_obstacle_margin_weight,
                    attack_prob=args.attack_prob,
                ),
                label="attacked-raw", keep_traces=is_primary,
            )
            per_seed_results["raw"].append(result)
            if is_primary:
                primary["raw"] = result

        if not args.no_alpha:
            print(f"Rolling out attacked alpha-smoothing (max_clusters=1) "
                  f"trajectories — this calls the policy ~{2 * args.smoothing_N} "
                  f"times per env step, plus the attack search...")
            result = _collect_trajectories(
                env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
                num_trajectories=args.num_trajectories,
                action_fn=_attacked_robust_action_fn(
                    stochastic_fn=stochastic_fn, env=env, parameters=parameters_alpha,
                    sigma=args.sigma, radius=args.attack_radius,
                    num_candidates=args.attack_num_candidates,
                    screen_samples=args.attack_screen_samples, device=device,
                    obstacle_margin_weight=args.attack_obstacle_margin_weight,
                    attack_prob=args.attack_prob,
                ),
                label="attacked-alpha", keep_traces=is_primary,
            )
            per_seed_results["alpha"].append(result)
            if is_primary:
                primary["alpha"] = result

        print(f"Rolling out attacked robustified trajectories "
              f"— this calls the policy ~{2 * args.smoothing_N} times per env "
              f"step, plus the attack search...")
        result = _collect_trajectories(
            env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
            num_trajectories=args.num_trajectories,
            action_fn=_attacked_robust_action_fn(
                stochastic_fn=stochastic_fn, env=env, parameters=parameters_robust,
                sigma=args.sigma, radius=args.attack_radius,
                num_candidates=args.attack_num_candidates,
                screen_samples=args.attack_screen_samples, device=device,
                obstacle_margin_weight=args.attack_obstacle_margin_weight,
                attack_prob=args.attack_prob,
            ),
            label="attacked-robust", keep_traces=is_primary,
        )
        per_seed_results["robust"].append(result)
        if is_primary:
            primary["robust"] = result

    print("")
    label_map = {"raw": "Raw   ", "alpha": "Alpha ", "robust": "Robust"}
    policies: dict[str, PolicyRolloutData] = {}
    for name in policy_names:
        label = label_map[name]
        counts_seq = [r[2] for r in per_seed_results[name]]
        accel_all = np.concatenate([r[3] for r in per_seed_results[name]])
        jerk_all = np.concatenate([r[4] for r in per_seed_results[name]])
        coll_rates = np.array([c["collision"] / args.num_trajectories for c in counts_seq])
        goal_rates = np.array([c["goal"] / args.num_trajectories for c in counts_seq])
        timeout_rates = np.array([c["timeout"] / args.num_trajectories for c in counts_seq])
        rate_stats = {
            "goal": (float(goal_rates.mean()), float(goal_rates.std())),
            "collision": (float(coll_rates.mean()), float(coll_rates.std())),
        }
        traces, outcomes, counts, _, _ = primary[name]

        print(f"{label}: goal={counts['goal']} / collision={counts['collision']} / "
              f"timeout={counts['timeout']}  (seed={args.seed})")
        print(f"{label}  collision rate = {coll_rates.mean():.3f} ± {coll_rates.std():.3f} "
              f"(goal = {goal_rates.mean():.3f} ± {goal_rates.std():.3f}, "
              f"timeout = {timeout_rates.mean():.3f} ± {timeout_rates.std():.3f}) "
              f"over {num_seeds} seed(s) × {args.num_trajectories} trajectories")

        policies[name] = PolicyRolloutData(
            traces=traces,
            outcomes=outcomes,
            counts=counts,
            rate_stats=rate_stats,
            accel_mean=float(accel_all.mean()),
            accel_std=float(accel_all.std()),
            jerk_mean=float(jerk_all.mean()),
            jerk_std=float(jerk_all.std()),
            n_kinematic_samples=int(accel_all.size),
        )

    bundle = RolloutBundle(
        meta={
            "checkpoint": str(checkpoint_path),
            "num_trajectories": args.num_trajectories,
            "horizon": args.horizon,
            "sigma": args.sigma,
            "smoothing_N": args.smoothing_N,
            "confidence_N": args.confidence_N,
            "alpha": args.alpha,
            "coverage": args.coverage,
            "max_clusters": args.max_clusters,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min_samples": args.dbscan_min_samples,
            "noise_std": args.noise_std,
            "attack_radius": args.attack_radius,
            "attack_num_candidates": args.attack_num_candidates,
            "attack_screen_samples": args.attack_screen_samples,
            "attack_obstacle_margin_weight": args.attack_obstacle_margin_weight,
            "attack_prob": args.attack_prob,
            "seed": args.seed,
            "num_seeds": num_seeds,
        },
        policies=policies,
    )
    out_path = Path(args.out)
    save_rollouts(bundle, out_path)
    print(f"\nSaved rollout bundle to {out_path}")
    print("Render figures from it with: "
          "python visualize_uav3d_robustified_pyvista.py "
          f"--data {out_path} --no-alpha")


if __name__ == "__main__":
    main()
