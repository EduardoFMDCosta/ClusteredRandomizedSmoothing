#!/usr/bin/env python3
"""Roll out UAV3D trajectories under a **robustified** controller and save
the results for later plotting.

At every environment step we:
  1. Wrap the VI-BNN policy in ``UAV3DPolicyStochasticFunction``.
  2. Build ``GaussianAlphaTrimmingSmoothing`` on the current (normalized) state
     with Gaussian state-space noise of std ``sigma``.
  3. Either sample from the resulting ``MixtureDirac`` or pick the mode
     (highest partition weight) as the action.

Optionally also rolls out the raw (non-robustified) policy and the
``max_clusters=1`` alpha-smoothing baseline for comparison.

This is the expensive half of the old combined script — it does not touch
pyvista at all. It writes a ``RolloutBundle`` (see ``rollout_io.py``) that
``visualize_uav3d_robustified_pyvista.py`` loads to render figures, so camera
angle / legend / color tweaks don't require re-running the benchmark.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE, PolicyRolloutData, RolloutBundle, save_rollouts
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import VariationalPolicy, normalize_state

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing


def _load_uav3d_policy(checkpoint_path: Path) -> tuple[VariationalPolicy, float]:
    """Load a UAV3D VI-BNN policy checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with model_state_dict (legacy checkpoints are not supported).")

    required_fields = ["state_dim", "action_dim", "hidden_dim", "action_scale"]
    missing_fields = [field for field in required_fields if field not in checkpoint]
    if missing_fields:
        raise ValueError(f"Checkpoint missing required fields: {missing_fields}")

    model_state_dict = checkpoint["model_state_dict"]
    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    action_scale = float(checkpoint["action_scale"])
    num_components = int(checkpoint.get("num_components", 1))

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        action_scale=action_scale,
        num_components=num_components,
    )
    policy.load_state_dict(model_state_dict)
    policy.eval()
    return policy, action_scale


def _robustified_action(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    state: np.ndarray,
    parameters: Parameters,
    sigma: float,
    device: torch.device,
) -> np.ndarray:
    """Build a smoother at ``state`` and return one action from the
    robustified ``MixtureDirac``.

    ``x`` is created directly on ``device`` so the ``confidence.num_samples``
    / ``smoothing.N`` additive-noise samples that ``smooth()`` evaluates the
    policy on are batched through the network on the GPU (when available)
    rather than the CPU.
    """
    x = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=parameters,
    )
    distribution = smoother.smooth()
    action_t = distribution.sample(size=())
    return action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _rollout(
    env: UAV3D,
    start_state: np.ndarray,
    horizon: int,
    action_fn,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Roll out until natural termination (goal / collision / corridor exit).

    The env should be constructed with ``max_steps=None`` so it never times
    out from inside ``step()``; ``horizon`` here is just a safety cap that
    should never bind in practice.
    """
    env._state = start_state.copy()
    env._step_count = 0
    env._done = False
    start_pos = np.array([start_state[0], start_state[2], start_state[4]], dtype=np.float32)
    env._prev_dist = float(np.linalg.norm(start_pos - env.goal_center))

    state = env._state.copy()
    trace = [start_pos.copy()]
    state_trace = [state.copy()]
    steps = 0
    while not env.is_done() and steps < horizon:
        action = action_fn(state)
        _, _reward, state = env.step(action)
        trace.append(np.array([state[0], state[2], state[4]], dtype=np.float32))
        state_trace.append(state.copy())
        steps += 1

    # Trust the env's terminal_reason: with swept-segment collision the
    # post-step position may lie past the obstacle, so deriving the outcome
    # from final_pos alone misclassifies those as "timeout".
    outcome = env._terminal_reason if env._terminal_reason is not None else "timeout"
    return np.stack(trace, axis=0), np.stack(state_trace, axis=0), outcome


def _kinematic_magnitudes(state_trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-step |acceleration| and |jerk|, estimated by finite-differencing
    the *realized* velocity rather than reading off the commanded action.

    Under process noise the commanded action and the actually-realized
    acceleration diverge, so differentiating velocity is the faithful
    estimate. The env's double-integrator matrices use dt = 1, so a plain
    first/second difference already has units of accel / jerk.
    """
    vel = state_trace[:, (1, 3, 5)]
    accel = np.diff(vel, axis=0)
    jerk = np.diff(accel, axis=0)
    return np.linalg.norm(accel, axis=1), np.linalg.norm(jerk, axis=1)


def _raw_policy_action_fn(policy: VariationalPolicy, device: torch.device):
    def f(state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = policy.sample(state_t)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)
    return f


def _robustified_action_fn(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    parameters: Parameters,
    sigma: float,
    device: torch.device,
):

    def action(state: np.ndarray) -> np.ndarray:
        a = _robustified_action(
            stochastic_fn=stochastic_fn,
            state=state,
            parameters=parameters,
            sigma=sigma,
            device=device,
        )
        return a

    return action


def _collect_trajectories(
    env: UAV3D,
    start_state: np.ndarray,
    horizon: int,
    num_trajectories: int,
    action_fn,
    label: str,
    keep_traces: bool = True,
) -> tuple[list[np.ndarray], list[str], dict[str, int], np.ndarray, np.ndarray]:
    counts = {"goal": 0, "collision": 0, "timeout": 0}
    traces: list[np.ndarray] = []
    outcomes: list[str] = []
    accel_mags: list[np.ndarray] = []
    jerk_mags: list[np.ndarray] = []
    t0 = time.time()
    for i in range(num_trajectories):
        trace, state_trace, outcome = _rollout(env, start_state, horizon, action_fn)
        if keep_traces:
            traces.append(trace)
        outcomes.append(outcome)
        counts[outcome] += 1
        accel_mag, jerk_mag = _kinematic_magnitudes(state_trace)
        accel_mags.append(accel_mag)
        jerk_mags.append(jerk_mag)
        print(f"  [{label}] {i + 1}/{num_trajectories} → {outcome} "
              f"({time.time() - t0:.1f}s elapsed)")
    return (
        traces, outcomes, counts,
        np.concatenate(accel_mags), np.concatenate(jerk_mags),
    )


_PolicyResult = tuple[list[np.ndarray], list[str], dict[str, int], np.ndarray, np.ndarray]


def _run_all_policies(
    env: UAV3D,
    args: argparse.Namespace,
    policy: VariationalPolicy,
    stochastic_fn: UAV3DPolicyStochasticFunction,
    parameters_robust: Parameters,
    parameters_alpha: Parameters,
    device: torch.device,
    keep_traces: bool,
) -> dict[str, _PolicyResult]:
    """Roll out raw / alpha / robust policies once under whatever seed the
    caller has already set. ``keep_traces=False`` drops per-step positions
    (kept only for the seed used for rendering) to save memory when this is
    called purely for aggregate statistics.
    """
    results: dict[str, _PolicyResult] = {}
    if not args.no_raw:
        print("Rolling out raw policy trajectories...")
        results["raw"] = _collect_trajectories(
            env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
            num_trajectories=args.num_trajectories,
            action_fn=_raw_policy_action_fn(policy, device=device), label="raw",
            keep_traces=keep_traces,
        )
    if not args.no_alpha:
        print(f"Rolling out alpha-smoothing (max_clusters=1) trajectories "
              f"— this calls the policy ~{2 * args.smoothing_N} times per env step...")
        results["alpha"] = _collect_trajectories(
            env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
            num_trajectories=args.num_trajectories,
            action_fn=_robustified_action_fn(
                stochastic_fn=stochastic_fn, parameters=parameters_alpha, sigma=args.sigma, device=device,
            ),
            label="alpha", keep_traces=keep_traces,
        )
    print(f"Rolling out clustered robustified trajectories "
          f"— this calls the policy ~{2 * args.smoothing_N} times per env step...")
    results["robust"] = _collect_trajectories(
        env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
        num_trajectories=args.num_trajectories,
        action_fn=_robustified_action_fn(
            stochastic_fn=stochastic_fn, parameters=parameters_robust, sigma=args.sigma, device=device,
        ),
        label="robust", keep_traces=keep_traces,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--num-trajectories", type=int, default=100)
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
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip rolling out the raw policy for comparison.")
    parser.add_argument("--no-alpha", action="store_true",
                        help="Skip rolling out the alpha-smoothing baseline "
                             "(``max_clusters=1``).")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="UAV3D process-noise std used when rolling out.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed. The rollouts saved for rendering come "
                             "from this seed; with --num-seeds > 1, additional "
                             "seeds args.seed+1, args.seed+2, ... are used for "
                             "the aggregate statistics only.")
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Number of seeds to roll out for the collision-rate "
                             "and kinematics statistics (mean ± std reported "
                             "across seeds).")
    parser.add_argument("--out", type=str,
                        default="experiments/uav_3d/bimodal_rollouts.pkl",
                        help="Where to write the rollout bundle consumed by "
                             "visualize_uav3d_robustified_pyvista.py.")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for the policy / smoothing forward passes "
                             "(e.g. 'cuda', 'cpu'). Defaults to cuda if available.")
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

    # ---- roll out over one or more seeds ----
    # Seed args.seed's rollouts are the ones saved for rendering; any
    # additional seeds (args.seed+1, args.seed+2, ...) only contribute to the
    # aggregate collision-rate / kinematics statistics below.
    num_seeds = max(1, args.num_seeds)
    seeds = [args.seed + i for i in range(num_seeds)]

    policy_names = (["raw"] if not args.no_raw else []) + \
                   (["alpha"] if not args.no_alpha else []) + ["robust"]
    per_seed_results: dict[str, list[_PolicyResult]] = {name: [] for name in policy_names}
    primary: dict[str, _PolicyResult] = {}

    for seed_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        is_primary = seed_idx == 0
        if num_seeds > 1:
            print(f"\n=== Seed {seed} ({seed_idx + 1}/{num_seeds}) ===")
        seed_results = _run_all_policies(
            env=env, args=args, policy=policy, stochastic_fn=stochastic_fn,
            parameters_robust=parameters_robust, parameters_alpha=parameters_alpha,
            device=device, keep_traces=is_primary,
        )
        for name, result in seed_results.items():
            per_seed_results[name].append(result)
            if is_primary:
                primary[name] = result

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
        print(f"{label}  |accel| (finite diff. of velocity) = "
              f"{accel_all.mean():.3f} ± {accel_all.std():.3f}  |  "
              f"|jerk| = {jerk_all.mean():.3f} ± {jerk_all.std():.3f}  "
              f"(n={accel_all.size} steps)")

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
          f"--data {out_path}")


if __name__ == "__main__":
    main()
