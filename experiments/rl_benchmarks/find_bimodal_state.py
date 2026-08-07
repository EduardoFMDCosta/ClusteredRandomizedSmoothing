#!/usr/bin/env python3
"""Verify (or re-pick) the state the adversarial experiments are anchored at.

The whole adversarial study is a single-state experiment run from
``rollout_io.BIMODAL_START_STATE``. That constant is tied to one ``policy.pt``,
so **any retrain invalidates it**: the gate network is re-initialised and
re-optimised, and there is no reason the new policy is bimodal at the same point.

Run this after every retrain, before anything else.

A usable start state must satisfy all of:

  1. **Zero velocity** -- no momentum the attacker did not choose, which would
     confound exactly the first few steps where the mode decision is made.
  2. **z = 0** -- fixed, so the search is a readable 2-D slice and the start
     sits mid-corridor in altitude.
  3. **x <= -11** -- west of the westmost obstacle face, so the entire obstacle
     field still separates the UAV from the goal and the two route bundles stay
     genuinely distinct.
  4. **Coordinates are multiples of ``--grid-step``** (default 0.1). Candidates
     are enumerated on the lattice, never sampled and rounded afterwards, so the
     reported state is exactly the one that was screened.
  5. **Raw minor-mode mass >= 0.20** -- read off the policy's own mixture gate,
     never off a smoother's clustering, so the measurement cannot be circular.
  6. **Mode separation >= 1.5** -- two coincident means are one mode wearing two
     hats; averaging them costs nothing, which is the null case.
  7. **Raw goal rate >= 0.70** -- the undefended policy must actually complete
     the mission. A state where nothing succeeds makes the comparison vacuous,
     because every branch collides regardless of what it does with the modes.
  8. **Alpha-smoothing collision rate >= 0.40** -- and the single-cluster defense
     must actually *fail* there. This is the phenomenon itself, measured in the
     task rather than inferred: the raw policy succeeds, and collapsing its
     bimodal action distribution onto one atom drives it into an obstacle.

Criteria 7 and 8 together are the whole point: a state where the raw policy is
fine and ``k=1`` smoothing is not. Both are measured by real rollouts through the
real env -- no critic, no value estimate, no proxy.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from collect_uav3d_rollouts import (
    _load_uav3d_policy,
    _raw_policy_action_fn,
    _robustified_action_fn,
    _rollout,
)
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import VariationalPolicy, normalize_state
from uav3d_attack_sweep import defense_parameters

# How much probability must sit in the second mode. Below ~0.2 the minority mode
# is something the smoothed controller would almost never emit, so collapsing it
# has no closed-loop consequence to measure.
MIN_MINOR_MASS = 0.20

# Component means closer than this (in action units, action_scale=4) are the same
# mode; averaging them is harmless.
MIN_SEPARATION = 1.5

# The undefended policy must succeed here...
MIN_RAW_GOAL_RATE = 0.70

# ...and single-cluster alpha-smoothing must not. This is measured, not assumed.
MIN_ALPHA_COLLISION_RATE = 0.40

# Per-axis divisors of ``train_rl.normalize_state``, derived from it rather than
# restated so the two cannot drift apart.
_NORMALIZER = 1.0 / normalize_state(np.ones(6, dtype=np.float32))


def gate_modality(
    policy: VariationalPolicy,
    states: np.ndarray,
    action_scale: float,
    device: torch.device,
    batch_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    """``(minor_mode_mass, mode_separation)`` for a batch of raw states.

    ``states`` is ``(n, 6)``; both outputs are ``(n,)``. Batched because the
    policy forward pass dominates the lattice screen.
    """
    states = np.atleast_2d(states)
    masses: list[np.ndarray] = []
    separations: list[np.ndarray] = []

    for start in range(0, states.shape[0], batch_size):
        chunk = states[start : start + batch_size] / _NORMALIZER
        x = torch.tensor(chunk, dtype=torch.float32, device=device)
        with torch.no_grad():
            means, _, log_mix_w = policy.distribution_params(x)
        weights = log_mix_w.exp()
        if weights.shape[1] < 2:
            masses.append(np.ones(chunk.shape[0]))
            separations.append(np.zeros(chunk.shape[0]))
            continue
        modes = torch.tanh(means) * action_scale
        masses.append(weights.min(dim=1).values.cpu().numpy())
        separations.append(torch.cdist(modes, modes).amax(dim=(1, 2)).cpu().numpy())

    return np.concatenate(masses), np.concatenate(separations)


def valid_positions(env: UAV3D, positions: np.ndarray) -> np.ndarray:
    """Vectorized form of the env's own validity predicates. ``(n, 3) -> (n,)``.

    Mirrors ``UAV3D._outside_corridor`` / ``_check_goal`` / the obstacle AABB
    test, all of which are scalar. A position inside the goal box is excluded
    because the episode would end immediately; one inside an obstacle is not a
    state the system can occupy.
    """
    lower = np.array([-15.0, -9.0, -7.0])
    upper = np.array([15.0, 9.0, 7.0])
    valid = np.all((positions >= lower) & (positions <= upper), axis=1)
    valid &= ~np.all((positions >= env.goal_min) & (positions <= env.goal_max), axis=1)

    for x_min, x_max, y_min, y_max, z_min, z_max in env.obstacles:
        box_low = np.array([x_min, y_min, z_min])
        box_high = np.array([x_max, y_max, z_max])
        valid &= ~np.all((positions >= box_low) & (positions <= box_high), axis=1)

    return valid


def measure_outcomes(
    env: UAV3D,
    action_fn,
    state: np.ndarray,
    num_trajectories: int,
    horizon: int,
    gamma: float,
    seed: int,
) -> tuple[dict[str, float], float]:
    """Outcome rates and mean discounted return from ``state``.

    Runs the real ``_rollout`` against the real env -- the same code path the
    experiments use -- so there is no second simulator to keep in sync with
    ``UAV3D.step``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    counts = {"goal": 0, "collision": 0, "timeout": 0}
    returns: list[float] = []
    for _ in range(num_trajectories):
        _, _, outcome, discounted_return = _rollout(env, state, horizon, action_fn, gamma)
        counts[outcome] += 1
        returns.append(discounted_return)
    rates = {k: v / num_trajectories for k, v in counts.items()}
    return rates, float(np.mean(returns))


def assess(
    env: UAV3D,
    policy: VariationalPolicy,
    stochastic_fn: UAV3DPolicyStochasticFunction,
    state: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    gamma: float,
    num_trajectories: int,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """``(raw_rates, raw_return, alpha_rates)`` for one candidate state."""
    raw_rates, raw_return = measure_outcomes(
        env, _raw_policy_action_fn(policy, device=device), state,
        num_trajectories, args.horizon, gamma, args.seed,
    )
    alpha_rates, _ = measure_outcomes(
        env,
        _robustified_action_fn(
            stochastic_fn=stochastic_fn,
            parameters=defense_parameters(1, args.alpha),
            sigma=args.sigma, device=device,
        ),
        state, num_trajectories, args.horizon, gamma, args.seed,
    )
    return raw_rates, raw_return, alpha_rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--grid-step", type=float, default=0.1,
                        help="Candidate lattice resolution. Every reported "
                             "coordinate is an exact multiple of this.")
    parser.add_argument("--min-minor-mass", type=float, default=MIN_MINOR_MASS)
    parser.add_argument("--min-separation", type=float, default=MIN_SEPARATION)
    parser.add_argument("--min-raw-goal-rate", type=float, default=MIN_RAW_GOAL_RATE)
    parser.add_argument("--min-alpha-collision-rate", type=float,
                        default=MIN_ALPHA_COLLISION_RATE)
    parser.add_argument("--shortlist", type=int, default=40,
                        help="Spatially-distinct candidates sent to the rollout screen.")
    parser.add_argument("--shortlist-spacing", type=float, default=0.8,
                        help="Minimum separation between shortlisted candidates, so "
                             "the shortlist explores distinct positions rather than "
                             "many lattice neighbours of the same point.")
    parser.add_argument("--screen-trajectories", type=int, default=40,
                        help="Rollouts per candidate in the coarse raw-policy screen.")
    parser.add_argument("--verify-top", type=int, default=6,
                        help="Best screened candidates taken to the alpha-smoothing "
                             "stage, which is ~100x more expensive per rollout.")
    parser.add_argument("--verify-trajectories", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1000)
    # Must match the defense the experiments run, or the measured alpha collision
    # rate is not the one the sweep will see.
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha-trim fraction, shared by both smoothed branches. "
                             "Must be BELOW the state's minor-mode mass: alpha-trimming "
                             "averages the modes only while some minority samples "
                             "survive the trim. At alpha >= minor mass it removes the "
                             "minority mode outright and k=1 silently becomes a mode "
                             "*selector*, which does not fail and is not the baseline "
                             "the paper is comparing against.")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    policy.to(device)
    gamma = float(torch.load(checkpoint_path, map_location="cpu").get("gamma", 0.99))
    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
    env = UAV3D(gaussian_noise=True, noise_std=args.noise_std, max_steps=None)

    if policy.num_components < 2:
        raise SystemExit(
            f"Policy has num_components={policy.num_components}; a bimodal state "
            "cannot exist. Retrain with --num-components 2."
        )

    westmost = min(obstacle[0] for obstacle in env.obstacles)
    decimals = max(0, int(np.ceil(-np.log10(args.grid_step))) + 1)

    def as_literal(state: np.ndarray) -> str:
        return f"np.array({[round(float(v), decimals) for v in state]}, dtype=np.float32)"

    def report(label: str, state: np.ndarray, mass: float, separation: float,
               raw_rates: dict[str, float], raw_return: float,
               alpha_rates: dict[str, float]) -> bool:
        position = state[[0, 2, 4]]
        checks = [
            ("zero velocity", float(np.abs(state[[1, 3, 5]]).max()), "== 0",
             bool(np.all(state[[1, 3, 5]] == 0.0))),
            ("z", float(position[2]), "== 0", float(position[2]) == 0.0),
            ("x", float(position[0]), f"<= {westmost}", float(position[0]) <= westmost),
            ("minor-mode mass", mass, f">= {args.min_minor_mass}",
             mass >= args.min_minor_mass),
            ("mode separation", separation, f">= {args.min_separation}",
             separation >= args.min_separation),
            ("raw goal rate", raw_rates["goal"], f">= {args.min_raw_goal_rate}",
             raw_rates["goal"] >= args.min_raw_goal_rate),
            ("alpha collision rate", alpha_rates["collision"],
             f">= {args.min_alpha_collision_rate}",
             alpha_rates["collision"] >= args.min_alpha_collision_rate),
        ]
        print(f"{label} = {[round(float(v), decimals) for v in state]}")
        for name, value, requirement, passed in checks:
            print(f"  {name:>20} : {value:+8.4f}  (need {requirement:<9}) "
                  f"{'ok' if passed else 'FAIL'}")
        print(f"  {'raw mean return':>20} : {raw_return:+8.3f}   "
              f"raw {raw_rates['goal']:.0%} goal / {raw_rates['collision']:.0%} collision, "
              f"alpha {alpha_rates['goal']:.0%} goal / {alpha_rates['collision']:.0%} collision")
        ok = all(passed for _, _, _, passed in checks)
        print(f"  => {'PASS' if ok else 'FAIL'}\n")
        return ok

    mass, separation = gate_modality(policy, BIMODAL_START_STATE, action_scale, device)
    raw_rates, raw_return, alpha_rates = assess(
        env, policy, stochastic_fn, BIMODAL_START_STATE, args, device, gamma,
        args.verify_trajectories,
    )
    if report("BIMODAL_START_STATE", BIMODAL_START_STATE, float(mass[0]),
              float(separation[0]), raw_rates, raw_return, alpha_rates):
        print("Gate passed. rollout_io.BIMODAL_START_STATE is still usable; proceed.")
        return

    # ---- lattice: z = 0 and zero velocity reduce the search to an (x, y) slice ----
    axis = lambda lo, hi: np.round(
        np.arange(lo, hi + 1e-9, args.grid_step), decimals
    )
    xs, ys = np.meshgrid(axis(-15.0, westmost), axis(-9.0, 9.0), indexing="ij")
    positions = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1
    )

    keep = valid_positions(env, positions)
    stages = [(f"on the z=0, x<={westmost} lattice", positions.shape[0]),
              ("valid (corridor / obstacle-free / not in goal)", int(keep.sum()))]

    candidates = np.zeros((positions.shape[0], 6), dtype=np.float32)
    candidates[:, [0, 2, 4]] = positions          # velocities stay exactly zero
    masses, separations = gate_modality(policy, candidates, action_scale, device)
    # Requirement 8 is only reachable where alpha-trimming *averages*, which needs
    # minority samples to survive the trim -- i.e. minor mass strictly above alpha.
    # At or below it, k=1 removes the minority mode entirely and becomes a mode
    # selector, which succeeds and produces no collisions to measure.
    mass_floor = max(args.min_minor_mass, args.alpha + args.grid_step)
    if mass_floor > args.min_minor_mass:
        print(f"note: raising the minor-mass floor from {args.min_minor_mass} to "
              f"{mass_floor:.3f} (just above --alpha {args.alpha}); below it, k=1 "
              f"selects the majority mode instead of averaging, so requirement 8 "
              f"is unreachable by construction.\n")
    keep &= (masses >= mass_floor) & (separations >= args.min_separation)
    stages.append((f"bimodal (mass >= {mass_floor:.3f}, "
                   f"sep >= {args.min_separation})", int(keep.sum())))

    print("candidates surviving each screen:")
    for label, count in stages:
        print(f"  {count:>8}  {label}")

    if not keep.any():
        raise SystemExit(
            "\nNo lattice state is bimodal. Retrain with a different --seed or a "
            "higher --entropy-coef: on a unimodal policy k=1 and k=3 smoothing "
            "provably coincide and the experiment can show nothing."
        )

    # Shortlist spread across the surviving region, so the rollout screen explores
    # distinct positions rather than many neighbours of one point.
    index = np.flatnonzero(keep)
    index = index[np.argsort(-masses[index])]
    shortlist: list[int] = []
    for i in index:
        if all(np.linalg.norm(positions[i] - positions[j]) >= args.shortlist_spacing
               for j in shortlist):
            shortlist.append(int(i))
        if len(shortlist) >= args.shortlist:
            break

    # Raw rollouts first: they are ~100x cheaper than alpha-smoothed ones (no
    # smoother to build per env step), so requirement 7 is used to narrow the
    # field before paying for requirement 8.
    print(f"\nScreening {len(shortlist)} candidates with {args.screen_trajectories} "
          f"raw rollouts each (requirement 7)...")
    started = time.perf_counter()
    screened: list[tuple[float, int]] = []
    for i in shortlist:
        rates, _ = measure_outcomes(
            env, _raw_policy_action_fn(policy, device=device), candidates[i],
            args.screen_trajectories, args.horizon, gamma, args.seed,
        )
        screened.append((rates["goal"], i))
    survivors = [(r, i) for r, i in screened if r >= args.min_raw_goal_rate]
    print(f"  {len(survivors)}/{len(shortlist)} reached goal >= "
          f"{args.min_raw_goal_rate:.0%}  ({time.perf_counter() - started:.0f}s)")

    if not survivors:
        best_rate, best_i = max(screened, default=(0.0, -1))
        raise SystemExit(
            f"\nNo bimodal lattice state reached the goal {args.min_raw_goal_rate:.0%} "
            f"of the time; the best managed {best_rate:.1%} at "
            f"{np.round(positions[best_i], decimals).tolist()}.\n"
            "Lower --min-raw-goal-rate, or retrain: for this policy the gate only "
            "becomes bimodal where the policy is also unsure, and where it is "
            "unsure it fails."
        )

    finalists = [i for _, i in sorted(survivors, reverse=True)[: args.verify_top]]
    print(f"\nMeasuring {len(finalists)} finalists with {args.verify_trajectories} "
          f"raw AND alpha-smoothed rollouts each (requirements 7 + 8)...")
    print("  (the alpha branch rebuilds a smoother every env step, so this is slow)\n")

    print(f"  {'mass':>6}  {'sep':>6}  {'rawgoal':>8}  {'rawret':>8}  "
          f"{'a-coll':>7}  {'a-goal':>7}   position (velocity = 0, z = 0)")
    results: list[tuple[float, int, dict[str, float], dict[str, float], float]] = []
    for i in finalists:
        raw_rates, raw_return, alpha_rates = assess(
            env, policy, stochastic_fn, candidates[i], args, device, gamma,
            args.verify_trajectories,
        )
        passed = (raw_rates["goal"] >= args.min_raw_goal_rate
                  and alpha_rates["collision"] >= args.min_alpha_collision_rate)
        # Rank by the smaller of the two margins, so the winner clears *both*
        # thresholds comfortably rather than excelling on one and scraping the other.
        margin = min(raw_rates["goal"] - args.min_raw_goal_rate,
                     alpha_rates["collision"] - args.min_alpha_collision_rate)
        results.append((margin, i, raw_rates, alpha_rates, raw_return))
        pretty = ", ".join(f"{v:6.1f}" for v in positions[i])
        print(f" {' ' if passed else '*'}{masses[i]:6.3f}  {separations[i]:6.2f}  "
              f"{raw_rates['goal']:8.3f}  {raw_return:+8.2f}  "
              f"{alpha_rates['collision']:7.3f}  {alpha_rates['goal']:7.3f}   [{pretty}]")

    passing = [row for row in results if row[0] >= 0.0]
    if not passing:
        print(f"\n  (* = fails requirement 7 and/or 8)")
        raise SystemExit(
            "\nNo candidate satisfied both the raw goal rate and the alpha collision "
            "rate. The two are in tension: states where alpha-smoothing fails badly "
            "tend to be states the raw policy also struggles with.\n"
            "Options: raise --shortlist / lower --shortlist-spacing to search harder; "
            "relax --min-alpha-collision-rate; or retrain so the gate stays bimodal "
            "in regions the policy has mastered."
        )

    _, best_i, best_raw, best_alpha, best_return = max(passing)
    print(f"\n  {len(passing)}/{len(results)} finalists satisfy every requirement.")
    print(
        f"\nUpdate rollout_io.py:\n\n"
        f"  # minor-mode mass {masses[best_i]:.3f}, separation {separations[best_i]:.2f}\n"
        f"  # raw    {best_raw['goal']:.0%} goal / {best_raw['collision']:.0%} collision, "
        f"mean discounted return {best_return:+.2f}\n"
        f"  # alpha  {best_alpha['goal']:.0%} goal / {best_alpha['collision']:.0%} collision "
        f"-- single-cluster smoothing fails where the raw policy succeeds\n"
        f"  BIMODAL_START_STATE = {as_literal(candidates[best_i])}\n"
    )


if __name__ == "__main__":
    main()
