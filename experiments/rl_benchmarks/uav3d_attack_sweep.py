#!/usr/bin/env python3
"""Reach-avoid outcome under a PGD state attack, for the raw policy,
single-cluster alpha-smoothing, and clustered robust smoothing.

This is the whole adversarial experiment. One attack (PGD), one metric (goal and
collision rate), one start state (``BIMODAL_START_STATE``).

**Why outcome rates and not discounted return.** Goal and collision rate *are*
the specification: the task is reach-avoid, and the return is a shaped training
signal towards it, not the objective itself. Its dense ``0.5 * progress - 0.01``
term additionally rewards short paths, which no part of the specification asks
for -- a controller that reaches the goal slowly but never collides is a success,
and return would rank it below a faster one. Return is still recorded, and shown
in the console summary as a diagnostic: PGD minimizes it, so a flat return curve
is evidence the attack is not biting.

Reporting something other than the attacked quantity is also the more
defensible choice, since it forecloses "you measured exactly what you optimized".

Two arguments once made for return no longer apply. Rates were said to saturate
-- true on the old 0-to-8-sigma grid where every branch sat at ~100% collision
past 2 sigma, but the grid now stops at 1 sigma precisely to avoid that. And
rates were said to be low-powered -- at n=2000 the binomial standard error at
p=0.5 is 0.011, which is not a power problem.

**Why not a distributional metric.** An earlier version reported Wasserstein-2
between each branch's action distribution and a reference anchored on the *clean
raw policy*. That metric cannot show a defense working: ``W2(raw_clean,
raw_clean) = 0``, so the undefended baseline wins at eps=0 by construction, and
clustered smoothing -- whose entire purpose is to differ from the raw policy in a
useful way -- is penalized for doing so.

**Fairness.** The branches differ only in ``max_num_clusters`` (1 vs
``--max-clusters``). Every other smoothing parameter, the attack budget, the PGD
schedule, the trajectory count and the seed are shared. Each branch is attacked
white-box against *itself*, since the smoothing guarantee only holds for the
smoothed predictor; the critic supplying the task value is shared, so no branch
faces an easier objective.

**Replication.** ``--num-trajectories`` is the only sample-size knob, because the
trajectory is the only replicate unit there is: the policy, env, start state,
defense parameters and radius are all fixed, so every trajectory is an i.i.d.
draw from one distribution and the CI is computed over the whole set. 

``--seed`` is reset before every (branch, radius) pair, which fixes the stream
all three branches start from at a radius and gives the sweep common random
numbers along the x-axis.

**Radius grid.** Expressed and plotted in units of sigma. The certificate is
already vacuous beyond ~2 sigma and every branch has collapsed there, so the grid
is concentrated below sigma where the defense is meant to do something.

**Sharding.** The run is embarrassingly parallel over radius: one shard per
radius, written to ``--analysis-dir`` as ``shard_r*.json``. Launch them
independently on a cluster, then merge and plot with ``--plot``.

    # one HPC job per radius
    python uav3d_attack_sweep.py --radius-values 0.025
    ...
    # after all shards land
    python uav3d_attack_sweep.py --plot
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collect_uav3d_attacked_rollouts import (
    attacked_raw_action_fn,
    attacked_smoothed_action_fn,
)
from collect_uav3d_rollouts import _collect_trajectories, _load_uav3d_policy
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import normalize_state
from uav3d_attacks import _GATE_TEMPERATURE, load_critic

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)

# ``label`` is LaTeX for figures; ``plain`` is for the console summary.
BRANCH_STYLE = {
    "raw": {"color": "#CC3311", "marker": "o",
            "label": "Raw policy", "plain": "raw"},
    "alpha": {"color": "#EE7733", "marker": "^",
              "label": r"$\alpha$-smoothing ($k=1$)", "plain": "alpha (k=1)"},
    "robust": {"color": "#0077BB", "marker": "s",
               "label": "Clustered robust smoothing", "plain": "robust (clustered)"},
}
BRANCHES = ["raw", "alpha", "robust"]

# Defense parameters. These describe the *defense*, not the experiment, so they
# are fixed here rather than exposed as knobs -- they match
# ``collect_uav3d_rollouts.py``'s defaults, and the clean and attacked runs have
# to use the same defense for their numbers to be comparable at all.
SMOOTHING_N = 30
CONFIDENCE_N = 4000
COVERAGE = 0.9
DBSCAN_EPS = 0.45
DBSCAN_MIN_SAMPLES = 50

# Radii and sigma live in normalized (Phi-space) units, which is what makes
# eps/sigma an exact like-for-like ratio on every axis. Phi divides each axis by
# its natural scale -- corridor half-extent 15/9/7 m for x/y/z position, the
# +/-7 m/s speed limit for the velocities -- so a normalized radius is a
# *fraction of that axis's scale*, and quoting it in metres needs a nominated
# axis. x-position is the reference here; other axes scale by their own divisor.
# Derived from ``normalize_state`` rather than restated, so the two cannot drift.
REFERENCE_AXIS = "x-position"
REFERENCE_AXIS_SCALE = float(1.0 / normalize_state(np.ones(6, dtype=np.float32))[0])


def in_reference_units(normalized: float) -> float:
    """A normalized radius as metres along the reference axis."""
    return normalized * REFERENCE_AXIS_SCALE


def defense_parameters(max_clusters: int, alpha: float) -> Parameters:
    return Parameters(
        smoothing=SmoothingParameters(N=SMOOTHING_N, alpha=alpha),
        clustering=ClusteringParameters(
            max_num_clusters=max_clusters, coverage=COVERAGE,
            dbscan_eps=DBSCAN_EPS, dbscan_min_samples=DBSCAN_MIN_SAMPLES,
        ),
        confidence=ConfidenceParameters(num_samples=CONFIDENCE_N, beta=1e-2),
    )


def format_duration(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def parse_radius_values(text: str) -> list[float]:
    radii = [float(v) for v in (item.strip() for item in text.split(",")) if v]
    if not radii:
        raise ValueError("--radius-values must contain at least one value")
    if any(r < 0.0 for r in radii):
        raise ValueError("all radius values must be non-negative")
    return radii


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mean_and_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation CI **on the mean**.

    Deliberately not the ``+/- 1 s.d.`` band the previous figures used. That band
    described the spread across trajectories, which does not shrink with more
    data, so it stayed enormous and fully overlapping even where the underlying
    effect was overwhelming -- the figures looked null when they were not. The
    standard error is what answers "do these branches differ", and with the
    trajectory counts here the CLT approximation is comfortable.
    """
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), 0.0
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    return float(values.mean()), float(z * values.std(ddof=1) / np.sqrt(n))


def proportion_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """``(rate, lo, hi)`` -- Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation ``p +/- z*sqrt(p(1-p)/n)``
    because the interesting radii drive these rates towards 0 and 1, where the
    normal interval is both too narrow and runs outside [0, 1] -- it would claim
    a negative collision rate at eps=0. Wilson stays inside the unit interval and
    is asymmetric near the ends, which is the honest shape.
    """
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = (z / denominator) * np.sqrt(
        rate * (1.0 - rate) / total + z * z / (4 * total * total)
    )
    return rate, float(max(0.0, center - half)), float(min(1.0, center + half))


# ---------------------------------------------------------------------------
# One shard: one radius x every branch
# ---------------------------------------------------------------------------
def run_shard(
    radius: float,
    args: argparse.Namespace,
    policy,
    critic,
    gamma: float,
    stochastic_fn: UAV3DPolicyStochasticFunction,
    env: UAV3D,
    device: torch.device,
) -> dict[str, Any]:
    parameters = {
        "alpha": defense_parameters(1, args.alpha),
        "robust": defense_parameters(args.max_clusters, args.alpha),
    }
    shard: dict[str, Any] = {"radius": float(radius), "seed": int(args.seed), "branches": {}}

    for branch in BRANCHES:
        # Reset to the same seed before every (branch, radius) pair. Two effects,
        # both wanted:
        #   * across branches at one radius, all three start from an identical
        #     process-noise and policy-sampling stream;
        #   * across radii for one branch, the sweep uses common random numbers,
        #     so the *shape* of the curve is not confounded by fresh noise at
        #     every point.
        # The coupling is nominal rather than exact once the branches diverge --
        # the smoothed branches draw thousands of samples per env step against
        # the raw branch's handful, so the streams decorrelate after the first
        # action. The CIs below are computed per branch and claim no pairing.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if branch == "raw":
            action_fn = attacked_raw_action_fn(
                policy=policy, env=env, critic=critic, gamma=gamma, radius=radius,
                pgd_steps=args.pgd_steps, pgd_restarts=args.pgd_restarts,
                eot_samples=args.eot_samples, device=device,
            )
        else:
            action_fn = attacked_smoothed_action_fn(
                policy=policy, stochastic_fn=stochastic_fn, env=env, critic=critic,
                gamma=gamma, parameters=parameters[branch], sigma=args.sigma,
                radius=radius, pgd_steps=args.pgd_steps, pgd_restarts=args.pgd_restarts,
                eot_samples=args.eot_samples, device=device,
                gate_temperature=args.gate_temperature,
                partition_temperature=args.rho_temperature,
            )

        _, _, counts, _, _, returns = _collect_trajectories(
            env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
            num_trajectories=args.num_trajectories, action_fn=action_fn,
            label=f"r={radius:.4f} {branch}", keep_traces=False, gamma=gamma,
        )
        shard["branches"][branch] = {
            "returns": [float(r) for r in returns],
            "counts": dict(counts),
        }
        mean, half_width = mean_and_ci(returns)
        print(f"  r={radius:.4f} {branch:>6}: "
              f"return {mean:+.3f} +/- {half_width:.3f} (95% CI)  {counts}")

    return shard


def shard_path(analysis_dir: Path, radius: float) -> Path:
    return analysis_dir / f"shard_r{radius:.4f}.json"


def load_shards(analysis_dir: Path) -> list[dict[str, Any]]:
    shards = []
    for path in sorted(analysis_dir.glob("shard_r*.json")):
        with path.open(encoding="utf-8") as f:
            shards.append(json.load(f))
    if not shards:
        raise SystemExit(
            f"No shards found in {analysis_dir}. Run the sweep before --plot."
        )
    return shards


def aggregate(shards: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    """Pool trajectories per (radius, branch) across shards.

    One shard per radius, so normally this is a straight read-through; the
    concatenation matters only if a radius was run in several pieces. The CI is
    computed on the full trajectory sample -- the trajectories *are* the
    replicates, and there is no second level of variation to model, since
    nothing about the experiment configuration varies between them.
    """
    radii = sorted({float(s["radius"]) for s in shards})
    pooled: dict[str, Any] = {}
    for branch in BRANCHES:
        rows = []
        for radius in radii:
            returns = np.concatenate([
                np.asarray(s["branches"][branch]["returns"], dtype=float)
                for s in shards
                if float(s["radius"]) == radius and branch in s["branches"]
            ] or [np.array([])])
            counts: dict[str, int] = {"goal": 0, "collision": 0, "timeout": 0}
            for s in shards:
                if float(s["radius"]) == radius and branch in s["branches"]:
                    for outcome, value in s["branches"][branch]["counts"].items():
                        counts[outcome] += value
            mean, half_width = mean_and_ci(returns)
            total = sum(counts.values())
            row = {
                "radius": radius, "n": int(returns.size), "counts": counts,
                "total": total,
                # Return is retained as a diagnostic -- it is what PGD minimizes,
                # so a flat return curve means the attack is not biting -- but it
                # is not the reported result. See the module docstring.
                "return_mean": mean, "return_ci95": half_width,
            }
            for outcome in ("goal", "collision", "timeout"):
                rate, lo, hi = proportion_ci(counts[outcome], total)
                row[outcome] = {
                    "rate": rate, "lo": lo, "hi": hi,
                }
            rows.append(row)
        pooled[branch] = rows
    return radii, pooled


# ---------------------------------------------------------------------------
# Figure 1: the headline
# ---------------------------------------------------------------------------
def plot_outcome_sweep(
    radii: list[float],
    pooled: dict[str, Any],
    sigma: float,
    output_path: Path,
) -> None:
    """Reach-avoid outcome rates vs. attack radius.

    The two panels are the reach-avoid specification itself: reach the goal (a),
    avoid the obstacles (b). They do not sum to one -- a trajectory that runs out
    of horizon without either is a timeout, reported in the console table and the
    JSON, and it is a distinct kind of failure from a collision.

    Two bands, answering different questions:

      * filled  -- Wilson 95% CI on the *rate*. How well the rate is pinned down;
        shrinks as ``--num-trajectories`` grows. This is what decides whether two
        branches differ. Asymmetric, and pinches near 0 and 1 rather than running
        off the axis.
      * dashed  -- ``rate +/- 1 s.d.`` of the per-trajectory 0/1 indicator,
        ``sqrt(p(1-p))``. How much a single trajectory varies; does *not* shrink
        with more data.

    The s.d. is a deterministic function of the rate for a Bernoulli outcome, so
    it carries no information the curve does not already -- it is drawn as a
    reminder that individual trajectories are 0/1 events, and is widest exactly
    where the rate is near 0.5. It is emphatically not an error bar: reading the
    overlap of the dashed envelopes as "no difference" is the mistake that made
    the previous figure set look null when the effects were overwhelming.
    """
    x = np.array(radii) / sigma
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharex=True)

    total_n = 0
    panels = ((axes[0], "goal", "Goal-reach rate", "(a) Reach: fraction arriving at the goal"),
              (axes[1], "collision", "Collision rate", "(b) Avoid: fraction hitting an obstacle"))

    for ax, outcome, ylabel, title in panels:
        for branch in BRANCHES:
            style = BRANCH_STYLE[branch]
            rows = pooled[branch]
            rate = np.array([r[outcome]["rate"] for r in rows])
            lo = np.array([r[outcome]["lo"] for r in rows])
            hi = np.array([r[outcome]["hi"] for r in rows])
            total_n = max(total_n, max(r["total"] for r in rows))

            ax.fill_between(x, lo, hi, alpha=0.2, color=style["color"], linewidth=0)
            ax.plot(x, rate, marker=style["marker"], linewidth=1.5,
                    color=style["color"], label=style["label"])

        ax.set_xlabel(r"Attack radius $\epsilon / \sigma$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.35)

    # Band semantics belong in a legend, not a caption: two visually different
    # envelopes that mean different things are exactly the thing a reader
    # misreads. Proxy artists keep it independent of the branch entries.
    band_handles = [
        Patch(facecolor="0.45", alpha=0.25, label="95% CI on the rate"),
    ]
    axes[0].legend(loc="best", frameon=True, fontsize=8)
    axes[1].legend(handles=band_handles, loc="best", frameon=True, fontsize=7)

    fig.suptitle(
        f"Reach-avoid outcome under white-box PGD state attack "
        rf"(n={total_n} trajectories/point)"
        "\n"
        rf"$\epsilon$ and $\sigma$ are normalized; $\sigma={sigma:.4g}$ is "
        rf"{in_reference_units(sigma):.3g} m of {REFERENCE_AXIS}, "
        "other axes scale by their own divisor",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--radius-values", type=str,
                        default="0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05",
                        help="Attack radii in normalized state-space L2 units.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG state, reset before every (branch, radius) pair.")
    parser.add_argument("--num-trajectories", type=int, default=1000,
                        help="Trajectories per (branch, radius) -- the sample the "
                             "plotted mean and its CI are computed from. The "
                             "trajectories are the replicates: nothing about the "
                             "experiment varies between them, so more trajectories "
                             "is exactly what tightens the interval.")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha-trim fraction, shared by both smoothed branches. "
                             "Must be BELOW the state's minor-mode mass: alpha-trimming "
                             "averages the modes only while some minority samples "
                             "survive the trim. At alpha >= minor mass it removes the "
                             "minority mode outright and k=1 silently becomes a mode "
                             "*selector*, which does not fail and is not the baseline "
                             "the paper is comparing against.")
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pgd-restarts", type=int, default=3)
    parser.add_argument("--eot-samples", type=int, default=64)
    parser.add_argument("--gate-temperature", type=float, default=_GATE_TEMPERATURE,
                        help="Gumbel-softmax temperature for the policy gate in the "
                             "smoothed branches' attack objective. Straight-through, "
                             "so it leaves the objective's value untouched and only "
                             "shapes the gradient.")
    parser.add_argument("--rho-temperature", type=float, default=None,
                        help="Softmax temperature for the region-occupancy mixture "
                             "weights; also straight-through and value-neutral. "
                             "Default derives it from the clean anchor's own region "
                             "distances.")
    parser.add_argument("--plot", action="store_true",
                        help="Skip simulation: merge every shard in --analysis-dir "
                             "and render the figure.")
    parser.add_argument("--analysis-dir", type=str,
                        default="experiments/uav_3d/analysis/adversarial")
    parser.add_argument("--figures-dir", type=str,
                        default="experiments/uav_3d/adversarial_figures")
    parser.add_argument("--figure-name", type=str, default="uav3d_attack_outcomes.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    figure_path = Path(args.figures_dir) / args.figure_name

    if not args.plot:
        radii = parse_radius_values(args.radius_values)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint_path = Path(args.checkpoint)
        policy, action_scale = _load_uav3d_policy(checkpoint_path)
        policy.to(device)
        critic, gamma = load_critic(checkpoint_path, device)
        print(f"Loaded policy (action_scale={action_scale}) and critic (gamma={gamma}) "
              f"from {checkpoint_path} onto {device}.")

        stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
        env = UAV3D(gaussian_noise=True, noise_std=args.noise_std, max_steps=None)

        analysis_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for index, radius in enumerate(radii, start=1):
            shard = run_shard(
                radius=radius, args=args, policy=policy, critic=critic,
                gamma=gamma, stochastic_fn=stochastic_fn, env=env, device=device,
            )
            shard["config"] = {
                "checkpoint": str(checkpoint_path),
                "num_trajectories": int(args.num_trajectories),
                "seed": int(args.seed),
                "horizon": int(args.horizon),
                "sigma": float(args.sigma),
                "alpha": float(args.alpha),
                "max_clusters": int(args.max_clusters),
                "noise_std": float(args.noise_std),
                "pgd_steps": int(args.pgd_steps),
                "pgd_restarts": int(args.pgd_restarts),
                "eot_samples": int(args.eot_samples),
                "gate_temperature": float(args.gate_temperature),
                "rho_temperature": (None if args.rho_temperature is None
                                    else float(args.rho_temperature)),
                "gamma": float(gamma),
                "smoothing_N": SMOOTHING_N,
                "confidence_N": CONFIDENCE_N,
                "coverage": COVERAGE,
                "dbscan_eps": DBSCAN_EPS,
                "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
                "start_state": BIMODAL_START_STATE.tolist(),
                "reference_axis": REFERENCE_AXIS,
                "reference_axis_scale": REFERENCE_AXIS_SCALE,
                "radius_metres": in_reference_units(radius),
                "sigma_metres": in_reference_units(float(args.sigma)),
            }
            path = shard_path(analysis_dir, radius)
            with path.open("w", encoding="utf-8") as f:
                json.dump(shard, f, indent=2)

            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(radii)}] wrote {path} | elapsed {format_duration(elapsed)} "
                  f"| ETA {format_duration(elapsed / index * (len(radii) - index))}")

    shards = load_shards(analysis_dir)
    radii, pooled = aggregate(shards)
    sigma = float(shards[0].get("config", {}).get("sigma", args.sigma))
    plot_outcome_sweep(radii, pooled, sigma, figure_path)

    summary_path = analysis_dir / "uav3d_attack_sweep.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"config": shards[0].get("config", {}),
             "num_shards": len(shards),
             "radii": radii,
             "per_branch": pooled},
            f, indent=2,
        )
    print(f"saved: {summary_path}")

    for outcome, title in (("goal", "goal-reach rate"), ("collision", "collision rate")):
        print(f"\n=== UAV3D attack sweep: {title} (95% Wilson CI) ===")
        print(f"  (sigma = {sigma:.4g} normalized = {in_reference_units(sigma):.3g} m "
              f"of {REFERENCE_AXIS}; other axes scale by their own divisor)")
        print("  eps/sigma  " + "".join(f"{BRANCH_STYLE[b]['plain']:>26}" for b in BRANCHES))
        for i, radius in enumerate(radii):
            cells = "".join(
                f"{pooled[b][i][outcome]['rate']:8.3f} "
                f"[{pooled[b][i][outcome]['lo']:.2f},{pooled[b][i][outcome]['hi']:.2f}]"
                for b in BRANCHES
            )
            print(f"  {radius / sigma:9.2f}  {cells}")

    timeouts = {b: sum(r["counts"]["timeout"] for r in pooled[b]) for b in BRANCHES}
    if any(timeouts.values()):
        print(f"\n  timeouts (neither goal nor collision within --horizon): {timeouts}")

    print("\n  discounted return, diagnostic only -- this is what PGD minimizes, so a")
    print("  flat curve here means the attack is not biting:")
    print("  eps/sigma  " + "".join(f"{BRANCH_STYLE[b]['plain']:>26}" for b in BRANCHES))
    for i, radius in enumerate(radii):
        cells = "".join(
            f"{pooled[b][i]['return_mean']:+13.3f} +/-{pooled[b][i]['return_ci95']:8.3f}"
            for b in BRANCHES
        )
        print(f"  {radius / sigma:9.2f}  {cells}")
    print(f"\n  (n={max(pooled[BRANCHES[0]][0]['total'], 1)} trajectories per point)")


if __name__ == "__main__":
    main()
