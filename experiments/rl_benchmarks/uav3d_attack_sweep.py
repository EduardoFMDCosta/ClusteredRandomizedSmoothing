#!/usr/bin/env python3
"""Expected discounted return under a PGD state attack, for the raw policy,
single-cluster alpha-smoothing, and clustered robust smoothing.

This is the whole adversarial experiment. One attack (PGD), one metric
(discounted return), one start state (``BIMODAL_START_STATE``).

**Why discounted return and not a distributional metric.** The previous version
reported Wasserstein-2 between each branch's action distribution and a reference
anchored on the *clean raw policy*. That metric cannot show a defense working:
``W2(raw_clean, raw_clean) = 0``, so the undefended baseline wins at eps=0 by
construction, and clustered smoothing -- whose entire purpose is to differ from
the raw policy in a useful way -- is penalized for doing so. Discounted return is
task-absolute: no branch is privileged, nothing is free at eps=0, and it is
*exactly what the attack minimizes*, so objective and metric coincide.

It is also far higher-powered than the goal/collision rates it replaces. Those
are a coarse categorical readout that saturates: past ~2 sigma every branch is at
~100% collision and the curves are indistinguishable, which is how a five-point
radius sweep ended up with three uninformative points.

**Fairness.** The branches differ only in ``max_num_clusters`` (1 vs
``--max-clusters``). Every other smoothing parameter, the attack budget, the PGD
schedule, the trajectory count and the seeds are shared. Each branch is attacked
white-box against *itself*, since the smoothing guarantee only holds for the
smoothed predictor; the critic supplying the task value is shared, so no branch
faces an easier objective.

**Radius grid.** Expressed and plotted in units of sigma. The certificate is
already vacuous beyond ~2 sigma and every branch has collapsed there, so the grid
is concentrated below sigma where the defense is meant to do something.

**Sharding.** The run is embarrassingly parallel over (radius, seed): one shard
per combination, written to ``--analysis-dir`` as ``shard_r*_s*.json``. Launch
them independently on a cluster, then merge and plot with ``--plot``.

    # one HPC job per shard
    python uav3d_attack_sweep.py --radius-values 0.025 --seeds 0
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
from collect_uav3d_attacked_rollouts import (
    attacked_raw_action_fn,
    attacked_smoothed_action_fn,
)
from collect_uav3d_rollouts import _collect_trajectories, _load_uav3d_policy
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from uav3d_attacks import load_critic

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


# ---------------------------------------------------------------------------
# One shard: (radius, seed) x every branch
# ---------------------------------------------------------------------------
def run_shard(
    radius: float,
    seed: int,
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
    shard: dict[str, Any] = {"radius": float(radius), "seed": int(seed), "branches": {}}

    for branch in BRANCHES:
        # Reseed identically before each branch so all three see the same
        # process-noise and policy-sampling stream at this radius.
        np.random.seed(seed)
        torch.manual_seed(seed)

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
            )

        _, _, counts, _, _, returns = _collect_trajectories(
            env=env, start_state=BIMODAL_START_STATE, horizon=args.horizon,
            num_trajectories=args.num_trajectories, action_fn=action_fn,
            label=f"r={radius:.4f} {branch} seed={seed}", keep_traces=False, gamma=gamma,
        )
        shard["branches"][branch] = {
            "returns": [float(r) for r in returns],
            "counts": dict(counts),
        }
        mean, half_width = mean_and_ci(returns)
        print(f"  r={radius:.4f} seed={seed} {branch:>6}: "
              f"return {mean:+.3f} +/- {half_width:.3f} (95% CI)  {counts}")

    return shard


def shard_path(analysis_dir: Path, radius: float, seed: int) -> Path:
    return analysis_dir / f"shard_r{radius:.4f}_s{seed}.json"


def load_shards(analysis_dir: Path) -> list[dict[str, Any]]:
    shards = []
    for path in sorted(analysis_dir.glob("shard_r*_s*.json")):
        with path.open(encoding="utf-8") as f:
            shards.append(json.load(f))
    if not shards:
        raise SystemExit(
            f"No shards found in {analysis_dir}. Run the sweep before --plot."
        )
    return shards


def aggregate(shards: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    """Pool trajectories across seeds, per (radius, branch).

    Trajectory counts are equal across seeds, so pooling and averaging the
    per-seed means coincide; pooling is used because the CI is then computed on
    the full sample rather than on a handful of seed means.
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
            rows.append({
                "radius": radius, "mean": mean, "ci95": half_width,
                "n": int(returns.size), "counts": counts,
                "collision_rate": counts["collision"] / max(1, sum(counts.values())),
            })
        pooled[branch] = rows
    return radii, pooled


# ---------------------------------------------------------------------------
# Figure 1: the headline
# ---------------------------------------------------------------------------
def plot_return_sweep(
    radii: list[float],
    pooled: dict[str, Any],
    sigma: float,
    output_path: Path,
) -> None:
    """Discounted return vs. attack radius, absolute and relative to clean.

    Panel (a) is the result. Panel (b) subtracts each branch's own clean return so
    the *robustness* advantage is separated from the *clean-performance*
    advantage -- without it a reader cannot tell whether a gap at eps>0 is the
    defense resisting the attack or merely the head start it already had at
    eps=0, and the honest answer is worth stating rather than hiding.

    A difference rather than a retained *fraction*: return is signed here (a
    branch that mostly collides has a negative clean return), and a ratio against
    a negative baseline inverts and is unreadable. The difference is well defined
    either way and pins all three curves to 0 at eps=0 by construction, which is
    what makes the panel legible as "degradation only".
    """
    x = np.array(radii) / sigma
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharex=True)

    total_n = 0
    for branch in BRANCHES:
        style = BRANCH_STYLE[branch]
        rows = pooled[branch]
        mean = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci95"] for r in rows])
        total_n = max(total_n, max(r["n"] for r in rows))

        axes[0].fill_between(x, mean - ci, mean + ci, alpha=0.2, color=style["color"], linewidth=0)
        axes[0].plot(x, mean, marker=style["marker"], linewidth=1.5,
                     color=style["color"], label=style["label"])

        lost = mean - mean[0]
        axes[1].fill_between(x, lost - ci, lost + ci, alpha=0.2,
                             color=style["color"], linewidth=0)
        axes[1].plot(x, lost, marker=style["marker"], linewidth=1.5,
                     color=style["color"], label=style["label"])

    for ax in axes:
        ax.set_xlabel(r"Attack radius $\epsilon / \sigma$")
        ax.grid(True, linestyle=":", alpha=0.35)
        # The smoothing scale: to the right of it the attack is larger than the
        # noise the defense integrates over, and no guarantee survives.
        ax.axvline(1.0, color="0.45", linewidth=0.9, linestyle="--", alpha=0.8)

    axes[0].set_ylabel("Expected discounted return")
    axes[0].set_title("(a) Task performance under attack", fontsize=9)
    axes[1].set_ylabel("Return lost vs. own clean run")
    axes[1].set_title(r"(b) Degradation only (each branch zeroed at $\epsilon=0$)", fontsize=9)
    axes[1].axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    axes[0].legend(loc="best", frameon=True, fontsize=8)

    fig.suptitle(
        f"Discounted return under white-box PGD state attack "
        f"(n={total_n} trajectories/point, band = 95% CI on the mean; "
        rf"dashed line = $\epsilon=\sigma$)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--radius-values", type=str,
                        default="0.0,0.0125,0.025,0.0375,0.05,0.075,0.1",
                        help="Attack radii in normalized state-space L2 units. The "
                             "default grid is 0 to 2*sigma at sigma=0.05, concentrated "
                             "below sigma where the defense can still act.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--num-trajectories", type=int, default=400,
                        help="Trajectories per (branch, radius, seed). With 5 seeds "
                             "this is 2000 per plotted point.")
    parser.add_argument("--horizon", type=int, default=1000)
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
    parser.add_argument("--plot", action="store_true",
                        help="Skip simulation: merge every shard in --analysis-dir "
                             "and render the figure.")
    parser.add_argument("--analysis-dir", type=str,
                        default="experiments/uav_3d/analysis/adversarial")
    parser.add_argument("--figures-dir", type=str,
                        default="experiments/uav_3d/adversarial_figures")
    parser.add_argument("--figure-name", type=str, default="uav3d_attack_return.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    figure_path = Path(args.figures_dir) / args.figure_name

    if not args.plot:
        radii = parse_radius_values(args.radius_values)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
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
        for index, (radius, seed) in enumerate(
            [(r, s) for r in radii for s in seeds], start=1
        ):
            shard = run_shard(
                radius=radius, seed=seed, args=args, policy=policy, critic=critic,
                gamma=gamma, stochastic_fn=stochastic_fn, env=env, device=device,
            )
            shard["config"] = {
                "checkpoint": str(checkpoint_path),
                "num_trajectories": int(args.num_trajectories),
                "horizon": int(args.horizon),
                "sigma": float(args.sigma),
                "alpha": float(args.alpha),
                "max_clusters": int(args.max_clusters),
                "noise_std": float(args.noise_std),
                "pgd_steps": int(args.pgd_steps),
                "pgd_restarts": int(args.pgd_restarts),
                "eot_samples": int(args.eot_samples),
                "gamma": float(gamma),
                "smoothing_N": SMOOTHING_N,
                "confidence_N": CONFIDENCE_N,
                "coverage": COVERAGE,
                "dbscan_eps": DBSCAN_EPS,
                "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
                "start_state": BIMODAL_START_STATE.tolist(),
            }
            path = shard_path(analysis_dir, radius, seed)
            with path.open("w", encoding="utf-8") as f:
                json.dump(shard, f, indent=2)

            elapsed = time.perf_counter() - started
            total = len(radii) * len(seeds)
            print(f"[{index}/{total}] wrote {path} | elapsed {format_duration(elapsed)} "
                  f"| ETA {format_duration(elapsed / index * (total - index))}")

    shards = load_shards(analysis_dir)
    radii, pooled = aggregate(shards)
    sigma = float(shards[0].get("config", {}).get("sigma", args.sigma))
    plot_return_sweep(radii, pooled, sigma, figure_path)

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

    print("\n=== UAV3D attack sweep: discounted return (mean +/- 95% CI) ===")
    print("  eps/sigma  " + "".join(f"{BRANCH_STYLE[b]['plain']:>26}" for b in BRANCHES))
    for i, radius in enumerate(radii):
        cells = "".join(
            f"{pooled[b][i]['mean']:+13.3f} +/-{pooled[b][i]['ci95']:8.3f}" for b in BRANCHES
        )
        print(f"  {radius / sigma:9.2f}  {cells}")
    print(f"  (n={max(pooled[BRANCHES[0]][0]['n'], 1)} trajectories per point)")


if __name__ == "__main__":
    main()
