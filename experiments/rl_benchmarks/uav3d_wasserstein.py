#!/usr/bin/env python3
"""How much of the raw policy's action distribution survives smoothing, measured
across the state space.

One number per state: ``W2(h_w(x), branch(x))``, the empirical 2-Wasserstein
distance in action space between the raw stochastic policy at the *unperturbed*
state and the smoothed predictor built at that state. Two branches, one
histogram panel each -- single-cluster alpha-smoothing (``k=1``) and clustered
alpha-smoothing (``k = --max-clusters``) -- sharing bins so the two panels are
read against each other.

**Why this is not the attacked metric.** ``uav3d_attack_sweep.py`` deliberately
does *not* report a distance to the clean raw policy, because under attack that
metric hands the win to the undefended baseline by construction. Here there is
no attack and no winner: the reference is the raw predictive distribution, and
the quantity being measured is *distortion* -- how far each defense moves the
action distribution away from the model's own. It is descriptive, not a score.
A branch is not "better" for being closer to ``h_w(x)``; the raw policy is
trivially at zero and is not robust.

**Why the distance is not near zero for either branch.** A smoothed draw is an
alpha-trimmed mean of ``N`` noisy samples, so it concentrates by roughly
``1/sqrt(N)`` relative to the raw distribution even when nothing else happens.
That concentration is a floor both branches pay, and it is the whole story on a
unimodal state. The branches separate only where ``h_w(x)`` has more than one
mode: ``k=1`` averages the modes into a single atom somewhere between them and
pays the mode separation on top of the floor, while clustered smoothing keeps an
atom per mode and stays near the floor. So the interesting comparison is not the
overall histogram but the multi-modal subpopulation within it.

**Modality is read off the policy's own gate, never off a smoother's
clustering.** A state counts as multi-modal when the mixture gate puts at least
``--min-minor-mass`` on its least-likely component *and* the tanh-squashed
component means are at least ``--min-separation`` apart -- the same two criteria
``find_bimodal_state.py`` screens with, imported from it rather than restated.
Classifying with the clustering that one of the two branches is built on would
make the comparison circular.

**Sampling.** States are drawn i.i.d. uniformly from the free state space
(inside the corridor, outside every obstacle, outside the goal box), so every
histogram is a marginal over the state space and the multi-modal share of it is
a measured quantity rather than a chosen one. Velocities are zero by default --
``--velocity-max`` gives them a uniform range instead -- matching the convention
of the other UAV3D experiments, where momentum the experiment did not choose is
a confounder.

Multi-modality is rare under that sample: on the shipped checkpoint roughly one
state in twenty qualifies. The multi-modal panels are therefore built from a
small fraction of ``--num-states``, and the fix is to draw more states, not to
reweight them -- a stratified sample would make the panels a conditional
comparison and the counts meaningless. Each state costs about a tenth of a
second, so the default is set high enough to fill the panels.

    # compute the distances and render the figure, caching to JSON
    python uav3d_wasserstein.py

    # re-render the figure from the cached JSON
    python uav3d_wasserstein.py --plot
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from collect_uav3d_rollouts import _load_uav3d_policy
from dynamics import UAV3D
from find_bimodal_state import (
    MIN_MINOR_MASS,
    MIN_SEPARATION,
    gate_modality,
    valid_positions,
)
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import normalize_state
from uav3d_attack_sweep import BRANCH_STYLE, defense_parameters

from randomized_smoothing.dataclasses import RuntimeHeuristics
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

# Branch hues are the ones the rest of the UAV3D figures already use, so a
# reader carries the same colour association across figures. Within a panel the
# two modality groups are a light and a full step of that one hue (OKLab dE 18
# and 28 respectively, separated almost entirely in lightness, which CVD
# preserves), plus a hatch on the multi-modal segment for print and
# forced-colours.
MODALITY_TINT = {"alpha": "#F6C3A0", "robust": "#9FCBE0"}
MULTIMODAL_HATCH = "///"

# Corridor bounds, matching ``find_bimodal_state.valid_positions`` -- which is
# the validity predicate this samples against, so the two must agree.
POSITION_LOWER = np.array([-15.0, -9.0, -7.0])
POSITION_UPPER = np.array([15.0, 9.0, 7.0])


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
def sample_free_states(
    env: UAV3D,
    num_states: int,
    velocity_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``num_states`` i.i.d. uniform draws from the free state space, ``(n, 6)``.

    Rejection sampling against the env's own validity predicate rather than a
    hand-carved box, so "free" means exactly what it means during a rollout.
    """
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if velocity_max < 0.0:
        raise ValueError("velocity-max must be non-negative")

    kept: list[np.ndarray] = []
    total = 0
    while total < num_states:
        draw = max(num_states - total, 64) * 2
        positions = rng.uniform(POSITION_LOWER, POSITION_UPPER, size=(draw, 3))
        positions = positions[valid_positions(env, positions)]
        if positions.size:
            kept.append(positions)
            total += positions.shape[0]

    positions = np.concatenate(kept, axis=0)[:num_states]
    states = np.zeros((num_states, 6), dtype=np.float32)
    states[:, [0, 2, 4]] = positions
    if velocity_max > 0.0:
        states[:, [1, 3, 5]] = rng.uniform(
            -velocity_max, velocity_max, size=(num_states, 3)
        )
    return states


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def wasserstein_2(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    ot_method: str,
    reg: float,
) -> float:
    """Empirical 2-Wasserstein distance between two equal-dimension clouds.

    Same quantity, same solver options as
    ``experiments/paper/trajflow_wasserstein.py``.

    **Prefer ``--ot-method emd`` here.** The clouds are 3-D and a few hundred
    points, so the exact network simplex costs ~0.02 s per pair, against 0.1-6 s
    for ``sinkhorn_log`` at ``reg=0.1``. The entropic bias is the bigger problem:
    measured against exact OT on this data it inflates small distances badly and
    large ones hardly at all (+97%, +45%, +20% on the three smallest pairs of a
    sample batch; +0.2% to +2.7% on the largest). Small distances are the
    unimodal floor and large ones are the mode-collapse cases, so the
    regularized solver compresses precisely the gap these panels exist to show.
    """
    if x_samples.ndim != 2 or y_samples.ndim != 2:
        raise ValueError("Both sample sets must be 2-D arrays of shape (n, d).")
    if x_samples.shape[1] != y_samples.shape[1]:
        raise ValueError(
            "Both sample sets must have the same feature dimension; "
            f"got {x_samples.shape[1]} and {y_samples.shape[1]}"
        )

    a = np.full(x_samples.shape[0], 1.0 / x_samples.shape[0], dtype=np.float64)
    b = np.full(y_samples.shape[0], 1.0 / y_samples.shape[0], dtype=np.float64)
    cost_matrix = ot.dist(
        x_samples.astype(np.float64, copy=False),
        y_samples.astype(np.float64, copy=False),
        metric="sqeuclidean",
    )

    if ot_method == "emd":
        w2_squared = float(ot.emd2(a, b, cost_matrix, numItermax=1_000_000))
    else:
        if reg <= 0.0:
            raise ValueError("sinkhorn regularization (reg) must be positive")
        w2_squared = float(
            ot.sinkhorn2(
                a, b, cost_matrix, reg=reg, method=ot_method,
                numItermax=5000, stopThr=1e-9,
            )
        )

    return float(math.sqrt(max(w2_squared, 0.0)))


def build_smoother(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    x: torch.Tensor,
    max_clusters: int,
    alpha: float,
    sigma: float,
    noise_sub_batch_size: int,
) -> GaussianAlphaTrimmingSmoothing:
    """A smoother over a *batch* of states, one cluster set per row.

    ``defense_parameters`` supplies every smoothing knob, so this branch is the
    same defense the rollout experiments run; only ``max_num_clusters`` differs
    between the two branches here.
    """
    parameters = defense_parameters(max_clusters, alpha)
    parameters.runtime = RuntimeHeuristics(noise_sub_batch_size=noise_sub_batch_size)
    return GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn, x=x, sigma=sigma, parameters=parameters,
    )


def raw_samples(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    x: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """``h_w(x)`` at the unperturbed states: ``(batch, num_samples, action_dim)``.

    The reference distribution. No input noise -- that is what makes it the raw
    policy rather than the k=1 smoother's input cloud.
    """
    repeated = x.unsqueeze(1).expand(x.shape[0], num_samples, x.shape[-1])
    with torch.no_grad():
        return stochastic_fn(repeated)


def evaluate_batch(
    stochastic_fn: UAV3DPolicyStochasticFunction,
    states: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    """One row per state: both distances plus the diagnostics behind them."""
    normalized = np.stack([normalize_state(s) for s in states])
    x = torch.from_numpy(normalized).to(device=device, dtype=torch.float32)

    reference = raw_samples(stochastic_fn, x, args.distribution_samples)

    rows: list[dict[str, Any]] = []
    branch_samples: dict[str, np.ndarray] = {}
    k_used: dict[str, np.ndarray] = {}
    for branch, max_clusters in (("alpha", 1), ("robust", args.max_clusters)):
        smoother = build_smoother(
            stochastic_fn=stochastic_fn, x=x, max_clusters=max_clusters,
            alpha=args.alpha, sigma=args.sigma,
            noise_sub_batch_size=args.noise_sub_batch_size,
        )
        draws = smoother.monte_carlo(num_simulations=args.distribution_samples)
        branch_samples[branch] = draws.detach().cpu().numpy()
        k_used[branch] = smoother.cluster.k_per_row.detach().cpu().numpy()

    reference_np = reference.detach().cpu().numpy()
    for i, state in enumerate(states):
        row: dict[str, Any] = {"state": [float(v) for v in state]}
        for branch in ("alpha", "robust"):
            row[f"w2_{branch}"] = wasserstein_2(
                reference_np[i], branch_samples[branch][i],
                ot_method=args.ot_method, reg=args.sinkhorn_reg,
            )
            row[f"k_used_{branch}"] = int(k_used[branch][i])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_wasserstein_histograms(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
    bins: int,
    split_modality: bool,
) -> None:
    """One panel per (modality group, branch), on shared bins.

    Shared bin edges across every panel are the point of the figure: a bar at
    the same x has to mean the same distance everywhere, or the panels cannot be
    read against one another.

    The y axis is shared *within a row* and not across rows. Across a row the
    counts are two views of the same states, so a common scale is what makes
    the branches comparable; across rows the group sizes differ by an order of
    magnitude under a uniform sample, and a common scale would flatten the
    multi-modal row into invisibility. Each panel's own ``n`` is in its legend.
    """
    values = {b: np.array([r[f"w2_{b}"] for r in rows], dtype=float)
              for b in ("alpha", "robust")}
    multimodal = np.array([bool(r["multimodal"]) for r in rows])

    combined = np.concatenate([values["alpha"], values["robust"]])
    low, high = float(combined.min()), float(combined.max())
    if math.isclose(low, high):
        pad = 1e-6 if abs(low) < 1e-3 else abs(low) * 1e-3
        low, high = low - pad, high + pad
    edges = np.linspace(low, high, bins + 1)

    branches = (
        ("alpha", "$\\alpha$-smoothing ($k=1$)"),
        ("robust", f"Clustered smoothing ($k \\leq {config['max_clusters']}$)"),
    )
    if split_modality:
        groups = [("Unimodal states", ~multimodal, False),
                  ("Multi-modal states", multimodal, True)]
    else:
        groups = [("All states", np.ones_like(multimodal), False)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(10.0, 3.4 * len(groups)),
        sharex=True, sharey="row", squeeze=False,
    )

    panel_letters = iter("abcdefgh")
    for row_id, (group_name, mask, hatched) in enumerate(groups):
        for col_id, (branch, branch_label) in enumerate(branches):
            ax = axes[row_id][col_id]
            color = BRANCH_STYLE[branch]["color"]
            selected = values[branch][mask]
            # The mean rides in the panel's own legend entry rather than in a
            # second one for the line, so the legend has one row per thing shown.
            label = f"n={selected.size}"
            if selected.size:
                label += f", mean {selected.mean():.2f}"
                # White separators give a surface gap between neighbouring bars,
                # so a dense run does not read as one solid block.
                _, _, patches = ax.hist(
                    selected, bins=edges, color=color, label=label,
                    edgecolor="white", linewidth=0.6,
                )
                if hatched:
                    # Redundant with the row position, and the encoding that
                    # survives greyscale printing and forced-colours.
                    for patch in patches:
                        patch.set_hatch(MULTIMODAL_HATCH)
                ax.axvline(float(selected.mean()), color="#202020", linestyle="--",
                           linewidth=1.4)
            else:
                ax.hist([], bins=edges, color=color, label=label)

            ax.set_title(f"({next(panel_letters)}) {branch_label}"
                         f" -- {group_name.lower()}", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", frameon=True, fontsize=8, framealpha=0.95)
            if row_id == len(groups) - 1:
                ax.set_xlabel("$W_2$ to $h_w(x)$ (action units)")
        axes[row_id][0].set_ylabel(f"{group_name}\n(count)")

    # fig.suptitle(
    #     "Distortion of the action distribution: $W_2$ from the raw policy "
    #     f"$h_w(x)$ to each smoothed branch\n"
    #     f"{len(rows)} states drawn uniformly from the free state space; "
    #     rf"$\sigma={config['sigma']:.4g}$, $N={config['smoothing_N']}$, "
    #     rf"$\alpha={config['alpha']:.2g}$, "
    #     f"{config['distribution_samples']} draws per distribution",
    #     fontsize=9,
    # )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"saved: {output_path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    multimodal = np.array([bool(r["multimodal"]) for r in rows])
    groups = (("all states", np.ones_like(multimodal)),
              ("uni-modal", ~multimodal),
              ("multi-modal", multimodal))

    print("\n=== W2 to the raw policy h_w(x) ===")
    print(f"  {'group':>12}  {'n':>5}  {'alpha (k=1)':>18}  {'clustered':>18}")
    for label, mask in groups:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label:>12}  {n:>5}  {'--':>18}  {'--':>18}")
            continue
        cells = []
        for branch in ("alpha", "robust"):
            v = np.array([r[f"w2_{branch}"] for r in rows], dtype=float)[mask]
            cells.append(f"{v.mean():8.3f} +/- {v.std():6.3f}")
        print(f"  {label:>12}  {n:>5}  {cells[0]:>18}  {cells[1]:>18}")

    k = np.array([r["k_used_robust"] for r in rows])
    print(f"\n  clusters found by the clustered branch: "
          f"{ {int(v): int((k == v).sum()) for v in np.unique(k)} }")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--analysis-dir", type=str,
                        default="experiments/uav_3d/analysis/wasserstein")
    parser.add_argument("--figures-dir", type=str,
                        default="experiments/uav_3d/wasserstein_figures")
    parser.add_argument("--run-name", type=str, default="uav3d_wasserstein",
                        help="Basename of both the JSON and the figure.")
    parser.add_argument("--plot", action="store_true",
                        help="Re-render the figure from the cached JSON without recomputing.")

    parser.add_argument("--num-states", type=int, default=10000,
                        help="States evaluated. One histogram sample per state. The "
                             "multi-modal panels get whatever fraction of these "
                             "qualifies -- about one in twenty -- so this is the knob "
                             "that fills them.")
    parser.add_argument("--velocity-max", type=float, default=0.0,
                        help="Velocities are sampled uniformly in [-v, v]; 0 pins "
                             "them at rest, as in the other UAV3D experiments.")

    parser.add_argument("--distribution-samples", type=int, default=500,
                        help="Draws from each distribution per state.")
    parser.add_argument("--state-batch-size", type=int, default=8,
                        help="States smoothed at once.")
    parser.add_argument("--noise-sub-batch-size", type=int, default=1500)
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Smoothing noise std, normalized state units.")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha-trim fraction, shared by both branches.")
    parser.add_argument("--max-clusters", type=int, default=3,
                        help="Cluster cap of the clustered branch. The alpha "
                             "branch is fixed at 1 -- that is the comparison.")

    parser.add_argument("--min-minor-mass", type=float, default=MIN_MINOR_MASS,
                        help="Gate mass on the least-likely component for a state "
                             "to count as multi-modal.")
    parser.add_argument("--min-separation", type=float, default=MIN_SEPARATION,
                        help="Minimum distance between component means (action units).")
    parser.add_argument("--no-split-modality", dest="split_modality",
                        action="store_false",
                        help="Collapse to a single row of panels over all states, "
                             "instead of one row per modality group.")

    parser.add_argument("--ot-method", type=str, default="sinkhorn_log",
                        choices=("emd", "sinkhorn_log", "sinkhorn_stabilized", "sinkhorn"))
    parser.add_argument("--sinkhorn-reg", type=float, default=0.1)
    parser.add_argument("--hist-bins", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.num_states <= 0:
        raise ValueError("num-states must be positive")
    if args.distribution_samples <= 0:
        raise ValueError("distribution-samples must be positive")
    if args.state_batch_size <= 0:
        raise ValueError("state-batch-size must be positive")
    if args.sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if args.hist_bins <= 0:
        raise ValueError("hist-bins must be positive")

    json_path = Path(args.analysis_dir) / f"{args.run_name}.json"
    figure_path = Path(args.figures_dir) / f"{args.run_name}.png"

    if args.plot:
        if not json_path.exists():
            raise SystemExit(f"No cached data at {json_path}; run without --plot first.")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows, config = payload["per_state"], payload["config"]
        print(f"loaded: {json_path}")
        print_summary(rows)
        plot_wasserstein_histograms(
            rows=rows, config=config, output_path=figure_path,
            bins=args.hist_bins, split_modality=args.split_modality,
        )
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, action_scale = _load_uav3d_policy(Path(args.checkpoint))
    policy.to(device)
    if policy.num_components < 2:
        print(f"[warn] Checkpoint has K={policy.num_components}: no state can be "
              "multi-modal, and the two branches coincide by construction.")
    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
    env = UAV3D(max_steps=None)

    states = sample_free_states(env, args.num_states, args.velocity_max, rng)
    masses, separations = gate_modality(policy, states, action_scale, device)
    is_multimodal = (masses >= args.min_minor_mass) & (separations >= args.min_separation)
    num_multimodal = int(is_multimodal.sum())
    print(f"sampled {args.num_states} free states: {num_multimodal} multi-modal "
          f"({is_multimodal.mean():.1%}) at mass >= {args.min_minor_mass}, "
          f"separation >= {args.min_separation}")
    if args.split_modality and num_multimodal < 30:
        print(f"[warn] {num_multimodal} multi-modal states is thin for a histogram. "
              "Raise --num-states; the share is a property of the policy, so it is "
              "the sample size and not the sampling that fixes this.")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    num_batches = math.ceil(args.num_states / args.state_batch_size)
    for batch_id, start in enumerate(range(0, args.num_states, args.state_batch_size), 1):
        index = np.arange(start, min(start + args.state_batch_size, args.num_states))
        t0 = time.perf_counter()
        batch_rows = evaluate_batch(stochastic_fn, states[index], args, device)
        for row, i in zip(batch_rows, index):
            row["minor_mode_mass"] = float(masses[i])
            row["mode_separation"] = float(separations[i])
            row["multimodal"] = bool(is_multimodal[i])
        rows.extend(batch_rows)
        print(f"batch {batch_id}/{num_batches} ({len(index)} states) "
              f"in {time.perf_counter() - t0:.1f}s")
    print(f"total {time.perf_counter() - started:.1f}s")

    config = {
        "num_states": int(args.num_states),
        "velocity_max": float(args.velocity_max),
        "distribution_samples": int(args.distribution_samples),
        "sigma": float(args.sigma),
        "alpha": float(args.alpha),
        "max_clusters": int(args.max_clusters),
        "smoothing_N": int(defense_parameters(1, args.alpha).smoothing.N),
        "confidence_num_samples": int(
            defense_parameters(1, args.alpha).confidence.num_samples
        ),
        "min_minor_mass": float(args.min_minor_mass),
        "min_separation": float(args.min_separation),
        "multimodal_share_of_state_space": float(is_multimodal.mean()),
        "ot_method": args.ot_method,
        "sinkhorn_reg": float(args.sinkhorn_reg),
        "action_scale": float(action_scale),
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"config": config, "per_state": rows}, indent=2), encoding="utf-8"
    )
    print(f"saved: {json_path}")

    print_summary(rows)
    plot_wasserstein_histograms(
        rows=rows, config=config, output_path=figure_path,
        bins=args.hist_bins, split_modality=args.split_modality,
    )


if __name__ == "__main__":
    main()
