#!/usr/bin/env python3
"""Why clustered smoothing survives the attack and alpha-smoothing does not.

The sweep in ``uav3d_attack_sweep.py`` reports *that* the branches separate. This
figure shows *why*, in one picture, at the state the whole experiment runs from.

The claim being made concrete: at a decision point the policy is bimodal -- two
distinct, individually valid routes. Single-cluster alpha-smoothing averages the
whole action cloud into one atom, and **the average of two good actions is not a
good action**: it lands in the valley between the modes, which here means flying
at the obstacle the two routes go around. Clustered smoothing keeps one atom per
mode, so every emitted action is one a mode actually endorses.

That "valley" is usually asserted. Here it is drawn: the background is the
critic's own action value ``Q(s, a) = r(s, a) + gamma * V(s')`` over the action
plane, so the reader can see the low-value ridge the averaged atom sits on and
read off how much value the averaging costs.

Layout: two columns (clean input / worst-case input at ``eps = sigma``), with the
raw action cloud, the k=1 atom, and the k>1 atoms overlaid on the Q field.

Geometry: the 3-D action space is projected onto the plane through the two
mixture-component mean actions -- the plane the modes and everything between them
live in, so the projection loses nothing that matters to this argument.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from collect_uav3d_rollouts import _load_uav3d_policy
from dynamics import UAV3D
from matplotlib.lines import Line2D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import normalize_state
from uav3d_attack_sweep import BRANCH_STYLE, defense_parameters
from uav3d_attacks import (
    expected_q,
    load_critic,
    make_smoothed_objective,
    pgd_attack,
)

from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

GRID_RESOLUTION = 220


def mode_plane(policy, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal basis ``(origin, e1, e2)`` of the plane through the two modes.

    ``e1`` points from the first mode to the second, so the horizontal axis of
    the figure *is* the mode-to-mode direction and the midpoint -- where a
    single-atom average necessarily lands -- sits at a fixed, readable place.
    """
    with torch.no_grad():
        means, _, _ = policy.distribution_params(x)
    modes = (torch.tanh(means[0]) * policy.action_scale).cpu().numpy()
    if modes.shape[0] < 2:
        raise SystemExit(
            "Policy has a single mixture component -- there is no mode-averaging "
            "failure to draw. Check find_bimodal_state.py first."
        )

    origin = modes.mean(axis=0)
    e1 = modes[1] - modes[0]
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)

    # Second axis: any direction orthogonal to e1. Take the action-space axis
    # least aligned with e1 and orthogonalize, so e2 is numerically well-conditioned.
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(e1)))] = 1.0
    e2 = helper - np.dot(helper, e1) * e1
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    return origin, e1, e2


def project(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    centered = points.reshape(-1, 3) - origin
    return np.stack([centered @ e1, centered @ e2], axis=1)


def q_field(
    env: UAV3D,
    true_state: np.ndarray,
    critic,
    gamma: float,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    extent: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``Q(s, a)`` on a grid over the mode plane."""
    axis = np.linspace(-extent, extent, GRID_RESOLUTION)
    grid_1, grid_2 = np.meshgrid(axis, axis)
    actions = (
        origin[None, None, :]
        + grid_1[..., None] * e1[None, None, :]
        + grid_2[..., None] * e2[None, None, :]
    )
    with torch.no_grad():
        values = expected_q(
            env, true_state,
            torch.tensor(actions, dtype=torch.float32, device=device),
            critic, gamma,
        )
    return grid_1, grid_2, values.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
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
    parser.add_argument("--radius", type=float, default=None,
                        help="Attack radius for the right-hand column. Defaults to sigma.")
    parser.add_argument("--num-samples", type=int, default=4000,
                        help="Raw-policy action draws shown as the background cloud.")
    parser.add_argument("--num-realizations", type=int, default=200,
                        help="Independent smooth() draws per branch. Each is one "
                             "realization of the controller's output; k=1 on a "
                             "balanced gate is highly variable, so a single draw "
                             "misrepresents it.")
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--pgd-restarts", type=int, default=3)
    parser.add_argument("--eot-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figures-dir", type=str,
                        default="experiments/uav_3d/adversarial_figures")
    parser.add_argument("--figure-name", type=str, default="uav3d_attack_mechanism.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    radius = args.sigma if args.radius is None else args.radius
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    policy.to(device)
    critic, gamma = load_critic(checkpoint_path, device)
    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)
    env = UAV3D(gaussian_noise=True, noise_std=0.05, max_steps=None)

    state = BIMODAL_START_STATE
    x = torch.tensor(normalize_state(state), dtype=torch.float32, device=device).unsqueeze(0)
    origin, e1, e2 = mode_plane(policy, x)

    parameters = {
        "alpha": defense_parameters(1, args.alpha),
        "robust": defense_parameters(args.max_clusters, args.alpha),
    }

    # Both branches are attacked white-box against themselves, exactly as in the
    # sweep, so the right-hand column shows the perturbation each defense actually
    # faces rather than a shared stand-in.
    columns = []
    for title, attacked in (
        (r"Clean input  $x$", False),
        (rf"Worst-case input  $x+\delta$,  $\|\delta\|_2={radius:g}$", True),
    ):
        entry: dict[str, object] = {"title": title}
        for branch in ("alpha", "robust"):
            smoother = GaussianAlphaTrimmingSmoothing(
                stochastic_function=stochastic_fn, x=x, sigma=args.sigma,
                parameters=parameters[branch],
            )
            if attacked:
                objective = make_smoothed_objective(
                    policy=policy, smoother=smoother, env=env, true_state=state,
                    critic=critic, gamma=gamma, eot_samples=args.eot_samples,
                )
                delta = pgd_attack(objective, x, radius=radius,
                                   steps=args.pgd_steps, restarts=args.pgd_restarts)
            else:
                delta = torch.zeros_like(x)

            # The executed controller re-clusters at the perturbed input, so the
            # atoms drawn here are the ones actually emitted.
            executed = GaussianAlphaTrimmingSmoothing(
                stochastic_function=stochastic_fn, x=x + delta, sigma=args.sigma,
                parameters=parameters[branch],
            )
            # ``smooth()`` redraws its N samples every call, so a single call is
            # one realization, not the controller's output distribution -- and for
            # k=1 on a balanced gate the variance is the whole story. The
            # alpha-trimmed mean of a bimodal cloud depends on how many of the N
            # samples happened to land in each mode, so it concentrates on a
            # ladder of values rather than a point (the quantization effect in
            # experiments/paper/quantization_illustration.py). Drawing one atom
            # would misrepresent that as a clean point estimate.
            realizations = [executed.smooth() for _ in range(args.num_realizations)]
            entry[branch] = {
                "atoms": np.concatenate(
                    [d.locs[0].detach().cpu().numpy() for d in realizations]
                ),
                "weights": np.concatenate(
                    [d.probs[0].detach().cpu().numpy() for d in realizations]
                ) / args.num_realizations,
                "k": int(executed.cluster.k_per_row[0].item()),
                "delta": delta,
            }

        # Raw cloud at the raw branch's own attacked input.
        raw_delta = entry["robust"]["delta"] if attacked else torch.zeros_like(x)
        with torch.no_grad():
            samples, _, _ = policy.sample((x + raw_delta).expand(args.num_samples, -1))
        entry["raw"] = samples.cpu().numpy()
        columns.append(entry)

    # A shared colour scale across both columns, so the attacked panel's shift is
    # a real change in value and not a rescaled colormap.
    extent = 1.15 * action_scale
    fields = [
        q_field(env, state, critic, gamma, origin, e1, e2, extent, device)
        for _ in columns
    ]
    vmin = min(f[2].min() for f in fields)
    vmax = max(f[2].max() for f in fields)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, sharey=True)
    mesh = None
    for ax, entry, (grid_1, grid_2, values) in zip(axes, columns, fields):
        mesh = ax.pcolormesh(grid_1, grid_2, values, cmap="viridis",
                             vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
        ax.contour(grid_1, grid_2, values, levels=10, colors="white",
                   linewidths=0.4, alpha=0.35)

        raw_xy = project(entry["raw"], origin, e1, e2)
        ax.scatter(raw_xy[:, 0], raw_xy[:, 1], s=2.5, alpha=0.16,
                   color=BRANCH_STYLE["raw"]["color"], linewidths=0, zorder=2)

        for branch, marker, size in (("alpha", "^", 55), ("robust", "s", 45)):
            data = entry[branch]
            atoms_xy = project(data["atoms"], origin, e1, e2)
            weights = np.asarray(data["weights"], dtype=float)
            # Phantom cluster slots carry ~0 mass; drawing them would invent
            # atoms the controller never emits.
            keep = weights > 1e-3 / max(1, args.num_realizations)
            ax.scatter(
                atoms_xy[keep, 0], atoms_xy[keep, 1], s=size,
                marker=marker, color=BRANCH_STYLE[branch]["color"],
                edgecolors="none", alpha=0.25, zorder=4,
            )

        ax.set_title(entry["title"], fontsize=10)
        ax.set_xlabel("Action, mode-to-mode axis")
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

    axes[0].set_ylabel("Action, orthogonal axis")

    k_robust = columns[0]["robust"]["k"]
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               color=BRANCH_STYLE["raw"]["color"], label="Raw policy samples"),
        Line2D([], [], marker="^", linestyle="none", markersize=8,
               color=BRANCH_STYLE["alpha"]["color"],
               label=rf"$\alpha$-smoothing atoms ($k=1$, {args.num_realizations} draws)"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               color=BRANCH_STYLE["robust"]["color"],
               label=rf"Clustered atoms ($k={k_robust}$, {args.num_realizations} draws)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True, fontsize=8)

    colorbar = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label(r"Action value  $Q(s,a) = r(s,a) + \gamma V(s')$")

    fig.suptitle(
        "Mode-averaging puts the single-atom defense in the low-value valley "
        "between two valid modes",
        fontsize=11,
    )

    output_path = Path(args.figures_dir) / args.figure_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # The quantitative version of the picture, for the caption.
    print(f"saved: {output_path}")
    for entry in columns:
        print(f"\n{entry['title']}")
        for branch in ("alpha", "robust"):
            data = entry[branch]
            with torch.no_grad():
                q = expected_q(
                    env, state,
                    torch.tensor(data["atoms"], dtype=torch.float32, device=device),
                    critic, gamma,
                ).cpu().numpy()
            weights = np.asarray(data["weights"], dtype=float)
            mean_q = float((weights * q).sum())
            # Spread of the emitted action's value across realizations. For k=1 on
            # a balanced gate this is large and multi-modal -- the quantization
            # ladder -- which is the point, so it is reported rather than averaged
            # away.
            spread = float(np.sqrt((weights * (q - mean_q) ** 2).sum() / weights.sum()))
            deciles = np.round(np.percentile(q, [10, 50, 90]), 2).tolist()
            print(f"  {branch:>6} (k={data['k']}): E[Q] = {mean_q:+.3f}  "
                  f"sd {spread:.3f}  Q deciles(10/50/90) {deciles}  "
                  f"{len(np.unique(np.round(q, 1)))} distinct atom values")


if __name__ == "__main__":
    main()
