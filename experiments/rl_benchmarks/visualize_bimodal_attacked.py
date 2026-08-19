#!/usr/bin/env python3
"""The bimodal action-distribution figures of ``visualize_bimodal_robustified.py``,
redrawn under the PGD state attack of ``uav3d_attacks.py``.

Same state, same defense, same plotting code, same axis window -- the only
difference is that each controller is shown the *attacked* observation
``x + delta`` instead of the clean ``x``:

  (i)   ``bimodal_raw_attacked.png``      -- samples from the raw stochastic
                                             policy at ``x + delta_raw``.
  (ii)  ``bimodal_smoothed_attacked.png`` -- Monte-Carlo draws from the clustered
                                             alpha-trimmed smoothed predictor
                                             re-anchored at ``x + delta_robust``.
  (iii) ``bimodal_alpha_attacked.png``    -- the ``max_clusters=1`` alpha-smoothing
                                             baseline at ``x + delta_alpha``.

**The certification boxes drawn are the clean ones.** The samples move under the
attack; the boxes do not. They are the regions the defense committed to at the
unattacked state, so these figures overlay exactly on their clean counterparts
and the only thing that changes between the two is the action cloud -- which is
the whole question: does the perturbed distribution stay inside the regions the
certificate was issued for? Drawing the attacked smoother's *own* re-fitted boxes
instead would move the goalposts with the cloud and could never show a violation.

The certified-probability-vs-radius inset of the clean figures is omitted here
for the same reason: it is a property of the clean anchor, already shown there,
and repeating it under an "attacked" caption would invite reading it as an
attacked certificate. The clean certificate is printed to the console instead,
next to the *measured* occupancy of the attacked cloud in those same boxes --
which is the number that actually tests the claim.

**One attack per branch, not one attack shared.** ``delta`` is found separately
for each branch against *that branch's own* emitted action distribution, exactly
as ``collect_uav3d_attacked_rollouts.py`` does per env step. A shared delta would
be an attack on whichever branch it was computed for and a transfer attack on the
others, which would flatter the defense for the wrong reason.

**The attacker's objective is the task's own metric**: the expected discounted
return ``E_{a ~ branch(.|x+delta), w}[ r(x, a) + gamma V(x') ]`` evaluated at the
*true* state ``x`` -- the perturbation corrupts the observation, never the
dynamics. See ``uav3d_attacks`` for how each branch is differentiated (analytic
gate for raw; straight-through Gumbel-max gate and region occupancy for the
smoothed branches, with cluster geometry frozen at the clean anchor).

**Two smoothers per smoothed branch**, matching deployment:

  * the **search** smoother, anchored at the clean ``x``, supplies the cluster
    geometry the attack differentiates against;
  * the **execution** smoother, anchored at ``x + delta``, is what is actually
    plotted and certified -- a deployed controller only ever sees the attacked
    reading, so it re-clusters there.

Because the execution smoother re-clusters, the cluster *count* it finds at the
attacked state need not match the clean one. That is a headline number rather
than a nuisance -- collapsing ``k`` is one way the attack could defeat clustered
smoothing -- so it is printed for both smoothed branches. Note that the drawn
boxes stay clean-anchored regardless, so a figure with more sample clumps than
boxes is exactly the ``k``-collapse case made visible.

Alongside the figures the script prints, per branch, the attacker's own objective
at the clean and attacked observation and the drop between them. Within a branch
this is an exact comparison (the objective is built over fixed noise, so it is a
deterministic function of ``delta``); across branches the two estimators differ
-- the raw objective sums the gate analytically, the smoothed one averages
``--eot-samples`` realizations of the smoothing draw -- so read the *drops*
across branches, not the levels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from collect_uav3d_rollouts import _load_uav3d_policy
from dynamics import UAV3D
from rollout_io import BIMODAL_START_STATE
from stochastic_functions import UAV3DPolicyStochasticFunction
from train_rl import ValueNetwork, VariationalPolicy, normalize_state
from uav3d_attacks import (
    _GATE_TEMPERATURE,
    load_critic,
    make_raw_objective,
    make_smoothed_objective,
    pgd_attack,
)
from visualize_bimodal_robustified import (
    AXIS_WINDOW,
    _build_smoother,
    _plot_raw,
    _plot_smoothed,
    _print_certificate,
)

from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

# The attack radius is in normalized (Phi-space) units, the same units as
# ``--sigma``, which is what makes eps/sigma a like-for-like ratio on every axis.
# Reporting it in metres needs a nominated axis; x-position is the convention
# used by ``uav3d_attack_sweep.py``. Derived from ``normalize_state`` rather than
# restated so the two cannot drift apart.
REFERENCE_AXIS = "x-position"
REFERENCE_AXIS_SCALE = float(1.0 / normalize_state(np.ones(6, dtype=np.float32))[0])


def _attack_raw(
    policy: VariationalPolicy,
    env: UAV3D,
    critic: ValueNetwork,
    gamma: float,
    x: torch.Tensor,
    true_state: np.ndarray,
    radius: float,
    pgd_steps: int,
    pgd_restarts: int,
    eot_samples: int,
) -> tuple[torch.Tensor, float, float]:
    """PGD delta for the raw policy, plus its clean and attacked objective values.

    The objective scores ``pi(.|x+delta)`` -- the same distribution
    ``policy.sample`` then draws the plotted actions from -- with the mixture gate
    summed analytically, so the attack can search for the perturbation that moves
    probability mass *between* the two modes rather than only dragging both.
    """
    objective = make_raw_objective(
        policy=policy, env=env, true_state=true_state, critic=critic,
        gamma=gamma, eot_samples=eot_samples,
    )
    delta = pgd_attack(objective, x, radius=radius, steps=pgd_steps, restarts=pgd_restarts)
    with torch.no_grad():
        return delta, float(objective(x)), float(objective(x + delta))


def _attack_smoothed(
    policy: VariationalPolicy,
    search_smoother: GaussianAlphaTrimmingSmoothing,
    env: UAV3D,
    critic: ValueNetwork,
    gamma: float,
    x: torch.Tensor,
    true_state: np.ndarray,
    radius: float,
    pgd_steps: int,
    pgd_restarts: int,
    eot_samples: int,
    gate_temperature: float,
    partition_temperature: float | None,
) -> tuple[torch.Tensor, float, float]:
    """PGD delta for one smoothed branch, plus its clean and attacked values.

    ``search_smoother`` must be anchored at the clean ``x``: its centroids and
    regions are the frozen geometry the attack differentiates against. The atoms
    and the mixture weights both still respond to ``delta``; only the geometry is
    held fixed, which is what avoids a DBSCAN fit inside every PGD step.
    """
    objective = make_smoothed_objective(
        policy=policy, smoother=search_smoother, env=env, true_state=true_state,
        critic=critic, gamma=gamma, eot_samples=eot_samples,
        gate_temperature=gate_temperature,
        partition_temperature=partition_temperature,
    )
    delta = pgd_attack(objective, x, radius=radius, steps=pgd_steps, restarts=pgd_restarts)
    with torch.no_grad():
        return delta, float(objective(x)), float(objective(x + delta))


def _reseed(seed: int) -> None:
    """Reset both RNGs at the start of each branch.

    The reason is comparability of the figures, not the attack itself: the
    clean-anchored geometry -- the certification boxes these figures draw -- is
    then a function of ``--seed`` and the defense parameters only, not of how much
    randomness the preceding branch's attack happened to consume. Without this,
    sweeping ``--attack-radius`` or ``--pgd-steps`` would silently redraw the
    boxes from figure to figure, which is the one thing that must stay fixed
    across such a series.

    It does *not* put the three branches on common random numbers. Each branch
    restarts from the same seed, but they consume it differently -- the raw
    branch goes straight into its attack, the smoothed branches build a smoother
    first -- so only the two smoothed branches, which consume identically up to
    the DBSCAN fit, end up aligned. Branch-to-branch differences here are
    therefore not variance-controlled; the sweep in ``uav3d_attack_sweep.py`` is
    what quantifies them.

    The geometry is stable enough that this is a tidiness measure, not a
    correctness one: across independent RNG states the fitted centroids agree to
    ~0.003 and the box half-widths to ~0.02 in action units, on axes spanning
    +/-4.2. That is also why these boxes match ``visualize_bimodal_robustified``'s
    to plotting precision despite the two scripts drawing different sample clouds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def _report_occupancy(
    clean_smoother: GaussianAlphaTrimmingSmoothing,
    attacked_samples: np.ndarray,
    delta: torch.Tensor,
    radius: float,
) -> None:
    """Measured occupancy of the attacked cloud in the *clean* certified boxes.

    The figure shows this comparison; this prints it. The certificate is issued
    at the clean anchor and holds over the L2 ball of ``radius`` around it, and
    the attacked observation sits ``||delta||`` away -- so when
    ``radius >= ||delta||`` the certified total is a genuine lower bound on the
    number measured here, and the two lines are a direct check of the claim
    rather than two loosely related statistics. When ``radius < ||delta||`` the
    attacker has stepped outside the certified ball and the bound simply does not
    apply; that case is called out rather than quietly compared.

    Boxes are axis-aligned (``AxisAlignedHyperRectangle``), so membership is a
    coordinate-wise interval test. Per-cluster figures are reported individually
    and the total as "inside *any* box", which is the right aggregate even if two
    boxes were to overlap.
    """
    k_used = int(clean_smoother.cluster.k_per_row[0].item())
    lower = clean_smoother.regions.lower[0].detach().cpu().numpy()[:k_used]
    upper = clean_smoother.regions.upper[0].detach().cpu().numpy()[:k_used]

    inside = np.all(
        (attacked_samples[:, None, :] >= lower[None]) &
        (attacked_samples[:, None, :] <= upper[None]),
        axis=-1,
    )                                             # (num_samples, k_used)
    per_cluster = inside.mean(axis=0)
    total = float(inside.any(axis=-1).mean())

    delta_norm = float(torch.linalg.norm(delta).item())
    print(f"  measured occupancy of the ATTACKED cloud in those clean boxes "
          f"(n={attacked_samples.shape[0]}):")
    print(f"    per-cluster: {np.round(per_cluster, 4).tolist()}")
    print(f"    total (in any box): {total:.4f}")
    if radius >= delta_norm:
        print(f"    radius {radius:.4f} >= ||delta|| {delta_norm:.4f}, so the "
              f"certified total above is a valid lower bound on this number.")
    else:
        print(f"    radius {radius:.4f} < ||delta|| {delta_norm:.4f}: the attacked "
              f"observation is outside the certified ball, so the certificate "
              f"above does not bound this number.")


def _report_delta(label: str, delta: torch.Tensor, clean: float, attacked: float) -> None:
    d = delta.squeeze(0).detach().cpu().numpy()
    norm = float(np.linalg.norm(d))
    print(f"\n[{label}] attack")
    print(f"  ||delta||_2 = {norm:.4f} normalized units "
          f"({norm * REFERENCE_AXIS_SCALE:.3f} m along {REFERENCE_AXIS})")
    print(f"  delta (per normalized state axis) = {np.round(d, 4).tolist()}")
    print(f"  E[r + gamma V]: clean {clean:+.4f} -> attacked {attacked:+.4f} "
          f"(drop {clean - attacked:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="experiments/uav_3d/policy.pt")
    parser.add_argument("--out-dir", type=str, default="experiments/uav_3d/bimodal_plots",
                        help="Written alongside the clean figures, with an "
                             "'_attacked' suffix, so the two sets sit together.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Std of the Gaussian state-space smoothing noise (normalized state units).")
    parser.add_argument("--smoothing-N", type=int, default=30,
                        help="Number of noisy state samples per smoothing call.")
    parser.add_argument("--confidence-N", type=int, default=4000,
                        help="Number of Monte Carlo samples for confidence estimation.")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Alpha-trimming fraction (0 <= alpha < 0.5). Must stay "
                             "BELOW the state's minor-mode mass (0.242 here) or the "
                             "k=1 baseline silently becomes a mode selector; see "
                             "rollout_io.BIMODAL_START_STATE.")
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--dbscan-eps", type=float, default=0.45)
    parser.add_argument("--dbscan-min-samples", type=int, default=50)
    parser.add_argument("--raw-samples", type=int, default=4000)
    parser.add_argument("--mc-samples", type=int, default=4000,
                        help="Draws from the robustified MixtureDirac for plotting.")
    parser.add_argument("--radius", type=float, default=0.16,
                        help="Input disturbance L2 radius (normalized state units) "
                             "for the CERTIFICATE. Distinct from --attack-radius: "
                             "this one is the defense's claim, that one is the "
                             "adversary's budget.")
    parser.add_argument("--attack-radius", type=float, default=0.05,
                        help="Epsilon: L2 ball radius for the PGD state attack, in "
                             "the same (normalized-state) units as --sigma.")
    parser.add_argument("--pgd-steps", type=int, default=40,
                        help="PGD iterations.")
    parser.add_argument("--pgd-restarts", type=int, default=3,
                        help="Random restarts per attack; the first always starts "
                             "at delta=0, and the best result across restarts wins, "
                             "so the attack can never score worse than clean.")
    parser.add_argument("--eot-samples", type=int, default=64,
                        help="Fixed noise draws the attack objective averages over "
                             "(expectation-over-transformation with common random "
                             "numbers). Held constant across PGD steps so the "
                             "objective is deterministic and the descent is real.")
    parser.add_argument("--gate-temperature", type=float, default=_GATE_TEMPERATURE,
                        help="Gumbel-softmax temperature for the policy gate in the "
                             "smoothed branches. Straight-through, hence value-neutral "
                             "-- a gradient knob only.")
    parser.add_argument("--rho-temperature", type=float, default=None,
                        help="Softmax temperature for the region-occupancy mixture "
                             "weights. Also straight-through. Default derives it from "
                             "the clean anchor's own region distances.")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="UAV3D process-noise std the attack's one-step lookahead "
                             "integrates over. Matches collect_uav3d_attacked_rollouts.py.")
    parser.add_argument("--axis-lim", type=float, default=None,
                        help="Override the shared plotting window with the symmetric "
                             "box [-L, L]^3. The default reuses the clean figures' "
                             "window verbatim so the two are directly comparable; "
                             "widen it only if the attacked cloud is being clipped "
                             "(the script warns when it is).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_path = Path(args.checkpoint)
    policy, action_scale = _load_uav3d_policy(checkpoint_path)
    policy.to(device)
    critic, gamma = load_critic(checkpoint_path, device)
    if policy.num_components < 2:
        print(f"[warn] Checkpoint has K={policy.num_components}; bimodality visualization assumes K>=2.")
    print(f"Loaded VI-BNN policy from {checkpoint_path} (action_scale={action_scale}) onto {device}.")
    print(f"Loaded critic (gamma={gamma}) — the attack minimizes E_a,w[r + gamma V(x')].")

    base_state = BIMODAL_START_STATE.astype(np.float32)
    normed = normalize_state(base_state)
    x = torch.tensor(normed, dtype=torch.float32, device=device).unsqueeze(0)

    stochastic_fn = UAV3DPolicyStochasticFunction(policy=policy)

    # max_steps=None to match the rollout scripts. Read-only here: the attack
    # objective touches env.A / env.B / env.goal_center / the action bounds and
    # the noise parameters, never env.step.
    env = UAV3D(gaussian_noise=True, noise_std=args.noise_std, max_steps=None)

    window = AXIS_WINDOW if args.axis_lim is None else (
        (-args.axis_lim, args.axis_lim),
    ) * 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _warn_if_clipped(label: str, samples: np.ndarray) -> None:
        """The default window is tight around the *clean* cloud; a perturbed one
        need not fit. Silently cropping the very effect the figure exists to show
        would be the worst outcome, so say so and point at the override."""
        lower = np.array([lim[0] for lim in window])
        upper = np.array([lim[1] for lim in window])
        outside = np.any((samples < lower) | (samples > upper), axis=-1)
        if outside.any():
            print(f"[warn] {label}: {outside.mean():.1%} of plotted samples fall "
                  f"outside the shared axis window {window}; they are cropped from "
                  f"the figure. Pass --axis-lim to widen it (at the cost of "
                  f"comparability with the clean figures).")

    common_attack = dict(
        env=env, critic=critic, gamma=gamma, x=x, true_state=base_state,
        radius=args.attack_radius, pgd_steps=args.pgd_steps,
        pgd_restarts=args.pgd_restarts, eot_samples=args.eot_samples,
    )
    print(f"\nAttack budget: ||delta||_2 <= {args.attack_radius:.4f} normalized units "
          f"({args.attack_radius * REFERENCE_AXIS_SCALE:.3f} m along {REFERENCE_AXIS}), "
          f"{args.pgd_steps} PGD steps x {args.pgd_restarts} restarts.")

    # ---- (i) Raw predictor under attack ----
    _reseed(args.seed)
    delta_raw, clean_raw, attacked_raw = _attack_raw(
        policy=policy, **common_attack)
    _report_delta("raw predictor", delta_raw, clean_raw, attacked_raw)

    with torch.no_grad():
        s_rep = (x + delta_raw).repeat(args.raw_samples, 1)
        raw_actions_t, _, _ = policy.sample(s_rep)
    raw_actions = raw_actions_t.cpu().numpy()
    raw_path = out_dir / "bimodal_raw_attacked.png"
    _warn_if_clipped("raw predictor", raw_actions)
    _plot_raw(raw_actions, raw_path, base_state, window=window)
    print(f"Saved attacked raw predictor figure to {raw_path}")

    def _run_smoothed_branch(max_clusters: int, filename: str, label: str) -> None:
        """Attack, re-anchor, plot against the clean boxes, and check the claim."""
        smoother_kwargs = dict(
            stochastic_fn=stochastic_fn, sigma=args.sigma,
            smoothing_N=args.smoothing_N, confidence_N=args.confidence_N,
            alpha=args.alpha, coverage=args.coverage, max_clusters=max_clusters,
            dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
        )
        # Clean-anchored smoother. Serves two roles, and they coincide by design:
        # it supplies the frozen cluster geometry the attack differentiates
        # against, and it is the geometry the figure and the certificate below
        # are stated in -- the same boxes the unattacked figure draws.
        _reseed(args.seed)
        clean_smoother = _build_smoother(x=x, **smoother_kwargs)
        k_clean = int(clean_smoother.cluster.k_per_row[0].item())

        delta, clean_value, attacked_value = _attack_smoothed(
            policy=policy, search_smoother=clean_smoother,
            gate_temperature=args.gate_temperature,
            partition_temperature=args.rho_temperature,
            **common_attack,
        )
        _report_delta(label, delta, clean_value, attacked_value)

        # Execution smoother: re-clusters at the attacked observation, which is
        # the only reading a deployed controller ever gets. Its samples are what
        # gets plotted; its boxes are deliberately not.
        exec_smoother = _build_smoother(x=x + delta, **smoother_kwargs)
        k_attacked = int(exec_smoother.cluster.k_per_row[0].item())
        print(f"  clusters found: {k_clean} (clean anchor) -> {k_attacked} (attacked anchor)")

        mc = exec_smoother.monte_carlo(
            num_simulations=args.mc_samples).squeeze(0).detach().cpu().numpy()
        out_path = out_dir / filename
        _warn_if_clipped(label, mc)
        _plot_smoothed(exec_smoother, mc, out_path, window=window,
                       regions_from=clean_smoother)
        print(f"Saved attacked {label} figure to {out_path}")

        _print_certificate(clean_smoother, sigma=args.sigma, radius=args.radius,
                           label=f"{label} @ clean anchor (the boxes drawn)")
        _report_occupancy(clean_smoother, mc, delta, args.radius)

    # ---- (ii) Clustered smoothed predictor under attack ----
    _run_smoothed_branch(args.max_clusters, "bimodal_smoothed_attacked.png",
                         "smoothed predictor")

    # ---- (iii) Alpha-smoothing baseline (K = 1) under attack ----
    _run_smoothed_branch(1, "bimodal_alpha_attacked.png", "alpha-smoothing (K=1)")


if __name__ == "__main__":
    main()
