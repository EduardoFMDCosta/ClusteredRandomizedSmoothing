#!/usr/bin/env python3
"""PGD state attack on the UAV3D controller's expected discounted return.

Threat model: the attacker perturbs the (normalized) **state** handed to the
controller inside an L2 ball ``||delta||_2 <= radius`` -- the same input space the
``GaussianAlphaTrimmingSmoothing`` defense smooths over. The true state, and hence
the actual dynamics, are untouched; only the observation is corrupted.

**Objective.** The attacker minimizes the expected discounted return

    J(delta) = E_{a ~ pi_branch(. | x + delta), w ~ P_w} [ r(x, a) + gamma * V(x') ],
    x' = A x + B a + w,

evaluated at the *true* state ``x``. This is the quantity the task is scored on,
so the attack and the reported metric are the same object -- no objective/metric
mismatch to argue about.

**Two distinct objects, both of which must respond to delta.** They are easy to
conflate and are kept under separate names throughout:

  * the **policy gate** ``omega_k(x)``, ``k = 1..K`` -- the *policy's* own mixture
    over behavioral modes, ``log_softmax(gate_head(features(x)))`` in
    ``VariationalPolicy.distribution_params``. A property of the trained network;
    it exists with or without any defense.
  * the **mixture weights** ``rho_j(x)``, ``j = 1..num_clusters`` -- the weights of
    the *smoothed controller's emitted* ``MixtureDirac``, i.e.
    ``Cluster.probs_partition``: the counted fraction of sampled actions landing in
    region ``j``. A property of the defense, recomputed at run time.

``rho`` is downstream of ``omega``, over a different index set: the smoother draws
N noisy inputs, each of which draws a component from ``omega``, and ``rho`` counts
where the resulting actions landed. The two are numbered independently -- DBSCAN
does not order clusters by policy component -- so they generally disagree even in
ordering.

**Differentiability.** The full path ``delta -> action -> x' -> V(x')`` is
differentiable:

  * The expectation over the process noise ``w`` is a deterministic
    Gauss-Hermite rule over the three noisy velocity axes, not a Monte-Carlo
    average, so it adds no variance and leaves the objective an exact function
    of ``delta`` -- which is what lets PGD compare iterates and restarts.
  * ``distribution_params`` exposes the gate in closed form, so the **raw** branch
    is handled analytically over its K components -- the gradient sees both the
    component means moving *and* the gate re-weighting.
  * The **smoothed** branches cannot do that: the alpha-trimmed mean is a
    non-linear function of the *joint* set of N samples, so the expectation over
    the gate does not push inside it (K**N combinations). Instead ``omega`` is
    sampled by a straight-through **Gumbel-max** draw with fixed noise. Because
    ``argmax_k(log omega_k + G_k)`` *is* an exact categorical draw from ``omega``,
    and dividing logits by a temperature does not move an argmax, the forward
    value is an exact sample from the true gate at ``x + delta`` for any
    temperature -- the temperature shapes only the gradient.
  * ``rho`` is likewise straight-through: forward it is the hard region occupancy
    (the same count ``Cluster._compute_probs_partition`` performs), backward it is
    a softmax over the negated region distances, which ``regions.distance`` makes
    differentiable in the sampled actions.
  * The alpha-trimming itself is differentiated and always has been: ``sort``
    propagates gradient through its permutation, and the trimmed mean is
    piecewise linear in its inputs, so autograd's gradient is correct almost
    everywhere -- it fails only where two entries tie.

**What is still frozen (BPDA).** The region geometry -- centroids, boxes, the
DBSCAN fit, ``num_clusters`` -- comes from a search smoother built once at the
clean anchor and is constant for the whole PGD run, whereas the executed
controller re-clusters at the perturbed input. Cluster *membership* and the
integer trim counts ``left = floor(alpha * n_j)``, ``right = n_j - left`` are also
constants in the backward pass.

**The objective is therefore discontinuous in delta** -- the hard gate index and
the hard region assignment flip -- so the gradient is a relaxation and PGD is a
heuristic on it. The forward value it compares across steps and restarts is the
real quantity, which is what makes the ``delta = 0`` incumbent in ``pgd_attack``
mean something.

**One value oracle for all branches, by design.** ``V`` is ``V^{pi_raw}``, not a
per-branch value function. This is deliberate: a single task-value yardstick means
no branch is attacked against an easier objective, so any difference in the
measured return is attributable to the defense rather than to the attack.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from dynamics import UAV3D
from train_rl import ValueNetwork, VariationalPolicy, normalize_state

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing

# PGD step size as a multiple of ``radius / steps``. The standard Madry-style
# choice: large enough that the iterate can cross the ball a couple of times and
# escape a poor initialization, small enough that the projection is not doing all
# the work.
_STEP_SIZE_FACTOR = 2.5

# Gumbel-softmax temperature for the policy gate. Dimensionless: the logits it
# divides are log-probabilities. Only the *gradient* depends on it -- the
# straight-through forward value is an exact categorical draw at any temperature
# -- so this trades gradient magnitude against how local the relaxation is.
#
# Swept over {0.1, 0.3, 1.0, 3.0} at BIMODAL_START_STATE, eps=0.05: the attained
# objective varies by under 1% across 0.1--1.0 for both smoothed branches and
# degrades ~10% at 3.0. 0.3 is the compromise -- within 0.1% of the best cell for
# each branch, where 1.0 costs the clustered branch ~1%.
_GATE_TEMPERATURE = 0.3

# Softmax temperature for the region-occupancy weights, as a fraction of the
# median positive region distance at the clean anchor. Derived rather than
# hardcoded because the distances live in action units, so a constant here would
# silently change meaning if ``action_scale`` did.
#
# The derived value beat both a 5x-smaller and a 5x-larger setting in every row of
# the gate sweep, so the scale this picks is the right one, not merely a safe one.
# Irrelevant at k=1, where a single region leaves the weights identically [1.0].
_PARTITION_TEMPERATURE_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_critic(checkpoint_path: Path, device: torch.device) -> tuple[ValueNetwork, float]:
    """Load the PPO critic and the discount it was fitted under.

    Mirrors ``collect_uav3d_rollouts._load_uav3d_policy``. Checkpoints written
    before the critic was persisted will fail here with an explicit message
    rather than silently falling back to a myopic objective.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "critic_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} has no 'critic_state_dict'. The critic used to be "
            "discarded by train_rl.train(); retrain with the current train_rl.py, "
            "which saves it alongside the policy."
        )
    if "gamma" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} has no 'gamma'. A value function is meaningless "
            "without the discount it was fitted under; retrain with the current "
            "train_rl.py."
        )

    critic = ValueNetwork(
        state_dim=int(checkpoint["state_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    critic.load_state_dict(checkpoint["critic_state_dict"])
    critic.eval()
    critic.to(device)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    return critic, float(checkpoint["gamma"])


# ---------------------------------------------------------------------------
# Differentiable one-step action value
# ---------------------------------------------------------------------------
# Derived from ``train_rl.normalize_state`` rather than restated, so the two
# cannot drift apart: normalizing the all-ones state yields the reciprocal of the
# per-axis divisors.
_NORMALIZER = torch.from_numpy(1.0 / normalize_state(np.ones(6, dtype=np.float32)))

# ``UAV3D.step`` clips velocities (state indices 1/3/5) to +/-7 after applying the
# dynamics; positions are unbounded within the corridor. Reproduced so the
# lookahead matches the env exactly.
_STATE_CLIP = torch.tensor([torch.inf, 7.0, torch.inf, 7.0, torch.inf, 7.0])

# Gauss-Hermite nodes per velocity axis. The rule is exact for polynomials of
# degree < 2*3 = 5 in each component, so with the process noise this small the
# quadrature error is far below the critic's own approximation error. Three
# axes carry noise, hence 3**3 = 27 evaluations of ``V`` per action.
_NOISE_NODES_PER_AXIS = 3

# Actions per critic call. The quadrature multiplies the batch by the node count,
# which the attack path never notices (a few hundred actions) but which the dense
# action grids in ``uav3d_attack_mechanism.py`` do: 220**2 actions x 27 nodes is
# 1.3M rows and ~2 GB of activations in one shot. Chunking bounds that; under
# autograd the graph is retained either way, but those batches are small.
_ACTION_CHUNK = 4096


@lru_cache(maxsize=None)
def _process_noise_quadrature(
    gaussian: bool, sigma: float, nodes_per_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Product Gauss-Hermite rule for ``E_{w ~ P_w}`` under ``UAV3D._sample_noise``.

    Returns ``(offsets, weights)``, ``(Q, 6)`` and ``(Q,)`` with the weights
    summing to one. The offsets are zero in the position slots: ``_sample_noise``
    perturbs only the velocity components 1/3/5, independently per axis.

    A deterministic rule rather than a Monte-Carlo average over sampled ``w``,
    because ``pgd_attack`` compares objective values across steps and restarts
    and needs the objective to be an exact function of ``delta``. Common random
    numbers would also work, but at equal cost they are strictly less accurate:
    27 nodes integrate this noise essentially exactly, 27 samples do not.
    """
    if sigma <= 0.0:
        return np.zeros((1, 6)), np.ones(1)

    nodes, weights = np.polynomial.hermite.hermgauss(nodes_per_axis)
    # Physicists' convention: ``int f(t) exp(-t^2) dt = sum_i w_i f(t_i)``. The
    # substitution ``t = v / (sqrt(2) sigma)`` turns it into ``E_{v~N(0,sigma^2)}``.
    axis_nodes = np.sqrt(2.0) * sigma * nodes
    axis_weights = weights / np.sqrt(np.pi)

    if not gaussian:
        # The heavy-tailed branch of ``_sample_noise`` draws N(0, sigma) or
        # N(0, 2 sigma) with equal probability, per axis. A mixture integrates as
        # the mixture of the component rules.
        axis_nodes = np.concatenate([axis_nodes, 2.0 * axis_nodes])
        axis_weights = 0.5 * np.concatenate([axis_weights, axis_weights])

    node_grid = np.meshgrid(*(3 * [axis_nodes]), indexing="ij")
    weight_grid = np.meshgrid(*(3 * [axis_weights]), indexing="ij")

    offsets = np.zeros((axis_nodes.size**3, 6))
    offsets[:, [1, 3, 5]] = np.stack([g.reshape(-1) for g in node_grid], axis=-1)
    return offsets, np.prod(np.stack([g.reshape(-1) for g in weight_grid]), axis=0)


def expected_q(
    env: UAV3D,
    true_state: np.ndarray | torch.Tensor,
    actions: torch.Tensor,
    critic: ValueNetwork,
    gamma: float,
    nodes_per_axis: int = _NOISE_NODES_PER_AXIS,
) -> torch.Tensor:
    """One-step action value ``r(x, a) + gamma * E_w[V(x')]``, batched over ``actions``.

    Two calling conventions:

      * ``true_state`` ``(6,)`` with ``actions`` ``(..., 3)`` -> ``(...)``. The
        attack path: one state, many candidate actions.
      * ``true_state`` ``(n, 6)`` with ``actions`` ``(n, k, 3)`` -> ``(n, k)``.
        Many states at once.

    Fully differentiable in ``actions``.

    The expectation is over the additive process noise of ``UAV3D.step``:
    ``x' = A x + B a + w`` with ``w`` perturbing the three velocity components.
    It is taken by the deterministic quadrature above, not by sampling, and it
    does not commute with ``V`` -- dropping it and evaluating ``V(A x + B a)``
    instead is a plug-in approximation off by the Jensen gap
    ``0.5 * sigma^2 * tr(D^2 V)`` over the velocity block. Small at the noise
    levels used here, but it is a bias, and it is one the *executed* controller
    is actually exposed to.

    Positions are noise-free at this horizon: ``w`` enters only the velocities,
    and the double integrator advances position by the *pre-step* velocity, so
    ``x'_pos = x_pos + x_vel + 0.5 a`` regardless of ``w``. The progress reward
    is therefore exact, and the expectation falls entirely on ``V``.

    Only the *dense* reward branch is used: ``0.5 * progress - 0.01``. The
    terminal branches of ``UAV3D.step`` (-5 corridor exit, -5 obstacle entry,
    +10 goal) are deliberately omitted -- they are discontinuous, and ``V``
    already encodes their value smoothly, having been fitted on returns that
    contain them. Dropping them is precisely what makes this objective
    differentiable.

    Side-effect free: reads only ``env.A``/``env.B``/``env.goal_center``, the
    action bounds and the noise parameters, never ``env.step``, which is stateful
    and consumes the global ``np.random`` stream shared with the live rollout.
    """
    device, dtype = actions.device, actions.dtype
    A = torch.as_tensor(env.A, device=device, dtype=dtype)
    B = torch.as_tensor(env.B, device=device, dtype=dtype)
    state = torch.as_tensor(true_state, device=device, dtype=dtype)
    goal = torch.as_tensor(env.goal_center, device=device, dtype=dtype)

    clipped = actions.clamp(float(env.u_min), float(env.u_max))
    position_index = [0, 2, 4]

    if state.ndim == 1:
        next_state = (A @ state) + clipped @ B.T          # (..., 6)
        old_position = state[position_index]
    else:
        next_state = (state @ A.T).unsqueeze(1) + clipped @ B.T   # (n, k, 6)
        old_position = state[:, position_index].unsqueeze(1)      # (n, 1, 3)

    new_position = next_state[..., position_index]
    previous_distance = torch.linalg.norm(old_position - goal, dim=-1)
    new_distance = torch.linalg.norm(new_position - goal, dim=-1)
    reward = 0.5 * (previous_distance - new_distance) - 0.01

    offsets, quadrature_weights = _process_noise_quadrature(
        bool(env.gaussian_noise), float(env.noise_std), int(nodes_per_axis)
    )
    # (..., 1, 6) + (Q, 6) -> (..., Q, 6). The velocity clip is applied *after*
    # the noise, matching the order in ``UAV3D.step``.
    noisy_next_state = next_state.unsqueeze(-2) + torch.as_tensor(
        offsets, device=device, dtype=dtype
    )
    clip = _STATE_CLIP.to(device=device, dtype=dtype)
    noisy_next_state = noisy_next_state.clamp(-clip, clip)

    normalizer = _NORMALIZER.to(device=device, dtype=dtype)
    weights = torch.as_tensor(quadrature_weights, device=device, dtype=dtype)
    value = torch.cat([
        (critic(chunk / normalizer) * weights).sum(-1)
        for chunk in noisy_next_state.reshape(-1, weights.shape[0], 6).split(_ACTION_CHUNK)
    ]).view(reward.shape)
    return reward + gamma * value


# ---------------------------------------------------------------------------
# Per-branch action distributions, as (atoms, weights)
# ---------------------------------------------------------------------------
def raw_action_atoms(
    policy: VariationalPolicy,
    x: torch.Tensor,
    component_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw policy's action distribution at ``x``, handled analytically.

    ``x`` is ``(1, state_dim)``; ``component_noise`` is a fixed ``(M, K, A)``
    standard-normal draw (common random numbers). Returns ``atoms`` of shape
    ``(K, M, A)`` -- ``M`` reparameterized draws from each mixture component --
    and ``weights`` of shape ``(K,)``, the exact gate probabilities.

    Keeping the gate analytic rather than sampling a component index matters: the
    gradient then flows into ``log_mix_weights``, so PGD can search for the
    perturbation that shifts probability mass *between* modes, not merely the one
    that drags both modes somewhere bad.

    This is already the objective exactly -- no relaxation, no frozen quantity --
    because the raw policy emits a K-atom mixture and K is small enough to sum
    over. The smoothed branches below need Gumbel-max precisely because their
    alpha-trimmed mean does not let the gate expectation push inside. Do not
    "fix" this function to match them.
    """
    means, log_stds, log_mix_weights = policy.distribution_params(x)
    stds = log_stds.exp()
    # (1, K, A) -> (M, K, A) by broadcasting against the fixed noise.
    raw = means[0].unsqueeze(0) + stds[0].unsqueeze(0) * component_noise
    atoms = torch.tanh(raw) * policy.action_scale
    return atoms.permute(1, 0, 2), log_mix_weights.exp()[0]


def _straight_through(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """``hard`` in the forward pass, ``soft``'s gradient in the backward pass.

    The forward value is bit-identical to ``hard``, which is what lets the
    smoothed objective claim it evaluates the real emitted distribution rather
    than a smoothed stand-in for it.

    The parenthesization is load-bearing. ``hard + soft - soft.detach()``
    associates as ``(hard + soft) - soft`` and rounds, so the forward value picks
    up an error that varies with the temperature -- small, but enough to make the
    objective temperature-dependent, which is exactly the property this idiom is
    supposed to deliver. Grouping the difference first makes it exactly zero.
    """
    return hard + (soft - soft.detach())


def _gumbel_softmax_components(
    log_mix_weights: torch.Tensor,
    gumbel_noise: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Straight-through Gumbel-max draw from the policy gate, ``(B, K) -> (B, K)``.

    ``gumbel_noise`` is a fixed ``(B, K)`` standard-Gumbel draw (common random
    numbers), so the result is a deterministic function of ``log_mix_weights``
    and hence of ``delta``.

    Forward this returns a one-hot at ``argmax_k(log omega_k + G_k)``, which is
    *exactly* a categorical draw from ``omega`` (the Gumbel-max trick), and
    dividing the logits by ``temperature > 0`` cannot move an argmax -- so the
    forward value is exact at any temperature. Backward it is the softmax, which
    is what carries ``d omega / d x`` into the objective.
    """
    logits = (log_mix_weights + gumbel_noise) / temperature
    soft = logits.softmax(dim=-1)
    hard = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return _straight_through(soft, hard)


def _soft_partition_weights(
    distances: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Straight-through region occupancy, ``(M, N, k) -> (M, k)``.

    ``distances`` is ``regions.distance(outputs)``. Forward, this is the hard
    fraction of the N samples nearest to each region -- the same count
    ``Cluster._compute_probs_partition`` performs, and therefore the same
    ``probs_partition`` the executed controller emits as its mixture weights.
    Backward, the softmax over negated distances carries ``d rho / d outputs``,
    which ``AxisAlignedHyperRectangle.distance`` (a clamp and a norm) makes
    differentiable.

    At ``k = 1`` the softmax over a single logit is identically 1 with zero
    gradient, matching the degenerate one-atom ``MixtureDirac`` that a
    single-cluster smoother emits.
    """
    soft = (-distances / temperature).softmax(dim=-1)
    hard = F.one_hot(distances.argmin(dim=-1), distances.shape[-1]).to(soft.dtype)
    return _straight_through(soft, hard).mean(dim=1)


def _derive_partition_temperature(
    outputs: torch.Tensor,
    cluster: Cluster,
    factor: float = _PARTITION_TEMPERATURE_FACTOR,
) -> float:
    """Occupancy temperature as ``factor`` x the median positive region distance.

    Distances are in action units, so the scale has to come from the data rather
    than a constant. Only distances to regions a sample is *outside* of enter the
    median -- a sample inside its own box is at distance zero, which carries no
    scale information.
    """
    if cluster.num_clusters <= 1:
        return 1.0
    with torch.no_grad():
        flat = outputs.reshape(1, -1, outputs.shape[-1])
        distances = cluster.regions.distance(flat)
        positive = distances[distances > 0]
    if positive.numel() == 0:
        return 1.0
    return float(factor * positive.median())


def _alpha_trimmed_atoms(
    outputs: torch.Tensor,
    cluster: Cluster,
    alpha: float,
    assignments: torch.Tensor,
) -> torch.Tensor:
    """Per-cluster alpha-trimmed means of ``outputs``, shape ``(M, N, A) -> (M, k, A)``.

    A batched transcription of ``AlphaTrimmingSmoothing.smooth`` (see
    ``src/randomized_smoothing/smoothing.py``) over ``M`` independent realizations,
    using a cluster frozen at the clean anchor. ``assignments`` is the hard
    ``(M, N)`` region assignment, passed in so the caller can share one
    ``regions.distance`` evaluation with ``_soft_partition_weights``.

    ``sort`` and ``torch.where`` both propagate gradients to the selected entries,
    so the trimming *is* differentiated in ``outputs``: the trimmed mean is
    piecewise linear in its inputs and autograd's gradient is correct except on
    the measure-zero set where two entries tie. What is BPDA here is membership --
    ``assignments`` -- and the integer trim counts derived from it.
    """
    num_realizations, num_samples, output_dim = outputs.shape
    num_clusters = cluster.num_clusters

    positions = torch.arange(num_samples, device=outputs.device).view(1, num_samples, 1)
    infinity = torch.tensor(torch.inf, device=outputs.device, dtype=outputs.dtype)
    zero = torch.zeros(1, device=outputs.device, dtype=outputs.dtype)

    atoms = torch.zeros(
        (num_realizations, num_clusters, output_dim),
        device=outputs.device,
        dtype=outputs.dtype,
    )
    for j in range(num_clusters):
        mask = assignments == j
        values = torch.where(mask.unsqueeze(-1), outputs, infinity)
        # Out-of-cluster entries sort to the end, so the trimmed core below only
        # ever indexes genuine members of cluster j.
        ordered = values.sort(dim=1).values

        counts = mask.sum(dim=1)
        left = torch.floor(alpha * counts.to(outputs.dtype)).to(torch.long)
        right = counts - left

        in_core = (positions >= left.view(num_realizations, 1, 1)) & (
            positions < right.view(num_realizations, 1, 1)
        )
        trimmed_sum = torch.where(in_core, ordered, zero).sum(dim=1)
        trimmed_count = (right - left).to(outputs.dtype).view(num_realizations, 1)

        atoms[:, j, :] = torch.where(
            (trimmed_count.squeeze(-1) > 0).unsqueeze(-1),
            trimmed_sum / trimmed_count.clamp(min=1),
            cluster.centroids[0, j, :].unsqueeze(0),
        )
    return atoms


def policy_outputs_under_smoothing(
    policy: VariationalPolicy,
    x: torch.Tensor,
    smoothing_noise: torch.Tensor,
    gumbel_noise: torch.Tensor,
    component_noise: torch.Tensor,
    gate_temperature: float = _GATE_TEMPERATURE,
) -> torch.Tensor:
    """The ``(M, N, A)`` actions the policy emits on the smoother's noisy inputs.

    ``smoothing_noise`` is ``(M, N, d)``, ``gumbel_noise`` is ``(M, N, K)`` and
    ``component_noise`` is ``(M, N, A)`` -- all fixed across the PGD run (common
    random numbers), so the result is a deterministic function of ``x``.

    The gate draw is straight-through Gumbel-max, so forward these are exactly
    samples from ``pi(.|x + eta)`` while the backward pass carries the gate's
    response to ``x``. ``K = 1`` short-circuits: the gate is degenerate there and
    ``log_mix_weights`` is identically zero.
    """
    num_realizations, num_samples, _ = smoothing_noise.shape
    flat = num_realizations * num_samples
    noisy = (x + smoothing_noise).reshape(flat, -1)

    means, log_stds, log_mix_weights = policy.distribution_params(noisy)
    stds = log_stds.exp()
    if policy.num_components == 1:
        selected_mean, selected_std = means[:, 0], stds[:, 0]
    else:
        one_hot = _gumbel_softmax_components(
            log_mix_weights, gumbel_noise.reshape(flat, -1), gate_temperature
        )
        selected_mean = (one_hot.unsqueeze(-1) * means).sum(dim=1)
        selected_std = (one_hot.unsqueeze(-1) * stds).sum(dim=1)

    raw = selected_mean + selected_std * component_noise.reshape(flat, -1)
    return (torch.tanh(raw) * policy.action_scale).view(
        num_realizations, num_samples, -1
    )


def smoothed_action_atoms(
    policy: VariationalPolicy,
    smoother: GaussianAlphaTrimmingSmoothing,
    x: torch.Tensor,
    smoothing_noise: torch.Tensor,
    gumbel_noise: torch.Tensor,
    component_noise: torch.Tensor,
    gate_temperature: float = _GATE_TEMPERATURE,
    partition_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The smoothed controller's emitted distribution at ``x``.

    Returns ``atoms`` ``(M, k, A)`` and ``weights`` ``(M, k)`` -- the ``locs`` and
    ``probs`` of the ``MixtureDirac`` that ``AlphaTrimmingSmoothing.smooth``
    returns, for each of ``M`` realizations of the smoothing draw. Both respond to
    ``x``: the atoms through the policy's actions, the weights through region
    occupancy.

    Only the cluster *geometry* is frozen at the smoother's clean anchor --
    centroids and regions. The executed controller re-clusters at the perturbed
    input; carrying that inside every PGD step would mean a DBSCAN fit per step.

    One ``regions.distance`` evaluation is shared between the hard assignment the
    alpha-trimming needs and the soft occupancy the weights need.
    """
    outputs = policy_outputs_under_smoothing(
        policy, x, smoothing_noise, gumbel_noise, component_noise, gate_temperature
    )
    num_realizations, num_samples, output_dim = outputs.shape

    cluster = smoother.cluster
    distances = cluster.regions.distance(
        outputs.reshape(1, num_realizations * num_samples, output_dim)
    ).view(num_realizations, num_samples, cluster.num_clusters)

    atoms = _alpha_trimmed_atoms(
        outputs,
        cluster,
        smoother.parameters.smoothing.alpha,
        distances.detach().argmin(dim=-1),
    )
    return atoms, _soft_partition_weights(distances, partition_temperature)


# ---------------------------------------------------------------------------
# Objective builders -- each returns a closure over fixed noise
# ---------------------------------------------------------------------------
def make_raw_objective(
    policy: VariationalPolicy,
    env: UAV3D,
    true_state: np.ndarray,
    critic: ValueNetwork,
    gamma: float,
    eot_samples: int,
    generator: torch.Generator | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x -> E_{a ~ pi(.|x), w}[ r + gamma V(x') ]`` for the raw policy."""
    device = next(policy.parameters()).device
    component_noise = torch.randn(
        (eot_samples, policy.num_components, policy.action_dim),
        device=device,
        generator=generator,
    )

    def objective(x: torch.Tensor) -> torch.Tensor:
        atoms, weights = raw_action_atoms(policy, x, component_noise)
        q = expected_q(env, true_state, atoms, critic, gamma)   # (K, M)
        return (weights * q.mean(dim=1)).sum()

    return objective


def make_smoothed_objective(
    policy: VariationalPolicy,
    smoother: GaussianAlphaTrimmingSmoothing,
    env: UAV3D,
    true_state: np.ndarray,
    critic: ValueNetwork,
    gamma: float,
    eot_samples: int,
    generator: torch.Generator | None = None,
    gate_temperature: float = _GATE_TEMPERATURE,
    partition_temperature: float | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x -> E_{a ~ smoothed(.|x), w}[ r + gamma V(x') ]`` for one smoothed branch.

    This is the emitted distribution itself, not a stand-in for it: ``a`` is drawn
    from the same ``MixtureDirac`` -- alpha-trimmed atoms, region-occupancy
    weights -- that ``AlphaTrimmingSmoothing.smooth`` returns, and both parts
    respond to ``x``. ``eot_samples`` realizations of the smoothing draw are
    averaged, all of them fixed across the PGD run.

    ``partition_temperature`` defaults to a scale derived from the clean anchor's
    own region distances; see ``_derive_partition_temperature``. Neither
    temperature changes the forward value -- both relaxations are
    straight-through -- so they are gradient knobs only.

    The occupancy weights are estimated from the same N draws as the atoms, while
    the executed controller estimates them from its (much larger)
    ``confidence.num_samples`` draw. Same estimand, coarser estimate, and the two
    are correlated here where deployment's are independent.
    """
    device = next(policy.parameters()).device
    num_samples = smoother.parameters.smoothing.N
    state_dim = smoother.x.shape[-1]

    smoothing_noise = smoother.smoothing_distribution.sample(
        size=(eot_samples, num_samples)
    ).to(device).view(eot_samples, num_samples, state_dim)

    # Standard Gumbel by inverse CDF, fixed for the whole run alongside the other
    # noise: the gate draw has to be common random numbers like everything else,
    # or the objective stops being a deterministic function of delta.
    uniforms = torch.rand(
        (eot_samples, num_samples, policy.num_components),
        device=device,
        generator=generator,
    )
    tiny = torch.finfo(uniforms.dtype).tiny
    # Clamp *after* each negation: -log(u) is the positive quantity the outer log
    # needs, and clamping the negative log(u) instead sends it through log(<0).
    exponential = (-torch.log(uniforms.clamp_min(tiny))).clamp_min(tiny)
    gumbel_noise = -torch.log(exponential)
    component_noise = torch.randn(
        (eot_samples, num_samples, policy.action_dim), device=device, generator=generator
    )

    if partition_temperature is None:
        with torch.no_grad():
            anchor_outputs = policy_outputs_under_smoothing(
                policy, smoother.x, smoothing_noise, gumbel_noise,
                component_noise, gate_temperature,
            )
        partition_temperature = _derive_partition_temperature(anchor_outputs, smoother.cluster)

    def objective(x: torch.Tensor) -> torch.Tensor:
        atoms, weights = smoothed_action_atoms(
            policy, smoother, x, smoothing_noise, gumbel_noise, component_noise,
            gate_temperature, partition_temperature,
        )
        q = expected_q(env, true_state, atoms, critic, gamma)   # (M, k)
        return (weights * q).sum(dim=1).mean()

    return objective


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------
def _random_in_ball(
    shape: tuple[int, ...],
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    direction = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    direction = direction / torch.linalg.norm(direction).clamp(min=1e-12)
    scale = torch.rand((), device=device, dtype=dtype, generator=generator)
    return radius * scale.pow(1.0 / float(np.prod(shape))) * direction


def _project_to_ball(delta: torch.Tensor, radius: float) -> torch.Tensor:
    norm = torch.linalg.norm(delta)
    if norm <= radius:
        return delta
    return delta * (radius / norm)


def pgd_attack(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    radius: float,
    steps: int,
    restarts: int = 3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Minimize ``objective(x + delta)`` over ``||delta||_2 <= radius`` by L2 PGD.

    Returns the best ``delta`` found, shaped like ``x``.

    ``delta = 0`` is both the first restart's initialization and the incumbent
    best, so the returned perturbation can never score worse than the clean input
    under the attacker's own objective -- at ``radius = 0`` the function is exactly
    a no-op. That guarantee is only worth as much as the objective's fidelity: it
    binds the returned ``delta`` to the quantity being scored here, not to
    whatever the deployed controller does, so the two had better agree.

    ``objective`` must be deterministic (built over fixed noise), otherwise the
    across-restart comparison and the ``best`` bookkeeping are noise-dominated.
    It need not be continuous, and for the smoothed branches it is not -- the
    straight-through gate index and region assignment flip as ``delta`` moves. The
    gradient is a relaxation across those flips, so the descent is a heuristic;
    the values compared here are the real ones.
    """
    zero = torch.zeros_like(x)
    if radius <= 0.0 or steps <= 0:
        return zero

    with torch.no_grad():
        best_value = objective(x)
    best_delta = zero
    step_size = _STEP_SIZE_FACTOR * radius / steps

    for restart in range(max(1, restarts)):
        delta = (
            zero.clone()
            if restart == 0
            else _random_in_ball(tuple(x.shape), radius, x.device, x.dtype, generator)
        )
        for _ in range(steps):
            delta.requires_grad_(True)
            value = objective(x + delta)
            (gradient,) = torch.autograd.grad(value, delta)
            with torch.no_grad():
                normalized = gradient / torch.linalg.norm(gradient).clamp(min=1e-12)
                delta = _project_to_ball(delta.detach() - step_size * normalized, radius)

        with torch.no_grad():
            value = objective(x + delta)
        if value < best_value:
            best_value, best_delta = value, delta.detach()

    return best_delta
