"""Shared data schema for UAV3D rollout results.

Splits the (expensive) trajectory rollout from the (cheap, iterative) pyvista
rendering: ``collect_uav3d_rollouts.py`` runs the policy/env and writes a
``RolloutBundle`` here; ``visualize_uav3d_robustified_pyvista.py`` only ever
reads one, so camera/legend/color tweaks don't require re-simulating.

Deliberately numpy-only (no torch / pyvista) so both the collection and the
plotting script — and anything else that just needs the fixed bimodal start
state — can import this without pulling in the other's heavy dependencies.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# State where the trained K=2 policy is sharply bimodal AND the raw policy is
# "clutch": a substantial fraction of trajectories collide, so the experiment
# shows a non-trivial failure mode that randomized smoothing can mitigate. From
# this far-west position the two mixture components encode two distinct route
# bundles toward the goal in the far-east corner.
#
# Selected by ``find_bimodal_state.py`` against the current ``policy.pt``, at
# alpha = 0.2. Measured over 200 rollouts per branch:
#
#   branch          goal   collision   discounted return
#   raw            0.695     0.305           +14.26
#   alpha (k=1)    0.535     0.465            +9.71   <- worse than no defense
#   robust (k=3)   0.980     0.020           +20.66
#
#   minor-mode mass 0.242, separation 7.86, z = 0, x = -14.8 <= -11, velocity 0.
#
# The alpha branch collapses the bimodal action distribution onto one atom
# between the two modes and flies into an obstacle nearly half the time -- worse
# than the undefended policy -- while clustered smoothing keeps one atom per mode
# and reaches the goal 98% of the time. That contrast is the experiment.
#
# **alpha must stay below the minor-mode mass (0.242).** Alpha-trimming averages
# the modes only while some minority samples survive the trim. At alpha >= 0.242
# the minority mode is removed outright and k=1 becomes a mode *selector*: at
# alpha=0.4 this same state gives the alpha branch 97% goal / 3% collision, and
# the phenomenon vanishes. Measured collision rate vs alpha here:
#   alpha  0.05  0.10  0.15  0.20 | 0.30  0.40
#   coll   0.37  0.44  0.47  0.49 | 0.23  0.03
#                                 ^ transition at alpha = minor mass
#
# Requirements each figure above is checked against (see find_bimodal_state.py):
#   zero velocity, z = 0, x <= -11 (behind every obstacle face, 28.0 from the
#   goal), coordinates multiples of 0.1, minor-mode mass >= 0.20, mode
#   separation >= 1.5, raw goal rate >= 0.70, alpha collision rate >= 0.40.
#
# **Re-run find_bimodal_state.py after any retrain.** This constant is tied to one
# checkpoint; the gate is re-initialised and re-optimised each time. The original
# value ([-13.5, 0, -3.0, 0, -5.0, 0]) fell to minor-mode mass 0.008 under the
# current policy -- effectively unimodal, where k=1 and k=3 provably coincide.
BIMODAL_START_STATE = np.array([-14.8, 0.0, 0.7, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class PolicyRolloutData:
    """Everything the plotting script needs for one policy (raw / alpha /
    robust), gathered by ``collect_uav3d_rollouts.py``.
    """
    traces: list[np.ndarray]              # primary-seed positions (T+1, 3), for rendering
    outcomes: list[str]                   # primary-seed outcomes, aligned with traces
    counts: dict[str, int]                # primary-seed raw goal/collision/timeout counts
    rate_stats: dict[str, tuple[float, float]]  # {"goal": (mean, std), "collision": (mean, std)} across seeds
    accel_mean: float
    accel_std: float
    jerk_mean: float
    jerk_std: float
    n_kinematic_samples: int


@dataclass
class RolloutBundle:
    meta: dict
    policies: dict[str, PolicyRolloutData]


def save_rollouts(bundle: RolloutBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_rollouts(path: Path) -> RolloutBundle:
    with open(path, "rb") as f:
        return pickle.load(f)
