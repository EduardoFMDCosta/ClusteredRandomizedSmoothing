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

# State where the trained K=2 policy is sharply bimodal AND the raw policy
# is "clutch": some trajectories collide, so the visualization shows a
# non-trivial failure mode that randomized smoothing can mitigate. From this
# far-west, low-y, low-z position the two mixture components encode two
# distinct route bundles toward the goal in the far-east, low-z corner.
# 200-trajectory rollout: ~25-unit cluster separation at quarter-horizon
# with a ~66/34 split, and ~72% goal / 24% collision / 4% timeout. Zero
# initial velocity.
BIMODAL_START_STATE = np.array([-13.5, 0.0, -3.0, 0.0, -5.0, 0.0], dtype=np.float32)


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
