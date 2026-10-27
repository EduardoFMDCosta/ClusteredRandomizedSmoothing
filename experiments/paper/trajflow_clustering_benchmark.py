"""
trajflow_clustering_benchmark.py  —  behavior_split vs DBSCAN timing.

Pure clustering comparison on synthetic bimodal data.
No model loading required; runs in < 1 minute.

Data geometry mirrors TrajFlow outputs:
  2D  — endpoint (x, y) of the predicted trajectory
  40D — full predicted trajectory (20 timesteps × 2 coords), linearly
        interpolated from origin to a bimodal endpoint

Two clusters per dataset:
  stay — target vehicle slows / stays near origin
  go   — target vehicle turns left and moves away

Compared algorithms:
  behavior_split — our O(N) heuristic (full Cluster pipeline)
  sklearn DBSCAN — standard density-based algorithm (fit_predict only)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
for _p in [str(SRC), str(EXPERIMENTS), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from experiment_configs import ROBUSTNESS as cfg

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N: int = cfg.confidence_num_samples          # 2000  — same as real experiments
N_WARMUP: int = 3
N_REPS: int = 30

# Noise std for synthetic data.  In 2D, within-cluster mean distance ≈ 0.35,
# safely below eps=0.5.  In 40D, within-cluster mean distance ≈ 1.79, safely
# below eps=3.0.
NOISE_STD: float = 0.20

# Endpoints defining the two clusters (stay and go).
# "stay": near origin, displacement norm 0.42 << behavior_stay_radius=1.5
# "go":   down-left, satisfies delta_y<=-0.5 and delta_x<=0 for behavior_split
STAY_ENDPOINT = np.array([0.3,  0.3])
GO_ENDPOINT   = np.array([-3.0, -4.0])

# behavior_split anchor = trajectory start position (origin)
ANCHOR = np.zeros((1, 2), dtype=np.float32)

# DBSCAN parameters
DBSCAN_EPS_2D:  float = cfg.dbscan_eps          # 0.5  — real experiment value
DBSCAN_EPS_40D: float = 3.0                      # recalibrated for 40D geometry
DBSCAN_MIN_SAMPLES: int = cfg.dbscan_min_samples # 200  — real experiment value

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_2d_bimodal(rng: np.random.Generator) -> np.ndarray:
    """Synthetic 2D endpoint data: N/2 stay + N/2 go samples."""
    n_stay = N // 2
    stay = rng.normal(STAY_ENDPOINT, NOISE_STD, size=(n_stay, 2))
    go   = rng.normal(GO_ENDPOINT,   NOISE_STD, size=(N - n_stay, 2))
    return np.vstack([stay, go]).astype(np.float32)


def make_40d_bimodal(rng: np.random.Generator) -> np.ndarray:
    """Synthetic 40D trajectory data: linearly interpolated from origin to endpoint.

    Dimension layout: [x₁, y₁, x₂, y₂, …, x₂₀, y₂₀]  (20 timesteps × 2 coords).
    The last 2 dims (x₂₀, y₂₀) are the endpoint, matching the 2D case.
    """
    frac = np.arange(1, 21) / 20.0  # timestep fractions: [0.05, …, 1.00]

    # Stay mean trajectory
    stay_mean = np.zeros(40)
    stay_mean[0::2] = STAY_ENDPOINT[0] * frac
    stay_mean[1::2] = STAY_ENDPOINT[1] * frac

    # Go mean trajectory
    go_mean = np.zeros(40)
    go_mean[0::2] = GO_ENDPOINT[0] * frac
    go_mean[1::2] = GO_ENDPOINT[1] * frac

    n_stay = N // 2
    stay = rng.normal(stay_mean, NOISE_STD, size=(n_stay, 40))
    go   = rng.normal(go_mean,   NOISE_STD, size=(N - n_stay, 40))
    return np.vstack([stay, go]).astype(np.float32)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class TimingStat(NamedTuple):
    mean_ms: float
    std_ms: float

    def __str__(self) -> str:
        return f"{self.mean_ms:7.2f} ± {self.std_ms:.2f} ms"


def _time(fn, n_warmup: int, n_reps: int) -> TimingStat:
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1_000)
    a = np.array(ts)
    return TimingStat(float(a.mean()), float(a.std(ddof=1)))


# ---------------------------------------------------------------------------
# Clustering wrappers
# ---------------------------------------------------------------------------

def _behavior_split_params() -> ClusteringParameters:
    return ClusteringParameters(
        max_num_clusters=2,
        coverage=cfg.coverage,
        strategy="behavior_split",
        dbscan_eps=cfg.dbscan_eps,
        dbscan_min_samples=cfg.dbscan_min_samples,
        behavior_stay_radius=cfg.behavior_stay_radius,
        behavior_min_down=cfg.behavior_min_down,
        behavior_min_left=cfg.behavior_min_left,
    )


def time_behavior_split(data_np: np.ndarray) -> tuple[TimingStat, int]:
    """Run full Cluster() pipeline with behavior_split; return timing and k found."""
    params = _behavior_split_params()
    anchor = torch.from_numpy(ANCHOR)
    outputs = torch.from_numpy(data_np).unsqueeze(0)  # (1, N, d)

    result_k: list[int] = []

    def _run() -> None:
        c = Cluster(outputs=outputs, parameters=params, behavior_anchor=anchor)
        result_k.clear()
        result_k.append(int(c.k_per_row.max().item()))

    stat = _time(_run, N_WARMUP, N_REPS)
    return stat, result_k[0]


def time_dbscan(data_np: np.ndarray, eps: float) -> tuple[TimingStat, int]:
    """Run sklearn DBSCAN fit_predict; return timing and cluster count found."""
    result_k: list[int] = []

    def _run() -> None:
        labels = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            algorithm="auto",
        ).fit_predict(data_np)
        result_k.clear()
        result_k.append(len(set(labels.tolist()) - {-1}))

    stat = _time(_run, N_WARMUP, N_REPS)
    return stat, result_k[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(cfg.seed)

    print("=" * 60)
    print("Clustering benchmark: behavior_split vs DBSCAN")
    print(f"N={N}, noise_std={NOISE_STD}, {N_REPS} reps ({N_WARMUP} warmup)")
    print("=" * 60)

    # ---- generate data ----
    data_2d  = make_2d_bimodal(rng)
    data_40d = make_40d_bimodal(rng)

    # sanity: confirm endpoint of 40D data matches 2D structure
    endpoint_40d = data_40d[:, -2:]
    between = np.linalg.norm(
        endpoint_40d[: N // 2].mean(axis=0) - endpoint_40d[N // 2 :].mean(axis=0)
    )
    print(f"\nSynthetic data: stay_center={data_2d[:N//2].mean(axis=0).round(2)}, "
          f"go_center={data_2d[N//2:].mean(axis=0).round(2)}")
    print(f"40D endpoint gap between cluster means: {between:.2f} m")

    # ---- 2D ----
    print(f"\n{'─'*60}")
    print(f"2D data  (endpoint, {N} samples, eps_dbscan={DBSCAN_EPS_2D})")
    print(f"{'─'*60}")

    stat_bs_2d, k_bs_2d = time_behavior_split(data_2d)
    stat_db_2d, k_db_2d = time_dbscan(data_2d, DBSCAN_EPS_2D)

    print(f"  behavior_split : {stat_bs_2d}  [k={k_bs_2d}]")
    print(f"  DBSCAN         : {stat_db_2d}  [k={k_db_2d}]")
    ratio_2d = stat_db_2d.mean_ms / max(stat_bs_2d.mean_ms, 1e-9)
    print(f"  → DBSCAN is ×{ratio_2d:.1f} slower")

    # ---- 40D ----
    print(f"\n{'─'*60}")
    print(f"40D data (trajectory, {N} samples, eps_dbscan={DBSCAN_EPS_40D})")
    print(f"{'─'*60}")

    stat_bs_40d, k_bs_40d = time_behavior_split(data_40d)
    stat_db_40d, k_db_40d = time_dbscan(data_40d, DBSCAN_EPS_40D)

    print(f"  behavior_split : {stat_bs_40d}  [k={k_bs_40d}]")
    print(f"  DBSCAN         : {stat_db_40d}  [k={k_db_40d}]")
    ratio_40d = stat_db_40d.mean_ms / max(stat_bs_40d.mean_ms, 1e-9)
    print(f"  → DBSCAN is ×{ratio_40d:.1f} slower")

    # ---- summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<18} {'2D':>18}  {'40D':>18}")
    print(f"  {'─'*56}")
    print(f"  {'behavior_split':<18} {str(stat_bs_2d):>18}  {str(stat_bs_40d):>18}")
    print(f"  {'DBSCAN':<18} {str(stat_db_2d):>18}  {str(stat_db_40d):>18}")
    print(f"  {'─'*56}")
    print(f"  {'DBSCAN / behavior_split':<18} {f'×{ratio_2d:.1f}':>18}  {f'×{ratio_40d:.1f}':>18}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
