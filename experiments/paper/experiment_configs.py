"""experiment_configs.py — Named parameter sets for all TrajFlow paper experiments.

To run with different settings, either:
  1. Edit the corresponding dataclass below and re-run.
  2. Import a different named preset, e.g. ``EXAMPLES_TRAJECTORY`` instead of
     ``EXAMPLES``.

All experiment scripts import exactly one named config from this file.
Runtime arguments (which data rows to process, where to save results, whether to
regenerate) are still accepted on the command line of each experiment script.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _SmootherParams:
    """Parameters shared by all TrajFlow smoothing experiments."""
    sigma: float = 1.0
    """Gaussian input perturbation std."""
    alpha: float = 0.35
    """Trimming fraction alpha for the alpha-trimmed mean."""
    smoothing_N: int = 8
    """Number of noisy model samples used to form one smoothed prediction."""
    coverage: float = 0.85
    """Probability mass that the fitted cluster regions must cover."""
    cluster_strategy: str = "behavior_split"
    """Clustering strategy for the k=2 smoother."""
    behavior_stay_radius: float = 1.5
    """Endpoint displacement radius (m) below which a trajectory is 'stay'."""
    behavior_min_down: float = 0.5
    """behavior_split: min downward displacement (m) for a 'turn' trajectory."""
    behavior_min_left: float = 0.0
    """behavior_split: min leftward displacement (m) for a 'turn' trajectory."""
    dbscan_eps: float = 0.5
    """DBSCAN epsilon in output space."""
    dbscan_min_samples: int = 200
    """DBSCAN min_samples."""
    noise_sub_batch_size: int = 200
    """Max noise samples processed per sub-batch during forward passes."""
    input_mode: str = "control"
    """Input representation fed to the smoother: 'control' or 'position'."""
    seed: int = 0
    """Random seed for reproducibility."""
    input_batch_size: int = 10
    """Dataset inputs processed together in one computation batch."""
    confidence_num_samples: int = 100
    """Samples drawn to fit cluster boundaries.
    100 is sufficient for all experiments that do NOT call robust_certification.
    trajflow_robustness overrides this to 2000 for tight Clopper-Pearson bounds."""


@dataclass
class ExamplesConfig(_SmootherParams):
    """Parameters for trajflow_examples.py — per-scenario smoother visualisation."""
    selected_indices: str = "157,203,153,2,41"
    """Comma-separated dataset row indices to visualise."""
    output_mode: str = "trajectory"
    """'endpoint' uses 2-D final position; 'trajectory' uses full 40-D path."""
    num_plot_samples: int = 500
    """MC samples drawn from each smoother for the example scatter plots."""
    # Overrides relative to _SmootherParams:
    smoothing_N: int = 10
    coverage: float = 0.80
    behavior_stay_radius: float = 1.0
    cluster_strategy: str = "behavior_split"


@dataclass
class WassersteinConfig(_SmootherParams):
    """Parameters for trajflow_wasserstein.py — W2 distance to clean h_w(x)."""
    output_mode: str = "trajectory"
    """'endpoint' or 'trajectory' (trajectory also produces a per-timestep plot)."""
    distribution_samples: int = 150
    """Samples drawn from each distribution to estimate W2."""
    ot_method: str = "sinkhorn_log"
    """Optimal-transport solver: 'emd' or 'sinkhorn_log'."""
    sinkhorn_reg: float = 0.1
    """Sinkhorn regularisation (only used when ot_method='sinkhorn_log')."""
    hist_bins: int = 24
    """Number of histogram bins in the W2 comparison figure."""
    timestep_eval_indices: list[int] = field(default_factory=lambda: [1, 5, 10, 15, 19])
    """0-based prediction timestep indices at which to evaluate per-step W2."""
    # Overrides:
    coverage: float = 0.80
    cluster_strategy: str = "behavior_split"


@dataclass
class SafetyConfig(_SmootherParams):
    """Parameters for trajflow_safety.py — decision-making risk analysis."""
    mc_samples: int = 300
    """Monte Carlo simulations of the smoother used to estimate P(stays)."""
    decision_threshold: float = 0.95
    """Ego decides to go when P(other stays) >= decision_threshold.
    0.95 is the threshold reported in the paper (Section 5, safety analysis).
    It only enters the analysis and the plot, not the Monte-Carlo rows, so it
    can also be changed at run time with ``--decision-threshold``."""


@dataclass
class RobustnessConfig(_SmootherParams):
    """Parameters for trajflow_robustness.py — certified probability bounds.

    confidence_num_samples is deliberately large: the Clopper-Pearson lower
    bounds on probs_region and probs_partition are the primary output, and
    require many samples to be statistically tight.
    """
    alpha_values: list[float] = field(default_factory=lambda: [0.4, 0.2, 0.0])
    """Alpha trimming values to sweep over."""
    radius_values: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0])
    """Robustness radii (input-space L2) to certify."""
    # Overrides:
    smoothing_N: int = 5
    confidence_num_samples: int = 2000   # must stay large for tight CP bounds
    input_batch_size: int = 5


# ---------------------------------------------------------------------------
# Output mode switch — the only line you need to edit to change mode.
# "trajectory" : full 40-D predicted path (20 timesteps × 2 coords)
# "endpoint"   : 2-D final position only
# ---------------------------------------------------------------------------
OUTPUT_MODE: str = "trajectory"

# Concrete presets for each mode — do not edit these directly.
EXAMPLES_TRAJECTORY = ExamplesConfig()
EXAMPLES_ENDPOINT = ExamplesConfig(
    output_mode="endpoint",
    cluster_strategy="behavior_split",
    dbscan_eps=0.5,
)

WASSERSTEIN_TRAJECTORY = WassersteinConfig()
WASSERSTEIN_ENDPOINT = WassersteinConfig(
    output_mode="endpoint",
    cluster_strategy="behavior_split",
    dbscan_eps=0.5,
)

# Active presets imported by each script — automatically follow OUTPUT_MODE.
EXAMPLES = EXAMPLES_TRAJECTORY if OUTPUT_MODE == "trajectory" else EXAMPLES_ENDPOINT
WASSERSTEIN = WASSERSTEIN_TRAJECTORY if OUTPUT_MODE == "trajectory" else WASSERSTEIN_ENDPOINT

SAFETY = SafetyConfig()

ROBUSTNESS = RobustnessConfig()
