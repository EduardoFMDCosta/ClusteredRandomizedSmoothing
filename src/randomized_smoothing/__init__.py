from .clusters import Cluster
from .dataclasses import (
    SmoothingParameters,
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    RobustnessCertificate,
)
from .distributions import (
    Distribution,
    MultivariateGaussian,
    Dirac,
    MixtureDirac,
)
from .smoothing import (
    Smoothing,
    AlphaTrimmingSmoothing,
    GaussianAlphaTrimmingSmoothing,
)
from .stochastic import StochasticFunction
from .clopper_pearson import ClopperPearsonInterval
from .optimization import compute_total_prob_lower_bound, eval_f_m, robustness_bound

__all__ = [
    # Core abstractions
    "StochasticFunction",
    "Smoothing",
    "Distribution",
    # Configuration
    "SmoothingParameters",
    "ClusteringParameters",
    "ConfidenceParameters",
    "Parameters",
    # Smoothers
    "AlphaTrimmingSmoothing",
    "GaussianAlphaTrimmingSmoothing",
    "RobustnessCertificate",
    "ClopperPearsonInterval",
    "compute_total_prob_lower_bound",
    "eval_f_m",
    "robustness_bound",
    # Clustering
    "Cluster",
    # Noise distributions
    "MultivariateGaussian",
    # Output distributions (returned by smooth())
    "Dirac",
    "MixtureDirac",
]
