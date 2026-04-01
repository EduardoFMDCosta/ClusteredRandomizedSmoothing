from .clusters import Cluster, ClusterParameters
from .distributions import (
    Distribution,
    MultivariateGaussian,
    Dirac,
    MixtureDirac,
)
from .smoothing import (
    Smoothing,
    SmoothingParameters,
    AlphaTrimmingSmoothing,
    ClusterizedAlphaTrimmingSmoothing,
)
from .stochastic import StochasticFunction
from .bounds import (
    robust_certification,
    clusterized_robust_certification,
    clopper_pearson_tail_probability,
    lower_bound_clopper_pearson,
)

__all__ = [
    # Core abstractions
    "StochasticFunction",
    "Smoothing",
    "Distribution",
    # Configuration
    "SmoothingParameters",
    "ClusterParameters",
    # Smoothers
    "AlphaTrimmingSmoothing",
    "ClusterizedAlphaTrimmingSmoothing",
    # Clustering
    "Cluster",
    # Noise distributions
    "MultivariateGaussian",
    # Output distributions (returned by smooth())
    "Dirac",
    "MixtureDirac",
    # Bounds
    "robust_certification",
    "clusterized_robust_certification",
    "clopper_pearson_tail_probability",
    "lower_bound_clopper_pearson",
]
