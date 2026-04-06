from .clusters import Cluster
from .dataclasses import SmoothingParameters, ClusteringParameters, RobustnessCertificate
from .distributions import (
    Distribution,
    MultivariateGaussian,
    Dirac,
    MixtureDirac,
)
from .smoothing import (
    Smoothing,
    AlphaTrimmingSmoothing,
)
from .stochastic import StochasticFunction, ThreeMode1DGaussianMixture
from .clopper_pearson import ClopperPearsonInterval

__all__ = [
    # Core abstractions
    "StochasticFunction",
    "ThreeMode1DGaussianMixture",
    "Smoothing",
    "Distribution",
    # Configuration
    "SmoothingParameters",
    "ClusteringParameters",
    # Smoothers
    "AlphaTrimmingSmoothing",
    "RobustnessCertificate",
    "ClopperPearsonInterval",
    # Clustering
    "Cluster",
    # Noise distributions
    "MultivariateGaussian",
    # Output distributions (returned by smooth())
    "Dirac",
    "MixtureDirac",
]
