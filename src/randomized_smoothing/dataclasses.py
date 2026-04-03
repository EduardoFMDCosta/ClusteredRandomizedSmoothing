from dataclasses import dataclass
import torch

@dataclass
class SmoothingParameters:
    N: int
    alpha: float

@dataclass
class ClusteringParameters:
    num_clusters: int
    region_coverage: float
    max_iter: int = 1000
    tol: float = 1e-5

@dataclass
class ConfidenceParameters:
    num_samples: int = 1000
    beta: float = 1e-2

@dataclass
class Parameters:
    smoothing: SmoothingParameters
    clustering: ClusteringParameters
    confidence: ConfidenceParameters

@dataclass
class RobustnessCertificate:
    certified_radius: float
    certified_probs: torch.Tensor