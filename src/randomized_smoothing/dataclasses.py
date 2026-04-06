from dataclasses import dataclass
import torch

@dataclass
class SmoothingParameters:
    N: int
    alpha: float

    def __post_init__(self) -> None:
        self.N = int(self.N)
        self.alpha = float(self.alpha)

        if self.N <= 0:
            raise ValueError("N must be positive")
        if not (0.0 <= self.alpha < 0.5):
            raise ValueError("alpha must satisfy 0 <= alpha < 0.5")

@dataclass
class ClusteringParameters:
    num_clusters: int
    coverage: float
    max_iter: int = 1000
    tol: float = 1e-5

    def __post_init__(self) -> None:
        self.num_clusters = int(self.num_clusters)
        self.coverage = float(self.coverage)
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)

        if self.num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if not (0.0 < self.coverage <= 1.0):
            raise ValueError("coverage must satisfy 0 < coverage <= 1")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

@dataclass
class ConfidenceParameters:
    num_samples: int = 1000
    beta: float = 1e-2

    def __post_init__(self) -> None:
        self.num_samples = int(self.num_samples)
        self.beta = float(self.beta)

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must satisfy 0 < beta < 1")

@dataclass
class Parameters:
    smoothing: SmoothingParameters
    clustering: ClusteringParameters
    confidence: ConfidenceParameters


@dataclass
class RobustnessCertificate:
    radius: torch.Tensor
    probs: torch.Tensor