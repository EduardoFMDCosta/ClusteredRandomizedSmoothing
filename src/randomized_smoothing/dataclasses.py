from dataclasses import dataclass, field

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
    max_num_clusters: int
    coverage: float
    strategy: str = "dbscan"
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_cluster_selection_method: str = "eom"
    behavior_stay_radius: float = 1.0
    behavior_min_down: float = 0.5
    behavior_min_left: float = 0.0
    behavior_force_two_clusters: bool = True
    max_iter: int = 1000
    tol: float = 1e-5
    elbow_threshold: float = 0.05

    def __post_init__(self) -> None:
        self.max_num_clusters = int(self.max_num_clusters)
        self.coverage = float(self.coverage)
        self.strategy = str(self.strategy)
        self.dbscan_eps = float(self.dbscan_eps)
        self.dbscan_min_samples = int(self.dbscan_min_samples)
        self.hdbscan_min_cluster_size = int(self.hdbscan_min_cluster_size)
        if self.hdbscan_min_samples is not None:
            self.hdbscan_min_samples = int(self.hdbscan_min_samples)
        self.hdbscan_cluster_selection_epsilon = float(self.hdbscan_cluster_selection_epsilon)
        self.hdbscan_cluster_selection_method = str(self.hdbscan_cluster_selection_method)
        self.behavior_stay_radius = float(self.behavior_stay_radius)
        self.behavior_min_down = float(self.behavior_min_down)
        self.behavior_min_left = float(self.behavior_min_left)
        self.behavior_force_two_clusters = bool(self.behavior_force_two_clusters)
        self.max_iter = int(self.max_iter)
        self.tol = float(self.tol)
        self.elbow_threshold = float(self.elbow_threshold)

        if self.max_num_clusters <= 0:
            raise ValueError("max_num_clusters must be positive")
        if not (0.0 < self.coverage <= 1.0):
            raise ValueError("coverage must satisfy 0 < coverage <= 1")
        if self.strategy not in ("dbscan", "hdbscan", "behavior_split", "trajectory_dbscan"):
            raise ValueError("strategy must be one of: 'dbscan', 'hdbscan', 'behavior_split', 'trajectory_dbscan'")
        if self.dbscan_eps <= 0.0:
            raise ValueError("dbscan_eps must be positive")
        if self.dbscan_min_samples <= 0:
            raise ValueError("dbscan_min_samples must be positive")
        if self.hdbscan_min_cluster_size <= 1:
            raise ValueError("hdbscan_min_cluster_size must be > 1")
        if self.hdbscan_min_samples is not None and self.hdbscan_min_samples <= 0:
            raise ValueError("hdbscan_min_samples must be positive")
        if self.hdbscan_cluster_selection_epsilon < 0.0:
            raise ValueError("hdbscan_cluster_selection_epsilon must be non-negative")
        if self.hdbscan_cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("hdbscan_cluster_selection_method must be 'eom' or 'leaf'")
        if self.behavior_stay_radius < 0.0:
            raise ValueError("behavior_stay_radius must be non-negative")
        if self.behavior_min_down < 0.0:
            raise ValueError("behavior_min_down must be non-negative")
        if self.behavior_min_left < 0.0:
            raise ValueError("behavior_min_left must be non-negative")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if not (0.0 < self.elbow_threshold < 1.0):
            raise ValueError("elbow_threshold must satisfy 0 < elbow_threshold < 1")

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
class RuntimeHeuristics:
    noise_sub_batch_size: int = 300

    def __post_init__(self) -> None:
        self.noise_sub_batch_size = int(self.noise_sub_batch_size)
        if self.noise_sub_batch_size <= 0:
            raise ValueError("noise_sub_batch_size must be positive")

@dataclass
class Parameters:
    smoothing: SmoothingParameters
    clustering: ClusteringParameters
    confidence: ConfidenceParameters
    runtime: RuntimeHeuristics = field(default_factory=RuntimeHeuristics)


@dataclass
class RobustnessCertificate:
    radius: torch.Tensor
    probs: torch.Tensor
    total_probs: torch.Tensor