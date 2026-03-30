import torch

from randomized_smoothing.stochastic import StochasticFunction
from randomized_smoothing.utils import _run_kmeans, _select_num_clusters_elbow


class Cluster:
    def __init__(
        self,
        stochastic_function: StochasticFunction,
        x: torch.Tensor,
        *,
        num_samples: int = 1000,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        self.stochastic_function = stochastic_function
        self.x = x

        self.max_iter = max_iter
        self.tol = tol

        self.centroids = self._compute_centroids(num_samples=num_samples)
        self.weights = self._compute_weights()

    def _compute_centroids(self, num_samples: int) -> torch.Tensor:
        outputs = self.stochastic_function.sample(self.x, num_samples)

        m, n, q = outputs.shape
        pooled = outputs.reshape(m * n, q)
        k = _select_num_clusters_elbow(
            pooled,
            max_k=self.max_num_clusters,
            max_iter=self.max_iter,
                tol=self.tol,
            )

        centroids = torch.empty((m, k, q))
        for i in range(m):
            centroids_i, _, _ = _run_kmeans(
                outputs[i],
                num_clusters=k,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            centroids[i] = centroids_i
        return centroids
    
    def _compute_weights(self) -> torch.Tensor:
        pass
