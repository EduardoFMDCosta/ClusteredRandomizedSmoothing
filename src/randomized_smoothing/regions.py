import torch


class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def __len__(self):
        return self.lower.size(0)

    def size(self, dim=None):
        if dim is None:
            return self.lower.size()

        return self.lower.size(dim)

    @staticmethod
    def from_eps(x, eps):
        lower, upper = x - eps, x + eps
        return HyperRectangle(lower, upper)
    
    @staticmethod
    def from_samples(samples, cluster, p):
        # This method should return a Hyperrectangle that contains a fraction p of the samples assigned to each cluster (which comes from being closer to the centroid)
        # samples is (m, N, q), cluster.centroids is (m, k, q), and p is a float in (0, 1)
        # The method should return a HyperRectangle with lower and upper bounds of shape (m, k, q) that contains a fraction p of the samples assigned to each cluster. This can be done by computing the distance of each sample to its assigned centroid, then finding the radius that contains fraction p of those distances, and finally constructing the hyperrectangle around the centroid with that radius.
        if not isinstance(samples, torch.Tensor):
            raise TypeError("samples must be a torch.Tensor")
        if samples.ndim != 3:
            raise ValueError("samples must have shape (m, N, q)")
        if not (0.0 < p < 1.0):
            raise ValueError("p must be in (0, 1)")

        centroids = cluster.centroids
        if not isinstance(centroids, torch.Tensor) or centroids.ndim != 3:
            raise ValueError("cluster.centroids must have shape (m, k, q)")

        m, n, q = samples.shape
        m_c, k, q_c = centroids.shape
        if m != m_c or q != q_c:
            raise ValueError("samples and cluster.centroids must agree on m and q dimensions")

        distances = torch.cdist(samples, centroids)  # (m, N, k)
        assignments = distances.argmin(dim=2)  # (m, N)
        assigned_distances = distances.gather(2, assignments.unsqueeze(-1)).squeeze(-1)  # (m, N)

        radii = torch.zeros((m, k), dtype=samples.dtype, device=samples.device)

        for i in range(m):
            for j in range(k):
                mask = assignments[i] == j
                d_ij = assigned_distances[i][mask]
                if d_ij.numel() == 0:
                    continue

                rank = max(1, int(torch.ceil(torch.tensor(p * d_ij.numel())).item()))
                rank = min(rank, d_ij.numel())
                radii[i, j] = d_ij.kthvalue(rank).values

        lower = centroids - radii.unsqueeze(-1)
        upper = centroids + radii.unsqueeze(-1)
        return HyperRectangle(lower, upper)