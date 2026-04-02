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
        centroids = cluster.centroids
        m, n, q = samples.shape
        m_c, k, q_c = centroids.shape

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