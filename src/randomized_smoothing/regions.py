import torch

class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def volume(self):
        return self.width.prod(dim=-1)

    @property
    def total_volume(self):
        return self.volume.sum()

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
    def from_samples(
        samples: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        coverage: float,
    ):
        nx, _, _ = samples.shape
        k = centroids.shape[1]

        diff = samples.unsqueeze(2) - centroids.unsqueeze(1)
        assigned_diff = diff.gather(
            2,
            assignments.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, diff.shape[-1]),
        ).squeeze(2)
        assigned_linf_distances = assigned_diff.abs().amax(dim=-1)

        radii = torch.zeros((nx, k), dtype=samples.dtype, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                d_ij = assigned_linf_distances[i][mask]
                if d_ij.numel() == 0:
                    continue

                rank = max(
                    1,
                    int(
                        torch.ceil(
                            torch.tensor(
                                coverage * d_ij.numel(),
                                dtype=d_ij.dtype,
                                device=d_ij.device,
                            )
                        ).item()
                    ),
                )
                rank = min(rank, d_ij.numel())
                radii[i, j] = d_ij.kthvalue(rank).values

        lower = centroids - radii.unsqueeze(-1)
        upper = centroids + radii.unsqueeze(-1)
        return HyperRectangle(lower, upper)