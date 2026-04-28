import torch


def _coverage_rank(num_points: int, coverage: float) -> int:
    rank = max(1, int(torch.ceil(torch.tensor(coverage * num_points)).item()))
    return min(rank, num_points)


def _assigned_diff(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    diff = samples.unsqueeze(2) - centroids.unsqueeze(1)
    return diff.gather(
        2,
        assignments.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, diff.shape[-1]),
    ).squeeze(2)


def _assigned_abs_diff(
    samples: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    return _assigned_diff(samples=samples, centroids=centroids, assignments=assignments).abs()

class HyperRectangle:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    @property
    def width(self):
        return self.upper - self.lower
    
    @property
    def log_volume(self):
        widths = self.width.to(torch.float64)
        log_widths = torch.where(
            widths > 0,
            widths.log(),
            torch.full_like(widths, float("-inf")),
        )
        return log_widths.sum(dim=-1)

    @property
    def volume(self):
        return torch.exp(self.log_volume)

    @property
    def center(self):
        return (self.upper + self.lower) / 2

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        in_lower = points.unsqueeze(2) >= self.lower.unsqueeze(1)
        in_upper = points.unsqueeze(2) <= self.upper.unsqueeze(1)
        return (in_lower & in_upper).all(dim=-1)

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
        nx, _, output_dim = samples.shape
        k = centroids.shape[1]

        assigned_abs_diff = _assigned_abs_diff(
            samples=samples,
            centroids=centroids,
            assignments=assignments,
        )

        # Cube formulation: L∞ distance (max absolute coordinate difference per point)
        half_widths = torch.zeros((nx, k, output_dim), dtype=samples.dtype, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                linf_distances_ij = assigned_abs_diff[i][mask].amax(dim=-1)  # L∞ distance per point
                if linf_distances_ij.numel() == 0:
                    continue

                rank = _coverage_rank(linf_distances_ij.shape[0], coverage)
                radius = linf_distances_ij.kthvalue(rank).values  # scalar radius
                half_widths[i, j] = radius  # broadcast to all dimensions

        lower = centroids - half_widths
        upper = centroids + half_widths
        return HyperRectangle(lower, upper)



class AxisAlignedHyperRectangle(HyperRectangle):
    """Per-axis tight axis-aligned box per cluster.

    Differs from :class:`HyperRectangle` in how :meth:`from_samples` is built:
    instead of producing a cube whose half-width is the L∞ ``coverage``-th
    rank, this drops the ``1 - coverage`` fraction of L2-furthest points from
    the centroid and then takes per-axis min/max over the remaining samples.
    The result is generally a non-cubic axis-aligned box that hugs the bulk
    of the cluster more tightly along thin directions.
    """

    @staticmethod
    def from_samples(
        samples: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        coverage: float,
    ):
        nx = samples.shape[0]
        k = centroids.shape[1]

        assigned_l2_distances = torch.linalg.vector_norm(
            _assigned_diff(samples=samples, centroids=centroids, assignments=assignments),
            dim=-1,
        )

        lower = centroids.clone()
        upper = centroids.clone()
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                if not mask.any():
                    continue

                d_ij = assigned_l2_distances[i][mask]
                samples_ij = samples[i][mask]

                rank = _coverage_rank(d_ij.numel(), coverage)
                threshold = d_ij.kthvalue(rank).values
                keep_mask = d_ij <= threshold
                kept = samples_ij[keep_mask]
                if kept.numel() == 0:
                    continue

                lower[i, j] = kept.amin(dim=0)
                upper[i, j] = kept.amax(dim=0)

        return AxisAlignedHyperRectangle(lower=lower, upper=upper)


class TimeStepHyperrectangles(HyperRectangle):
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        if lower.shape[-1] % 2 != 0:
            raise ValueError("Expected last dimension arranged as [x1,y1,...,xT,yT]")
        super().__init__(lower, upper)

    @property
    def timestep_width(self):
        return self.width.view(*self.width.shape[:-1], self.width.shape[-1] // 2, 2)

    @property
    def timestep_area(self):
        timestep_width = self.timestep_width.to(torch.float64)
        return timestep_width[..., 0] * timestep_width[..., 1]

    @staticmethod
    def from_samples(
        samples: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        coverage: float,
    ):
        nx, _, output_dim = samples.shape
        k = centroids.shape[1]
        if output_dim % 2 != 0:
            raise ValueError("Expected last dimension arranged as [x1,y1,...,xT,yT]")

        assigned_abs_diff = _assigned_abs_diff(
            samples=samples,
            centroids=centroids,
            assignments=assignments,
        )

        half_widths = torch.zeros((nx, k, output_dim), dtype=samples.dtype, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                diff_ij = assigned_abs_diff[i][mask]
                if diff_ij.numel() == 0:
                    continue

                rank = _coverage_rank(diff_ij.shape[0], coverage)
                half_widths[i, j] = diff_ij.kthvalue(rank, dim=0).values

        lower = centroids - half_widths
        upper = centroids + half_widths
        return TimeStepHyperrectangles(lower=lower, upper=upper)
    


class EuclideanBall:
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        self.center = center
        self.radius = radius.squeeze(-1) if radius.ndim == center.ndim else radius

    @property
    def lower(self):
        return self.center - self.radius.unsqueeze(-1)

    @property
    def upper(self):
        return self.center + self.radius.unsqueeze(-1)

    @property
    def volume(self):
        dimension = self.center.shape[-1]
        radius = self.radius.to(torch.float64)
        log_prefactor = 0.5 * dimension * torch.log(torch.tensor(torch.pi, dtype=torch.float64))
        log_prefactor = log_prefactor - torch.lgamma(torch.tensor(dimension / 2 + 1, dtype=torch.float64))
        return torch.exp(log_prefactor + dimension * torch.log(radius.clamp_min(0)))

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        distances = torch.linalg.vector_norm(points.unsqueeze(2) - self.center.unsqueeze(1), dim=-1)
        return distances <= self.radius.unsqueeze(1)

    @staticmethod
    def from_samples(
        samples: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        coverage: float,
    ):
        nx, _, _ = samples.shape
        k = centroids.shape[1]

        assigned_l2_distances = torch.linalg.vector_norm(
            _assigned_diff(samples=samples, centroids=centroids, assignments=assignments),
            dim=-1,
        )

        radii = torch.zeros((nx, k), dtype=samples.dtype, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                d_ij = assigned_l2_distances[i][mask]
                if d_ij.numel() == 0:
                    continue

                rank = _coverage_rank(d_ij.numel(), coverage)
                radii[i, j] = d_ij.kthvalue(rank).values

        return EuclideanBall(center=centroids, radius=radii)