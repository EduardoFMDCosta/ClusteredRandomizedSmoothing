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


def _project_half_widths_to_voronoi_cell(
    half_widths: torch.Tensor,
    centroids_row: torch.Tensor,
    cluster_index: int,
) -> torch.Tensor:
    """Scale axis-aligned half-widths so the centered box stays in the Voronoi cell."""
    k = centroids_row.shape[0]
    if k <= 1:
        return half_widths

    c_j = centroids_row[cluster_index]
    scale = torch.tensor(1.0, dtype=half_widths.dtype, device=half_widths.device)
    eps = torch.finfo(half_widths.dtype).eps

    for l in range(k):
        if l == cluster_index:
            continue
        delta = centroids_row[l] - c_j
        lhs = (delta.abs() * half_widths).sum()
        if lhs <= eps:
            continue

        rhs = 0.5 * (delta * delta).sum()
        candidate = (rhs / lhs).clamp(min=0.0, max=1.0)
        scale = torch.minimum(scale, candidate)

    return half_widths * scale

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

        half_widths = torch.zeros((nx, k, output_dim), dtype=samples.dtype, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                diff_ij = assigned_abs_diff[i][mask]
                if diff_ij.numel() == 0:
                    continue

                rank = _coverage_rank(diff_ij.shape[0], coverage)
                reference_half_widths = diff_ij.kthvalue(rank, dim=0).values
                half_widths[i, j] = _project_half_widths_to_voronoi_cell(
                    half_widths=reference_half_widths,
                    centroids_row=centroids[i],
                    cluster_index=j,
                )

        lower = centroids - half_widths
        upper = centroids + half_widths
        return HyperRectangle(lower, upper)



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


class DisjointRegion:
    """Wrap a region so membership is disjoint across clusters.

    If a point belongs to multiple cluster regions, it is assigned to exactly
    one region using nearest-centroid tie-breaking.
    """

    def __init__(self, region, centroids: torch.Tensor):
        self.region = region
        self.centroids = centroids

    def __getattr__(self, name):
        return getattr(self.region, name)

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        in_region = self.region.contains(points)
        if in_region.numel() == 0:
            return in_region

        num_clusters = self.centroids.shape[1]
        squared_distances = ((points.unsqueeze(2) - self.centroids.unsqueeze(1)) ** 2).sum(dim=-1)
        nearest = squared_distances.argmin(dim=-1)
        nearest_one_hot = torch.nn.functional.one_hot(nearest, num_classes=num_clusters).to(torch.bool)

        return in_region & nearest_one_hot