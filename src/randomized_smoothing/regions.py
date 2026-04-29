from abc import ABC, abstractmethod

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


def _shrink_axis_aligned_boxes_to_disjoint_support(
    lower: torch.Tensor,
    upper: torch.Tensor,
    centroids: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip overlapping boxes along a centroid-separating axis until disjoint."""
    adjusted_lower = lower.clone()
    adjusted_upper = upper.clone()
    eps = torch.finfo(lower.dtype).eps

    nx, k, _ = lower.shape
    for i in range(nx):
        active_indices = torch.where(active[i])[0]
        for left_idx in range(active_indices.numel()):
            j = int(active_indices[left_idx].item())
            for right_idx in range(left_idx + 1, active_indices.numel()):
                l = int(active_indices[right_idx].item())

                overlap_lower = torch.maximum(adjusted_lower[i, j], adjusted_lower[i, l])
                overlap_upper = torch.minimum(adjusted_upper[i, j], adjusted_upper[i, l])
                if not torch.all(overlap_lower <= overlap_upper):
                    continue

                delta = (centroids[i, l] - centroids[i, j]).abs()
                split_dim = int(delta.argmax().item())
                midpoint = 0.5 * (centroids[i, j, split_dim] + centroids[i, l, split_dim])
                gap = eps * (1.0 + midpoint.abs())

                if centroids[i, j, split_dim] <= centroids[i, l, split_dim]:
                    adjusted_upper[i, j, split_dim] = torch.minimum(
                        adjusted_upper[i, j, split_dim], midpoint - gap
                    )
                    adjusted_lower[i, l, split_dim] = torch.maximum(
                        adjusted_lower[i, l, split_dim], midpoint + gap
                    )
                else:
                    adjusted_upper[i, l, split_dim] = torch.minimum(
                        adjusted_upper[i, l, split_dim], midpoint - gap
                    )
                    adjusted_lower[i, j, split_dim] = torch.maximum(
                        adjusted_lower[i, j, split_dim], midpoint + gap
                    )

    return adjusted_lower, adjusted_upper

class HyperRectangle(ABC):
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

    def distance(self, points: torch.Tensor) -> torch.Tensor:
        lower_gap = (self.lower.unsqueeze(1) - points.unsqueeze(2)).clamp_min(0)
        upper_gap = (points.unsqueeze(2) - self.upper.unsqueeze(1)).clamp_min(0)
        return torch.linalg.vector_norm(lower_gap + upper_gap, dim=-1)

    def __len__(self):
        return self.lower.size(0)

    def size(self, dim=None):
        if dim is None:
            return self.lower.size()

        return self.lower.size(dim)

    @classmethod
    def from_eps(cls, x, eps):
        lower, upper = x - eps, x + eps
        return cls(lower, upper)

    @staticmethod
    @abstractmethod
    def from_samples(
        samples: torch.Tensor,
        centroids: torch.Tensor,
        assignments: torch.Tensor,
        coverage: float,
    ):
        raise NotImplementedError



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
        active = torch.zeros((nx, k), dtype=torch.bool, device=samples.device)
        for i in range(nx):
            for j in range(k):
                mask = assignments[i] == j
                if not mask.any():
                    continue

                active[i, j] = True

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

        lower, upper = _shrink_axis_aligned_boxes_to_disjoint_support(
            lower=lower,
            upper=upper,
            centroids=centroids,
            active=active,
        )

        return AxisAlignedHyperRectangle(lower=lower, upper=upper)