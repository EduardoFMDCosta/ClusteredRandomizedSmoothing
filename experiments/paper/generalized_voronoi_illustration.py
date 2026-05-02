"""Conceptual illustration of the L2 generalized--Voronoi partition for a set
of disjoint rectangular sites.

For a closed set ``R`` and a query point ``p``, the (point-to-set) distance is

    d(p, R) = min_{r in R} ||p - r||_2.

Given disjoint sites ``R_1, ..., R_K``, the generalized--Voronoi cell of ``R_i``
is the locus of points closer to ``R_i`` than to any other site,

    V_i = { p : d(p, R_i) <= d(p, R_j) for all j != i }.

For a rectangle the nearest point is obtained by per-axis clipping of the
query, so the plane is split by each rectangle into 9 regions (3x3 strips of
"corner cones" and "edge slabs" plus the interior).  In any sub-region defined
by intersecting the strips of two rectangles, both distance functions are
elementary and the bisector is exactly:

    * a straight line when both rectangles contribute by the same kind of
      anchor (corner-vs-corner, edge-vs-edge, edge-vs-perpendicular-edge);
    * a parabolic arc when one rectangle contributes a corner and the other
      contributes an edge.

We render each pair's bisector pieces analytically, drop any sub-segment where
a third rectangle is strictly closer, and overlay the result on a colored
nearest-cell map.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgba

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plotting import (  # noqa: E402  -- applies shared rcParams
    CLUSTER_PALETTE,
    TEXTWIDTH,
)


# ----------------------------------------------------------------------------
# Distance helpers.
# ----------------------------------------------------------------------------

def _nearest_point(point: np.ndarray, rectangle: np.ndarray) -> np.ndarray:
    xmin, ymin, xmax, ymax = rectangle
    return np.array([np.clip(point[0], xmin, xmax), np.clip(point[1], ymin, ymax)])


def _min_distances(points: np.ndarray, rectangles: np.ndarray) -> np.ndarray:
    """For each ``(point, rectangle)`` pair, return the L2 distance to the
    nearest point of the rectangle.

    Args:
        points: ``(P, 2)`` query points.
        rectangles: ``(K, 4)`` packed as ``(xmin, ymin, xmax, ymax)``.

    Returns:
        Array of shape ``(P, K)``.
    """
    px = points[:, 0:1]
    py = points[:, 1:2]
    xmin = rectangles[None, :, 0]
    ymin = rectangles[None, :, 1]
    xmax = rectangles[None, :, 2]
    ymax = rectangles[None, :, 3]
    cx = np.clip(px, xmin, xmax)
    cy = np.clip(py, ymin, ymax)
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


# ----------------------------------------------------------------------------
# Analytical bisector pieces.
# ----------------------------------------------------------------------------

def _x_band(rect: np.ndarray, sx: int) -> tuple[float, float]:
    if sx == -1:
        return (-np.inf, rect[0])
    if sx == 0:
        return (rect[0], rect[2])
    return (rect[2], np.inf)


def _y_band(rect: np.ndarray, sy: int) -> tuple[float, float]:
    if sy == -1:
        return (-np.inf, rect[1])
    if sy == 0:
        return (rect[1], rect[3])
    return (rect[3], np.inf)


def _clip_line_to_box(
    A: float, B: float, C: float,
    xlo: float, xhi: float, ylo: float, yhi: float,
) -> list[np.ndarray]:
    """Clip the implicit line ``A px + B py = C`` to ``[xlo, xhi] x [ylo, yhi]``.

    Returns either ``[]`` or ``[polyline]`` where ``polyline`` is ``(2, 2)``.
    """
    eps = 1e-9
    pts: list[tuple[float, float]] = []
    if abs(B) > 1e-12:
        for x_edge in (xlo, xhi):
            py = (C - A * x_edge) / B
            if ylo - eps <= py <= yhi + eps:
                pts.append((x_edge, py))
    if abs(A) > 1e-12:
        for y_edge in (ylo, yhi):
            px = (C - B * y_edge) / A
            if xlo - eps <= px <= xhi + eps:
                pts.append((px, y_edge))

    if len(pts) < 2:
        return []

    arr = np.array(pts)
    if len(arr) == 2:
        if np.allclose(arr[0], arr[1]):
            return []
        return [arr]
    # Degenerate case: line clips to >2 box edges (passes through corners).
    # Pick the pair with maximum separation.
    d2 = np.sum((arr[:, None] - arr[None, :]) ** 2, axis=-1)
    i_max, j_max = np.unravel_index(d2.argmax(), d2.shape)
    if d2[i_max, j_max] < 1e-18:
        return []
    return [np.array([arr[i_max], arr[j_max]])]


def _polylines_from_mask(
    px_grid: np.ndarray, py_grid: np.ndarray, mask: np.ndarray,
) -> list[np.ndarray]:
    if not mask.any():
        return []
    runs: list[np.ndarray] = []
    in_run = False
    start = 0
    for k in range(len(mask)):
        if mask[k] and not in_run:
            start = k
            in_run = True
        elif not mask[k] and in_run:
            if k - start >= 2:
                runs.append(np.column_stack([px_grid[start:k], py_grid[start:k]]))
            in_run = False
    if in_run and len(mask) - start >= 2:
        runs.append(np.column_stack([px_grid[start:], py_grid[start:]]))
    return runs


def _bisector_in_subregion(
    ri: np.ndarray, rj: np.ndarray,
    sxi: int, syi: int, sxj: int, syj: int,
    xlo: float, xhi: float, ylo: float, yhi: float,
    n_samples: int,
) -> list[np.ndarray]:
    """Closed-form bisector ``{p : d(p, ri) = d(p, rj)}`` clipped to the
    sub-region box, where each rectangle's strip indices ``(sxi, syi)`` /
    ``(sxj, syj)`` are in ``{-1, 0, +1}`` (with ``0`` meaning the central
    strip / interior).
    """
    # Anchor coordinates pulled from each rectangle's relevant edge.  ``NaN``
    # indicates the coordinate is the free variable in that strip; whether a
    # coordinate is free is decided up-front by the integer ``sx_*``/``sy_*``
    # signs, so we never read a NaN-anchor without first checking its sign.
    xa = ri[0] if sxi == -1 else (ri[2] if sxi == +1 else float("nan"))
    ya = ri[1] if syi == -1 else (ri[3] if syi == +1 else float("nan"))
    xb = rj[0] if sxj == -1 else (rj[2] if sxj == +1 else float("nan"))
    yb = rj[1] if syj == -1 else (rj[3] if syj == +1 else float("nan"))

    i_corner = sxi != 0 and syi != 0
    i_x_edge = sxi != 0 and syi == 0
    i_y_edge = sxi == 0 and syi != 0
    j_corner = sxj != 0 and syj != 0
    j_x_edge = sxj != 0 and syj == 0
    j_y_edge = sxj == 0 and syj != 0

    # Corner-corner: perpendicular bisector of two points (a line).
    if i_corner and j_corner:
        return _clip_line_to_box(
            2.0 * (xb - xa),
            2.0 * (yb - ya),
            (xb ** 2 + yb ** 2) - (xa ** 2 + ya ** 2),
            xlo, xhi, ylo, yhi,
        )

    # Two parallel x-edges: vertical line at midpoint (only when slabs face
    # each other, i.e. ``sxi != sxj``; otherwise the closer slab dominates).
    if i_x_edge and j_x_edge:
        if sxi == sxj:
            return []
        px_star = (sxi * xa - sxj * xb) / (sxi - sxj)
        if not (xlo <= px_star <= xhi):
            return []
        return [np.array([[px_star, ylo], [px_star, yhi]])]

    if i_y_edge and j_y_edge:
        if syi == syj:
            return []
        py_star = (syi * ya - syj * yb) / (syi - syj)
        if not (ylo <= py_star <= yhi):
            return []
        return [np.array([[xlo, py_star], [xhi, py_star]])]

    # Perpendicular edges: signed-equality picks the geometrically valid
    # branch of ``(px - xa)^2 = (py - yb)^2``.
    if i_x_edge and j_y_edge:
        return _clip_line_to_box(
            sxi, -syj,
            sxi * xa - syj * yb,
            xlo, xhi, ylo, yhi,
        )
    if i_y_edge and j_x_edge:
        return _clip_line_to_box(
            sxj, -syi,
            sxj * xb - syi * ya,
            xlo, xhi, ylo, yhi,
        )

    # Edge-vs-corner: parabolic arc.
    if i_x_edge and j_corner:
        # (px - xa)^2 = (px - xb)^2 + (py - yb)^2  =>
        #   2 (xb - xa) px = xb^2 - xa^2 + (py - yb)^2.
        if abs(xb - xa) < 1e-12:
            if ylo <= yb <= yhi:
                return [np.array([[xlo, yb], [xhi, yb]])]
            return []
        py_grid = np.linspace(ylo, yhi, n_samples)
        px_grid = (xb ** 2 - xa ** 2 + (py_grid - yb) ** 2) / (2.0 * (xb - xa))
        mask = (px_grid >= xlo - 1e-9) & (px_grid <= xhi + 1e-9)
        return _polylines_from_mask(px_grid, py_grid, mask)

    if i_corner and j_x_edge:
        if abs(xa - xb) < 1e-12:
            if ylo <= ya <= yhi:
                return [np.array([[xlo, ya], [xhi, ya]])]
            return []
        py_grid = np.linspace(ylo, yhi, n_samples)
        px_grid = (xa ** 2 - xb ** 2 + (py_grid - ya) ** 2) / (2.0 * (xa - xb))
        mask = (px_grid >= xlo - 1e-9) & (px_grid <= xhi + 1e-9)
        return _polylines_from_mask(px_grid, py_grid, mask)

    if i_y_edge and j_corner:
        if abs(yb - ya) < 1e-12:
            if xlo <= xb <= xhi:
                return [np.array([[xb, ylo], [xb, yhi]])]
            return []
        px_grid = np.linspace(xlo, xhi, n_samples)
        py_grid = ((px_grid - xb) ** 2 + yb ** 2 - ya ** 2) / (2.0 * (yb - ya))
        mask = (py_grid >= ylo - 1e-9) & (py_grid <= yhi + 1e-9)
        return _polylines_from_mask(px_grid, py_grid, mask)

    if i_corner and j_y_edge:
        if abs(ya - yb) < 1e-12:
            if xlo <= xa <= xhi:
                return [np.array([[xa, ylo], [xa, yhi]])]
            return []
        px_grid = np.linspace(xlo, xhi, n_samples)
        py_grid = ((px_grid - xa) ** 2 + ya ** 2 - yb ** 2) / (2.0 * (ya - yb))
        mask = (py_grid >= ylo - 1e-9) & (py_grid <= yhi + 1e-9)
        return _polylines_from_mask(px_grid, py_grid, mask)

    return []


def _filter_by_other_rectangles(
    polyline: np.ndarray, rectangles: np.ndarray, i: int, j: int,
    eps: float = 1e-7,
) -> list[np.ndarray]:
    """Drop sub-segments where some R_k (k != i, j) is strictly closer than
    R_i and R_j (which by construction are tied along the polyline)."""
    K = rectangles.shape[0]
    if K <= 2:
        return [polyline] if len(polyline) >= 2 else []

    d_ij = _min_distances(polyline, rectangles[[i]])[:, 0]
    valid = np.ones(len(polyline), dtype=bool)
    for k in range(K):
        if k == i or k == j:
            continue
        d_k = _min_distances(polyline, rectangles[[k]])[:, 0]
        valid &= d_k >= d_ij - eps

    runs: list[np.ndarray] = []
    in_run = False
    start = 0
    for idx in range(len(valid)):
        if valid[idx] and not in_run:
            start = idx
            in_run = True
        elif not valid[idx] and in_run:
            if idx - start >= 2:
                runs.append(polyline[start:idx])
            in_run = False
    if in_run and len(valid) - start >= 2:
        runs.append(polyline[start:])
    return runs


def _voronoi_boundary_segments(
    rectangles: np.ndarray, viewport: tuple[float, float, float, float],
    n_samples: int = 250,
) -> list[np.ndarray]:
    """Analytical generalized-Voronoi boundary segments inside ``viewport``."""
    xv_lo, xv_hi, yv_lo, yv_hi = viewport
    K = rectangles.shape[0]
    segments: list[np.ndarray] = []

    for i in range(K):
        for j in range(i + 1, K):
            ri = rectangles[i]
            rj = rectangles[j]
            for sxi in (-1, 0, 1):
                for syi in (-1, 0, 1):
                    if sxi == 0 and syi == 0:
                        continue  # interior of R_i: d_i = 0
                    xb_i = _x_band(ri, sxi)
                    yb_i = _y_band(ri, syi)
                    for sxj in (-1, 0, 1):
                        for syj in (-1, 0, 1):
                            if sxj == 0 and syj == 0:
                                continue
                            xb_j = _x_band(rj, sxj)
                            yb_j = _y_band(rj, syj)

                            xlo = max(xb_i[0], xb_j[0], xv_lo)
                            xhi = min(xb_i[1], xb_j[1], xv_hi)
                            ylo = max(yb_i[0], yb_j[0], yv_lo)
                            yhi = min(yb_i[1], yb_j[1], yv_hi)
                            if xlo >= xhi or ylo >= yhi:
                                continue

                            polylines = _bisector_in_subregion(
                                ri, rj, sxi, syi, sxj, syj,
                                xlo, xhi, ylo, yhi, n_samples,
                            )
                            for poly in polylines:
                                segments.extend(
                                    _filter_by_other_rectangles(poly, rectangles, i, j)
                                )
    return segments


# ----------------------------------------------------------------------------
# Plotting helpers.
# ----------------------------------------------------------------------------

def _draw_rectangles(ax: plt.Axes, rectangles: np.ndarray) -> None:
    for i, (xmin, ymin, xmax, ymax) in enumerate(rectangles):
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        ax.add_patch(
            mpatches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor=to_rgba(color, alpha=0.85),
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )
        )
        ax.text(
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            rf"$R_{{{i + 1}}}$",
            ha="center", va="center",
            color="white", fontsize=9, fontweight="bold",
            zorder=5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=str(EXPERIMENTS / "data" / "figures"))
    parser.add_argument("--grid", type=int, default=1200,
                        help="Resolution of the colored cell-map background.")
    parser.add_argument("--bisector-samples", type=int, default=250,
                        help="Sample count per parabolic arc segment.")
    args = parser.parse_args()

    rectangles = np.array([
        [-4.0, 1.2, -1.8, 3.4],
        [0.6, 1.5, 3.6, 3.6],
        [-2.4, -3.4, 0.4, -1.0],
        [2.0, -3.6, 4.2, -0.6],
    ])
    K = rectangles.shape[0]

    xlim = (-6.0, 6.0)
    ylim = (-6.0, 6.0)

    # Cell-map raster (used purely for the colored background).
    xs = np.linspace(*xlim, args.grid)
    ys = np.linspace(*ylim, args.grid)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    labels = _min_distances(grid_points, rectangles).argmin(-1).reshape(X.shape)

    # Analytical boundaries.
    boundary_segments = _voronoi_boundary_segments(
        rectangles, viewport=(*xlim, *ylim), n_samples=args.bisector_samples,
    )

    _w = TEXTWIDTH
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(_w, 0.52 * _w), constrained_layout=True,
    )

    # --- Panel (a): point-to-set distance from a query point. ---
    query_point = np.array([0.4, -0.1])
    rect_dists = _min_distances(query_point[None], rectangles)[0]
    witness_idx = int(rect_dists.argmin())

    _draw_rectangles(ax_a, rectangles)
    for i in range(K):
        target = _nearest_point(query_point, rectangles[i])
        is_witness = (i == witness_idx)
        color = "#222222" if is_witness else "#7A7A7A"
        lw = 1.3 if is_witness else 0.7
        ls = "-" if is_witness else (0, (3, 3))
        ax_a.annotate(
            "", xy=target, xytext=query_point,
            arrowprops=dict(
                arrowstyle="->", color=color,
                linewidth=lw, linestyle=ls, shrinkA=2.5, shrinkB=0.0,
            ),
            zorder=5,
        )
        ax_a.scatter(
            *target, color=CLUSTER_PALETTE[i], s=18,
            edgecolor="white", linewidth=0.5, zorder=6,
        )
    ax_a.scatter(*query_point, color="black", s=24, zorder=7)
    ax_a.text(
        query_point[0] + 0.25, query_point[1] - 0.2, r"$p$",
        fontsize=10, ha="left", va="top", zorder=7,
    )
    witness_target = _nearest_point(query_point, rectangles[witness_idx])
    midpoint = 0.5 * (query_point + witness_target)
    ax_a.text(
        midpoint[0] + 0.2, midpoint[1] + 0.1,
        rf"$d(p, R_{{{witness_idx + 1}}})$",
        fontsize=8, color="#222222", ha="left", va="bottom", zorder=7,
    )
    ax_a.set_xlim(*xlim)
    ax_a.set_ylim(*ylim)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel(r"$x_1$")
    ax_a.set_ylabel(r"$x_2$")
    ax_a.set_title(
        r"(a) $d(p, R) = \min_{r \in R} \|p - r\|_2$", fontsize=9,
    )

    # --- Panel (b): generalized--Voronoi partition. ---
    palette = [
        to_rgba(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)], alpha=0.30) for i in range(K)
    ]
    cmap = ListedColormap(palette)
    ax_b.imshow(
        labels,
        origin="lower",
        extent=[*xlim, *ylim],
        cmap=cmap,
        interpolation="nearest",
        zorder=0,
        aspect="equal",
    )
    # Densify every polyline to >= 16 vertices: some PDF readers drop
    # perfectly axis-aligned 2-point paths during hairline optimization, so
    # an identical multi-vertex representation makes all segments render.
    densified: list[np.ndarray] = []
    for seg in boundary_segments:
        if len(seg) < 2:
            continue
        if len(seg) >= 16:
            densified.append(seg)
            continue
        t = np.linspace(0.0, 1.0, 16)
        # Linear interpolation along the existing polyline.
        cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))))
        if cum[-1] < 1e-12:
            continue
        s = t * cum[-1]
        x_new = np.interp(s, cum, seg[:, 0])
        y_new = np.interp(s, cum, seg[:, 1])
        densified.append(np.column_stack([x_new, y_new]))
    boundary_collection = LineCollection(
        densified,
        colors="#101010", linewidths=1.1, capstyle="round",
        zorder=2,
    )
    ax_b.add_collection(boundary_collection)
    _draw_rectangles(ax_b, rectangles)
    ax_b.set_xlim(*xlim)
    ax_b.set_ylim(*ylim)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel(r"$x_1$")
    ax_b.tick_params(axis="y", labelleft=False)
    ax_b.set_title(
        r"(b) Generalized--Voronoi partition $\{V_i\}_{i=1}^{K}$", fontsize=9,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_pdf = out_dir / "generalized_voronoi_illustration.pdf"
    fig.savefig(out_path_pdf)
    plt.close(fig)
    print(f"Saved {out_path_pdf}")


if __name__ == "__main__":
    main()
