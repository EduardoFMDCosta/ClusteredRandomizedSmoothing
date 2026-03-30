from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make src/ importable when running `python bounds_test_and_plot.py`.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing import alpha_trimming_bound, clusterized_alpha_trimming_bound  # noqa: E402


def _parse_float_list(value: str) -> list[float]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected at least one numeric value")
    return [float(x) for x in items]


def _build_tensor(values: list[float], name: str) -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float64)
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    return t


def _weighted_global_bound(pi: torch.Tensor, p: torch.Tensor, alpha: float, n: int, p_check: torch.Tensor) -> float:
    per_region = torch.tensor(
        [
            alpha_trimming_bound(float(p[m].item()), alpha=alpha, n=n, p_check=float(p_check[m].item()))
            for m in range(p.shape[0])
        ],
        dtype=torch.float64,
    )
    return float(torch.sum(pi * per_region).item())


def compute_bound_grids(
    pi: torch.Tensor,
    p: torch.Tensor,
    p_check: torch.Tensor,
    n_values: list[int],
    alpha_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute two bound grids with shape (len(alpha_values), len(n_values))."""
    grid_global = np.zeros((len(alpha_values), len(n_values)), dtype=np.float64)
    grid_cluster = np.zeros((len(alpha_values), len(n_values)), dtype=np.float64)

    for i, alpha in enumerate(alpha_values):
        for j, n in enumerate(n_values):
            grid_global[i, j] = _weighted_global_bound(pi=pi, p=p, alpha=alpha, n=n, p_check=p_check)
            grid_cluster[i, j] = float(
                clusterized_alpha_trimming_bound(
                    pi=pi,
                    p_r=p,
                    p_v=p_check,
                    alpha=alpha,
                    n=n,
                    validate_pi_sum=True,
                ).item()
            )

    return grid_global, grid_cluster


def run_sanity_tests(grid_global: np.ndarray, grid_cluster: np.ndarray, alpha_values: list[float]) -> None:
    """Basic checks for numerical and structural consistency."""
    eps = 1e-10
    if np.any(grid_global < -eps) or np.any(grid_global > 1.0 + eps):
        raise AssertionError("Global bound is outside [0, 1]")
    if np.any(grid_cluster < -eps) or np.any(grid_cluster > 1.0 + eps):
        raise AssertionError("Clusterized bound is outside [0, 1]")

    # As alpha increases, floor(alpha * k) does not decrease, so both bounds
    # should be non-decreasing in alpha for fixed N.
    if len(alpha_values) > 1:
        if np.any(np.diff(grid_global, axis=0) < -1e-10):
            raise AssertionError("Global bound is not non-decreasing in alpha")
        if np.any(np.diff(grid_cluster, axis=0) < -1e-10):
            raise AssertionError("Clusterized bound is not non-decreasing in alpha")


def _plot_vs_n(
    n_values: list[int],
    alpha_values: list[float],
    grid_global: np.ndarray,
    grid_cluster: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, alpha in enumerate(alpha_values):
        axes[0].plot(n_values, grid_global[i], marker="o", linewidth=1.8, label=f"alpha={alpha:.2f}")
        axes[1].plot(n_values, grid_cluster[i], marker="o", linewidth=1.8, label=f"alpha={alpha:.2f}")

    axes[0].set_title("Bound 1 (weighted over regions)")
    axes[1].set_title("Bound 2 (clusterized)")
    for ax in axes:
        ax.set_xlabel("N")
        ax.set_ylabel("Bound value")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)

    fig.suptitle("Bounds vs N for multiple alpha values", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_vs_alpha(
    alpha_values: list[float],
    n_values: list[int],
    grid_global: np.ndarray,
    grid_cluster: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for j, n in enumerate(n_values):
        axes[0].plot(alpha_values, grid_global[:, j], marker="o", linewidth=1.8, label=f"N={n}")
        axes[1].plot(alpha_values, grid_cluster[:, j], marker="o", linewidth=1.8, label=f"N={n}")

    axes[0].set_title("Bound 1 (weighted over regions)")
    axes[1].set_title("Bound 2 (clusterized)")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.set_ylabel("Bound value")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)

    fig.suptitle("Bounds vs alpha for multiple N values", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and plot bound comparisons over N and alpha")
    parser.add_argument(
        "--p",
        type=str,
        default="0.6,0.6,0.6",
        help="Comma-separated p per region (tensor values)",
    )
    parser.add_argument(
        "--p-check",
        type=str,
        default="0.8,0.8,0.8",
        help="Comma-separated checked-p per region (tensor values)",
    )
    parser.add_argument(
        "--pi",
        type=str,
        default="0.333333,0.333333,0.333334",
        help="Comma-separated region weights pi",
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="10,20,40,80",
        help="Comma-separated N grid",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.05,0.1,0.2,0.3,0.4",
        help="Comma-separated alpha grid",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where plots are saved",
    )

    args = parser.parse_args()

    p = _build_tensor(_parse_float_list(args.p), name="p")
    p_check = _build_tensor(_parse_float_list(args.p_check), name="p_check")
    pi = _build_tensor(_parse_float_list(args.pi), name="pi")

    if not (p.shape == p_check.shape == pi.shape):
        raise ValueError("p, p_check and pi must all have the same length")
    if torch.any(p >= p_check):
        raise ValueError("Expected p < p_check element-wise")

    pi = pi / pi.sum()

    n_values = [int(x) for x in _parse_float_list(args.n_values)]
    alpha_values = [float(x) for x in _parse_float_list(args.alpha_values)]

    if any(n <= 0 for n in n_values):
        raise ValueError("All N values must be > 0")
    if any(not (0.0 <= alpha < 1.0) for alpha in alpha_values):
        raise ValueError("All alpha values must satisfy 0 <= alpha < 1")

    n_values = sorted(set(n_values))
    alpha_values = sorted(set(alpha_values))

    grid_global, grid_cluster = compute_bound_grids(
        pi=pi,
        p=p,
        p_check=p_check,
        n_values=n_values,
        alpha_values=alpha_values,
    )

    run_sanity_tests(grid_global=grid_global, grid_cluster=grid_cluster, alpha_values=alpha_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_vs_n = args.output_dir / "bounds_comparison_vs_n.png"
    out_vs_alpha = args.output_dir / "bounds_comparison_vs_alpha.png"

    _plot_vs_n(
        n_values=n_values,
        alpha_values=alpha_values,
        grid_global=grid_global,
        grid_cluster=grid_cluster,
        output_path=out_vs_n,
    )
    _plot_vs_alpha(
        alpha_values=alpha_values,
        n_values=n_values,
        grid_global=grid_global,
        grid_cluster=grid_cluster,
        output_path=out_vs_alpha,
    )

    print("=== Bound Comparison Test ===")
    print(f"regions M = {p.shape[0]}")
    print(f"p       = {p.tolist()}")
    print(f"p_check = {p_check.tolist()}")
    print(f"pi      = {pi.tolist()}")
    print(f"N grid      = {n_values}")
    print(f"alpha grid  = {alpha_values}")
    print(f"global grid shape     = {grid_global.shape}")
    print(f"clusterized grid shape = {grid_cluster.shape}")
    print(f"saved: {out_vs_n}")
    print(f"saved: {out_vs_alpha}")


if __name__ == "__main__":
    main()
