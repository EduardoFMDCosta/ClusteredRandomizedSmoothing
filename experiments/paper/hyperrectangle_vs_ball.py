from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters


def make_samples() -> torch.Tensor:
    torch.manual_seed(7)

    centers = torch.tensor(
        [
            [-2.5, -0.4],
            [0.8, 2.8],
            [2.7, -1.2],
        ],
        dtype=torch.float64,
    )
    scales = torch.tensor(
        [
            [0.45, 0.85],
            [0.55, 0.40],
            [0.75, 0.60],
        ],
        dtype=torch.float64,
    )
    counts = [240, 170, 210]

    draws = []
    for center, scale, count in zip(centers, scales, counts):
        draws.append(center + scale * torch.randn(count, 2, dtype=torch.float64))

    outputs = torch.cat(draws, dim=0)
    permutation = torch.randperm(outputs.shape[0])
    outputs = outputs[permutation]
    return outputs.unsqueeze(0)


def compute_partition_stats(cluster: Cluster, outputs: torch.Tensor) -> list[dict[str, float | int]]:
    assigned_mask = torch.nn.functional.one_hot(
        cluster.assignments.long(),
        num_classes=cluster.num_clusters,
    ).to(dtype=torch.bool)
    in_region = cluster.regions.contains(outputs)

    stats: list[dict[str, float | int]] = []
    active_k = int(cluster.k_per_row[0].item())
    for j in range(active_k):
        assigned = assigned_mask[0, :, j]
        num_assigned = int(assigned.sum().item())
        if num_assigned == 0:
            achieved = 0.0
        else:
            achieved = float((in_region[0, :, j] & assigned).sum().item() / num_assigned)

        stats.append(
            {
                "cluster_index": j,
                "num_assigned": num_assigned,
                "target_coverage": float(cluster.parameters.coverage),
                "achieved_coverage": achieved,
            }
        )

    return stats


def _add_region_patch(ax: plt.Axes, cluster: Cluster, cluster_index: int, color: str) -> None:
    region = cluster.regions
    centroid = cluster.centroids[0, cluster_index].detach().cpu().numpy()

    if hasattr(region, "radius"):
        radius = float(region.radius[0, cluster_index].detach().cpu().item())
        patch = mpatches.Circle(
            centroid,
            radius=radius,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle="-",
        )
    else:
        lower = region.lower[0, cluster_index].detach().cpu().numpy()
        upper = region.upper[0, cluster_index].detach().cpu().numpy()
        patch = mpatches.Rectangle(
            lower,
            *(upper - lower),
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle="-",
        )

    ax.add_patch(patch)


def plot_comparison(
    outputs: torch.Tensor,
    clusters_by_key: dict[tuple[str, float], Cluster],
    coverages: list[float],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_np = outputs[0].detach().cpu().numpy()
    x_pad = 0.8
    y_pad = 0.8

    all_x = samples_np[:, 0]
    all_y = samples_np[:, 1]
    x_lim = (float(all_x.min() - x_pad), float(all_x.max() + x_pad))
    y_lim = (float(all_y.min() - y_pad), float(all_y.max() + y_pad))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    paths: list[Path] = []

    for coverage in coverages:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(7, 3.2),
            sharex=True,
            sharey=True,
        )

        for col, set_type in enumerate(["HyperRectangle", "EuclideanBall"]):
            ax = axes[col]
            cluster = clusters_by_key[(set_type, coverage)]
            assignments = cluster.assignments[0].detach().cpu().numpy()
            active_k = int(cluster.k_per_row[0].item())

            for j in range(active_k):
                cluster_points = samples_np[assignments == j]
                ax.scatter(
                    cluster_points[:, 0],
                    cluster_points[:, 1],
                    s=12,
                    alpha=0.45,
                    color=colors[j % len(colors)],
                )
                centroid = cluster.centroids[0, j].detach().cpu().numpy()
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    marker="x",
                    s=90,
                    linewidths=2.2,
                    color=colors[j % len(colors)],
                )
                _add_region_patch(ax=ax, cluster=cluster, cluster_index=j, color=colors[j % len(colors)])

            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25, linestyle=":")

        fig.subplots_adjust(wspace=0.05)
        output_path = output_dir / f"hyperrectangle_vs_ball_cov{coverage:.2f}.pdf"
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)

    return paths


def main() -> None:
    outputs = make_samples()
    coverages = [0.60, 0.90]
    set_types = ["HyperRectangle", "EuclideanBall"]

    clusters_by_key: dict[tuple[str, float], Cluster] = {}
    summary: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for set_type in set_types:
        summary[set_type] = {}
        for coverage in coverages:
            cluster = Cluster(
                outputs=outputs,
                parameters=ClusteringParameters(
                    max_num_clusters=3,
                    coverage=coverage,
                    set_type=set_type,
                ),
            )
            clusters_by_key[(set_type, coverage)] = cluster
            summary[set_type][f"{coverage:.2f}"] = compute_partition_stats(cluster, outputs)

    chart_paths = plot_comparison(
        outputs=outputs,
        clusters_by_key=clusters_by_key,
        coverages=coverages,
        output_dir=EXPERIMENTS / "data" / "figures",
    )

    print("=== Hyperrectangle vs Ball ===")
    for set_type in set_types:
        print(f"set_type={set_type}")
        for coverage in coverages:
            stats = summary[set_type][f"{coverage:.2f}"]
            achieved = ", ".join(
                f"cluster {stat['cluster_index']}: target={stat['target_coverage']:.2f}, achieved={stat['achieved_coverage']:.2f}, n={stat['num_assigned']}"
                for stat in stats
            )
            print(f"  coverage={coverage:.2f} -> {achieved}")

    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()