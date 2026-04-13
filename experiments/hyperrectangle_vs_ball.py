from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_np = outputs[0].detach().cpu().numpy()
    x_pad = 0.8
    y_pad = 0.8

    fig, axes = plt.subplots(
        len(coverages),
        2,
        figsize=(11, 4.2 * len(coverages)),
        sharex=True,
        sharey=True,
    )
    if len(coverages) == 1:
        axes = np.array([axes])

    region_type_labels = {
        "hyperrectangle": "Hyperrectangle",
        "ball": "Ball",
    }
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for row, coverage in enumerate(coverages):
        for col, region_type in enumerate(["hyperrectangle", "ball"]):
            ax = axes[row, col]
            cluster = clusters_by_key[(region_type, coverage)]
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

            achieved = [stat["achieved_coverage"] for stat in compute_partition_stats(cluster, outputs)]
            achieved_text = ", ".join(f"{value:.2f}" for value in achieved)
            ax.set_title(
                f"{region_type_labels[region_type]} | coverage={coverage:.2f}\n"
                f"achieved per cluster: [{achieved_text}]"
            )
            ax.grid(alpha=0.25, linestyle=":")
            ax.set_aspect("equal", adjustable="box")

    all_x = samples_np[:, 0]
    all_y = samples_np[:, 1]
    x_lim = (float(all_x.min() - x_pad), float(all_x.max() + x_pad))
    y_lim = (float(all_y.min() - y_pad), float(all_y.max() + y_pad))
    for ax in axes.reshape(-1):
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    output_path = output_dir / "hyperrectangle_vs_ball.pdf"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    outputs = make_samples()
    coverages = [0.60, 0.80, 0.95]
    region_types = ["hyperrectangle", "ball"]

    clusters_by_key: dict[tuple[str, float], Cluster] = {}
    summary: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for region_type in region_types:
        summary[region_type] = {}
        for coverage in coverages:
            cluster = Cluster(
                outputs=outputs,
                parameters=ClusteringParameters(
                    max_num_clusters=3,
                    coverage=coverage,
                    region_type=region_type,
                ),
            )
            clusters_by_key[(region_type, coverage)] = cluster
            summary[region_type][f"{coverage:.2f}"] = compute_partition_stats(cluster, outputs)

    chart_path = plot_comparison(
        outputs=outputs,
        clusters_by_key=clusters_by_key,
        coverages=coverages,
        output_dir=ROOT / "outputs",
    )

    print("=== Hyperrectangle vs Ball ===")
    for region_type in region_types:
        print(f"region_type={region_type}")
        for coverage in coverages:
            stats = summary[region_type][f"{coverage:.2f}"]
            achieved = ", ".join(
                f"cluster {stat['cluster_index']}: target={stat['target_coverage']:.2f}, achieved={stat['achieved_coverage']:.2f}, n={stat['num_assigned']}"
                for stat in stats
            )
            print(f"  coverage={coverage:.2f} -> {achieved}")

    print("")
    print("Saved chart:")
    print(chart_path)


if __name__ == "__main__":
    main()