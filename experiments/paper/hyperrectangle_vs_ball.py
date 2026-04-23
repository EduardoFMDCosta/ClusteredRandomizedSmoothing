from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from plotting import plot_comparison


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