from __future__ import annotations

import torch

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import FinalStepHyperrectangle


def _make_outputs() -> torch.Tensor:
    # nx=2, num_samples=12, output_dim=2
    row0_cluster_a = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.2, 0.0],
            [0.0, 0.2],
        ],
        dtype=torch.float64,
    )
    row0_cluster_b = torch.tensor(
        [
            [4.0, 4.0],
            [4.1, 4.0],
            [4.0, 4.1],
            [4.1, 4.1],
            [3.9, 4.0],
            [4.0, 3.9],
        ],
        dtype=torch.float64,
    )
    row1_single_cluster = torch.tensor(
        [
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
            [1.1, 1.1],
            [1.2, 1.0],
            [1.0, 1.2],
            [1.2, 1.1],
            [1.1, 1.2],
            [1.3, 1.1],
            [1.1, 1.3],
            [1.2, 1.2],
            [1.25, 1.15],
        ],
        dtype=torch.float64,
    )

    return torch.stack((torch.cat((row0_cluster_a, row0_cluster_b), dim=0), row1_single_cluster), dim=0)


def test_dbscan_supports_variable_clusters_per_row_with_ghost_padding() -> None:
    outputs = _make_outputs()
    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=3,
            coverage=0.8,
            dbscan_eps=0.35,
            dbscan_min_samples=3,
        ),
    )

    assert cluster.k_per_row.shape == (2,)
    assert int(cluster.k_per_row[0].item()) == 2
    assert int(cluster.k_per_row[1].item()) == 1
    assert cluster.num_clusters == 2

    # Row 1 has one active cluster; remaining slot is a ghost centroid copy.
    assert torch.allclose(cluster.centroids[1, 0], cluster.centroids[1, 1])
    assert torch.all(cluster.assignments >= 0)
    assert torch.all(cluster.assignments < cluster.num_clusters)


def test_dbscan_respects_max_num_clusters_equals_one() -> None:
    outputs = _make_outputs()
    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=1,
            coverage=0.8,
            dbscan_eps=0.01,
            dbscan_min_samples=20,
        ),
    )

    assert cluster.num_clusters == 1
    assert torch.equal(cluster.k_per_row, torch.ones_like(cluster.k_per_row))
    assert torch.all(cluster.assignments == 0)


def test_dbscan_noise_points_are_reassigned_to_valid_clusters() -> None:
    outputs = _make_outputs().clone()
    outputs[0, 0] = torch.tensor([20.0, -20.0], dtype=outputs.dtype)

    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=3,
            coverage=0.8,
            dbscan_eps=0.35,
            dbscan_min_samples=3,
        ),
    )

    # No noise label is kept in final tensorized assignments.
    assert torch.all(cluster.assignments >= 0)
    assert torch.all(cluster.assignments < cluster.num_clusters)


def test_dbscan_caps_clusters_to_max_num_clusters() -> None:
    row = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
        dtype=torch.float64,
    )
    outputs = row.unsqueeze(0)

    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.8,
            dbscan_eps=0.25,
            dbscan_min_samples=3,
        ),
    )

    assert cluster.num_clusters == 2
    assert int(cluster.k_per_row[0].item()) == 2


def test_feature_clustering_final_displacement_separates_stay_vs_move() -> None:
    # One row with two clear behavioral modes in trajectory space:
    # six almost-static trajectories vs six forward-moving trajectories.
    stay = torch.tensor(
        [
            [0.00, 0.00, 0.05, 0.00],
            [0.00, 0.00, 0.04, 0.01],
            [0.00, 0.00, 0.03, -0.01],
            [0.00, 0.00, 0.02, 0.00],
            [0.00, 0.00, 0.06, -0.01],
            [0.00, 0.00, 0.05, 0.02],
        ],
        dtype=torch.float64,
    )
    move = torch.tensor(
        [
            [0.00, 0.00, 3.00, 0.00],
            [0.00, 0.00, 3.10, 0.10],
            [0.00, 0.00, 2.90, -0.10],
            [0.00, 0.00, 3.20, 0.00],
            [0.00, 0.00, 2.80, 0.10],
            [0.00, 0.00, 3.00, -0.10],
        ],
        dtype=torch.float64,
    )
    outputs = torch.cat((stay, move), dim=0).unsqueeze(0)

    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.9,
            dbscan_eps=0.3,
            dbscan_min_samples=3,
            cluster_feature="final_displacement",
        ),
    )

    assert cluster.num_clusters == 2
    probs = cluster.probs_partition[0]
    assert torch.isclose(probs.sum(), torch.tensor(1.0, dtype=probs.dtype))
    assert torch.all(probs > 0.0)


def test_final_step_hyperrectangle_contains_uses_last_xy_only() -> None:
    # Region bounds for one x and one cluster at final timestep.
    region = FinalStepHyperrectangle(
        lower=torch.tensor([[[1.0, 1.0]]], dtype=torch.float64),
        upper=torch.tensor([[[2.0, 2.0]]], dtype=torch.float64),
    )

    # Two samples, each encoded as [x1, y1, x2, y2].
    # Sample 0 ends inside [1,2]x[1,2], sample 1 ends outside.
    points = torch.tensor(
        [[
            [100.0, 100.0, 1.5, 1.5],
            [-100.0, -100.0, 2.5, 1.5],
        ]],
        dtype=torch.float64,
    )

    contains = region.contains(points)
    assert contains.shape == (1, 2, 1)
    assert bool(contains[0, 0, 0].item()) is True
    assert bool(contains[0, 1, 0].item()) is False


def test_final_step_hyperrectangle_x_intervals_are_disjoint() -> None:
    outputs = torch.tensor(
        [[
            [0.0, 0.0, -2.1, 0.0],
            [0.0, 0.0, -1.9, 0.1],
            [0.0, 0.0, -2.0, -0.1],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 1.9, 0.1],
            [0.0, 0.0, 2.1, -0.1],
        ]],
        dtype=torch.float64,
    )

    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.95,
            set_type="FinalStepHyperrectangle",
            cluster_feature="trajectory",
            dbscan_eps=0.4,
            dbscan_min_samples=2,
        ),
    )

    assert int(cluster.k_per_row[0].item()) == 2
    lower_x = cluster.regions.lower[0, :, 0]
    upper_x = cluster.regions.upper[0, :, 0]

    # Intervals are disjoint up to ordering.
    left = torch.argmin(cluster.centroids[0, :, -2])
    right = torch.argmax(cluster.centroids[0, :, -2])
    assert upper_x[left] <= lower_x[right]


def test_adaptive_cluster_coverage_reduces_small_cluster_area() -> None:
    # Large cluster near x=0 and small cluster near x=3.
    large = torch.tensor(
        [[0.0, 0.0, 0.1 * i, 0.0] for i in range(10)],
        dtype=torch.float64,
    )
    small = torch.tensor(
        [[0.0, 0.0, 3.0 + 0.4 * i, 0.0] for i in range(3)],
        dtype=torch.float64,
    )
    outputs = torch.cat((large, small), dim=0).unsqueeze(0)

    base = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.95,
            set_type="FinalStepHyperrectangle",
            cluster_feature="final_displacement",
            adaptive_cluster_coverage=False,
            dbscan_eps=0.35,
            dbscan_min_samples=2,
        ),
    )
    adaptive = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.95,
            set_type="FinalStepHyperrectangle",
            cluster_feature="final_displacement",
            adaptive_cluster_coverage=True,
            min_cluster_coverage=0.5,
            dbscan_eps=0.35,
            dbscan_min_samples=2,
        ),
    )

    # Identify smaller-mass cluster and compare final-step area.
    probs = base.probs_partition[0]
    small_idx = int(torch.argmin(probs).item())
    assert adaptive.regions.area[0, small_idx] <= base.regions.area[0, small_idx]
