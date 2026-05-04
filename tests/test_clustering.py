from __future__ import annotations

import pytest
import torch

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import AxisAlignedHyperRectangle, HyperRectangle


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


def test_hyperrectangle_is_abstract() -> None:
    lower = torch.zeros((1, 1, 2), dtype=torch.float64)
    upper = torch.ones((1, 1, 2), dtype=torch.float64)
    with pytest.raises(TypeError):
        HyperRectangle(lower=lower, upper=upper)


def test_axis_aligned_hyperrectangles_are_shrunk_to_be_disjoint() -> None:
    centroids = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
    assignments = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    samples = torch.tensor(
        [[
            [0.9, 0.0],
            [1.4, 0.2],
            [1.3, -0.1],
            [1.1, 0.0],
            [0.6, -0.2],
            [0.7, 0.1],
        ]],
        dtype=torch.float64,
    )

    region = AxisAlignedHyperRectangle.from_samples(
        samples=samples,
        centroids=centroids,
        assignments=assignments,
        coverage=1.0,
    )

    assert float(region.upper[0, 0, 0].item()) < float(region.lower[0, 1, 0].item())


def test_partition_probabilities_follow_nearest_region_distance() -> None:
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

    expected_assignments = cluster.regions.distance(outputs).argmin(dim=2)
    expected_probs = torch.nn.functional.one_hot(
        expected_assignments,
        num_classes=cluster.num_clusters,
    ).to(dtype=cluster.centroids.dtype).mean(dim=1)

    assert torch.equal(cluster.partition_assignments, expected_assignments)
    assert torch.allclose(cluster.probs_partition, expected_probs)


def test_hdbscan_finds_two_clusters_on_bimodal_data() -> None:
    outputs = _make_outputs()
    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=3,
            coverage=0.8,
            strategy="hdbscan",
            hdbscan_min_cluster_size=3,
        ),
    )

    assert cluster.k_per_row.shape == (2,)
    assert int(cluster.k_per_row[0].item()) == 2
    assert int(cluster.k_per_row[1].item()) == 1
    assert cluster.num_clusters == 2
    assert torch.all(cluster.assignments >= 0)
    assert torch.all(cluster.assignments < cluster.num_clusters)


def test_hdbscan_caps_clusters_to_max_num_clusters() -> None:
    rng = torch.Generator().manual_seed(0)
    blob_a = 0.05 * torch.randn((30, 2), generator=rng, dtype=torch.float64)
    blob_b = 0.05 * torch.randn((30, 2), generator=rng, dtype=torch.float64) + torch.tensor([5.0, 5.0], dtype=torch.float64)
    blob_c = 0.05 * torch.randn((30, 2), generator=rng, dtype=torch.float64) + torch.tensor([10.0, 10.0], dtype=torch.float64)
    outputs = torch.cat((blob_a, blob_b, blob_c), dim=0).unsqueeze(0)

    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=2,
            coverage=0.8,
            strategy="hdbscan",
            hdbscan_min_cluster_size=5,
        ),
    )

    assert cluster.num_clusters == 2
    assert int(cluster.k_per_row[0].item()) == 2


def test_hdbscan_respects_max_num_clusters_equals_one() -> None:
    outputs = _make_outputs()
    cluster = Cluster(
        outputs=outputs,
        parameters=ClusteringParameters(
            max_num_clusters=1,
            coverage=0.8,
            strategy="hdbscan",
            hdbscan_min_cluster_size=3,
        ),
    )

    assert cluster.num_clusters == 1
    assert torch.equal(cluster.k_per_row, torch.ones_like(cluster.k_per_row))
    assert torch.all(cluster.assignments == 0)


def test_hdbscan_invalid_selection_method_raises() -> None:
    with pytest.raises(ValueError):
        ClusteringParameters(
            max_num_clusters=2,
            coverage=0.8,
            strategy="hdbscan",
            hdbscan_cluster_selection_method="not-a-method",
        )


def test_behavior_split_requires_anchor() -> None:
    outputs = _make_outputs()
    with pytest.raises(ValueError):
        Cluster(
            outputs=outputs,
            parameters=ClusteringParameters(
                max_num_clusters=3,
                coverage=0.8,
                strategy="behavior_split",
            ),
        )


def test_behavior_split_finds_stay_and_turn_modes() -> None:
    row = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [-0.1, 0.1],
            [0.2, -0.2],
            [-2.0, -2.5],
            [-2.2, -2.3],
            [-1.8, -2.6],
            [-2.1, -2.4],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    anchor = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

    cluster = Cluster(
        outputs=row,
        parameters=ClusteringParameters(
            max_num_clusters=3,
            coverage=0.8,
            strategy="behavior_split",
            behavior_stay_radius=0.6,
            behavior_min_down=0.4,
            behavior_min_left=0.0,
        ),
        behavior_anchor=anchor,
    )

    assert int(cluster.k_per_row[0].item()) == 2
    assert cluster.num_clusters == 2
    assert float(cluster.probs_partition[0, 0].item()) > 0.0
    assert float(cluster.probs_partition[0, 1].item()) > 0.0
