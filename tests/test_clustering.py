from __future__ import annotations

import torch

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.dataclasses import ClusteringParameters
from randomized_smoothing.regions import DisjointRegion, HyperRectangle


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


def test_disjoint_region_assigns_overlap_to_single_cluster() -> None:
    lower = torch.tensor([[[0.0, 0.0], [0.3, 0.3]]], dtype=torch.float64)
    upper = torch.tensor([[[0.5, 0.5], [0.8, 0.8]]], dtype=torch.float64)
    centroids = torch.tensor([[[0.2, 0.2], [0.6, 0.6]]], dtype=torch.float64)
    points = torch.tensor([[[0.35, 0.35], [0.45, 0.45], [0.65, 0.65]]], dtype=torch.float64)

    region = DisjointRegion(region=HyperRectangle(lower=lower, upper=upper), centroids=centroids)
    contains = region.contains(points)

    assert torch.all(contains.sum(dim=-1) <= 1)
    assert bool(contains[0, 0, 0]) and not bool(contains[0, 0, 1])
    assert bool(contains[0, 1, 1]) and not bool(contains[0, 1, 0])
    assert bool(contains[0, 2, 1]) and not bool(contains[0, 2, 0])


def test_hyperrectangle_is_clipped_by_voronoi_boundary() -> None:
    # Two centroids on x-axis; Voronoi boundary is x=1.
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

    region = HyperRectangle.from_samples(
        samples=samples,
        centroids=centroids,
        assignments=assignments,
        coverage=1.0,
    )

    # Cluster-0 box right face and cluster-1 box left face must not cross x=1.
    assert float(region.upper[0, 0, 0].item()) <= 1.0 + 1e-9
    assert float(region.lower[0, 1, 0].item()) >= 1.0 - 1e-9
