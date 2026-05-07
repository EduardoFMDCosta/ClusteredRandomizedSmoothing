from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from plotting import plot_combined_smoother_distributions
from stochastic_functions import (
    MultiplicativeThreeMode1DGaussianMixture,
)

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.smoothing import (
    GaussianAlphaTrimmingSmoothing,
)


def main():
    torch.manual_seed(0)

    # Define stochastic function and inputs x
    x = torch.tensor([[2.0]], dtype=torch.float64)
    stochastic_fn = MultiplicativeThreeMode1DGaussianMixture(
        locs=torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64),
        probs=torch.tensor([0.20, 0.20, 0.60], dtype=torch.float64),
        stds=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
    )

    # Define parameters
    coverage = 0.9
    smoothing_parameters = SmoothingParameters(
        N=30,
        alpha=0.45,
    )
    clustering_parameters = ClusteringParameters(
        max_num_clusters=2,
        coverage=coverage,
        strategy="hdbscan",
        hdbscan_min_cluster_size=1000,
        hdbscan_min_samples=100,
    )
    confidence_parameters = ConfidenceParameters(
        beta=1e-2,
        num_samples=10000,
    )

    sigma = 1.0
    # Define smoothing distribution
    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(x.shape[-1], dtype=x.dtype), 
        cov=sigma ** 2 * torch.eye(x.shape[-1], dtype=x.dtype)
    )

    parameters = Parameters(
        smoothing=smoothing_parameters,
        clustering=clustering_parameters,
        confidence=confidence_parameters,
    )

    # Construct cluster and region of high probability mass based on samples from h_w(x + noise)
    raw_outputs = stochastic_fn.sample_from_noisy_inputs(
        x=x,
        smoothing_distribution=smoothing_distribution,
        num_samples=5000
    )
    
    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=parameters,
    )
    print(
        "DBSCAN cluster summary:",
        {
            "k_per_row": smoother.cluster.k_per_row.tolist(),
            "probs_partition": smoother.cluster.probs_partition[0].detach().cpu().tolist(),
        },
    )

    rekavandi_smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=Parameters(
            smoothing=smoothing_parameters,
            clustering=ClusteringParameters(
                max_num_clusters=1,
                coverage=coverage,
            ),
            confidence=confidence_parameters,
        ),
    )

    # Plot empirical distributions of raw outputs and smoothed outputs.
    rekavandi_mc_samples = rekavandi_smoother.monte_carlo(num_simulations=5000)
    clustered_mc_samples = smoother.monte_carlo(num_simulations=5000)

    # Share bins, x- and y-axis limits across the two plots so the
    # original-model histogram is identical on both panels.
    raw_np = raw_outputs[:, :, 0].detach().cpu().numpy().reshape(-1)
    rekavandi_np = rekavandi_mc_samples[:, :, 0].detach().cpu().numpy().reshape(-1)
    clustered_np = clustered_mc_samples[:, :, 0].detach().cpu().numpy().reshape(-1)
    raw_bin_edges = np.histogram_bin_edges(raw_np, bins=60)
    mc_bin_edges = np.histogram_bin_edges(
        np.concatenate([rekavandi_np, clustered_np]), bins=45
    )
    raw_bin_width = float(raw_bin_edges[1] - raw_bin_edges[0])
    mc_bin_width = float(mc_bin_edges[1] - mc_bin_edges[0])
    raw_density_max = (
        np.histogram(raw_np, bins=raw_bin_edges)[0].max() / (raw_np.size * raw_bin_width)
    )
    rekavandi_density_max = (
        np.histogram(rekavandi_np, bins=mc_bin_edges)[0].max()
        / (rekavandi_np.size * mc_bin_width)
    )
    clustered_density_max = (
        np.histogram(clustered_np, bins=mc_bin_edges)[0].max()
        / (clustered_np.size * mc_bin_width)
    )
    all_data = np.concatenate([raw_np, rekavandi_np, clustered_np])
    margin = 0.05 * (all_data.max() - all_data.min())
    shared_xlim = (float(all_data.min() - margin), float(all_data.max() + margin))
    shared_ylim = (
        0.0,
        1.10 * float(max(raw_density_max, rekavandi_density_max, clustered_density_max)),
    )

    figures_dir = EXPERIMENTS / "data" / "figures"
    combined_path = plot_combined_smoother_distributions(
        x=x,
        outputs=raw_outputs,
        panels=[
            (
                rekavandi_mc_samples,
                rekavandi_smoother.regions,
                r"$\alpha$-smoothing",
                "#E31A1C",
            ),
            (
                clustered_mc_samples,
                smoother.regions,
                r"Clustered $\alpha$-smoothing",
                "#1F78B4",
            ),
        ],
        output_path=figures_dir / "smoothing_distributions_comparison.pdf",
        raw_bins=raw_bin_edges,
        mc_bins=mc_bin_edges,
        xlim=shared_xlim,
        ylim=shared_ylim,
    )
    chart_paths = [combined_path]

    # Robustness certification
    radius = 0.01 * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    certificate = smoother.robust_certification(radius=radius)
    print(f"certificate: {certificate}")

    # certificate = clustered_alpha_trimming_smoother.robust_certification(
    #     region_coverage=region_coverage, 
    #     beta=beta, 
    #     num_samples=num_samples, 
    #     radius=0.1
    # )
    # print(f"certificate_clustered_alpha_trimming: {certificate}")

    # print("=== Experiment Summary ===")
    # print(f"region coverage: {region_coverage}")
    # #print(f"certified p: {certified_p}")
    # print(f"x shape: {tuple(x.shape)}")
    # print(f"raw outputs shape: {tuple(raw_outputs.shape)}")
    # print(f"alpha-trimming samples shape: {tuple(alpha_trimming_mc_samples.shape)}")
    # print(f"clustered samples shape: {tuple(clustered_alpha_trimming_mc_samples.shape)}")
    # print(f"cluster centroids: {cluster.centroids.squeeze(0).squeeze(-1).tolist()}")
    # print(f"cluster weights: {cluster.weights.squeeze(0).tolist()}")
    # print(f"region bounds shape: {tuple(regions.lower.shape)}")
    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()