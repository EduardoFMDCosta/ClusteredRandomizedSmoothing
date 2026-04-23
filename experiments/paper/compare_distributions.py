from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, Parameters, SmoothingParameters
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.smoothing import (
    GaussianAlphaTrimmingSmoothing,
)
from stochastic_functions import ThreeMode1DGaussianMixture
from plotting import plot_experiment_results


def main():
    torch.manual_seed(0)
    plot_region_shading = False

    # Define stochastic function and inputs x
    x = torch.tensor([[0.0]], dtype=torch.float64)
    stochastic_fn = ThreeMode1DGaussianMixture(
        locs=torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64),
        probs=torch.tensor([0.40, 0.20, 0.40], dtype=torch.float64),
        std=0.5,
    )

    # Define parameters
    coverage = 0.9
    smoothing_parameters = SmoothingParameters(
        N=50,
        alpha=0.45,
    )
    clustering_parameters = ClusteringParameters(
        max_num_clusters=3,
        coverage=coverage,
        dbscan_eps=0.2,
        dbscan_min_samples=80,
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
    clusterized_mc_samples = smoother.monte_carlo(num_simulations=5000)

    chart_paths = plot_experiment_results(
        x=x,
        outputs=raw_outputs,
        rekavandi_mc_samples=rekavandi_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        region=smoother.regions,
        plot_region_shading=plot_region_shading,
        output_dir=EXPERIMENTS / "data" / "figures",
    )

    # Robustness certification
    radius = 0.01 * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    certificate = smoother.robust_certification(radius=radius)
    print(f"certificate: {certificate}")

    # certificate = clusterized_alpha_trimming_smoother.robust_certification(
    #     region_coverage=region_coverage, 
    #     beta=beta, 
    #     num_samples=num_samples, 
    #     radius=0.1
    # )
    # print(f"certificate_clusterized_alpha_trimming: {certificate}")

    # print("=== Experiment Summary ===")
    # print(f"region coverage: {region_coverage}")
    # #print(f"certified p: {certified_p}")
    # print(f"x shape: {tuple(x.shape)}")
    # print(f"raw outputs shape: {tuple(raw_outputs.shape)}")
    # print(f"alpha-trimming samples shape: {tuple(alpha_trimming_mc_samples.shape)}")
    # print(f"clusterized samples shape: {tuple(clusterized_alpha_trimming_mc_samples.shape)}")
    # print(f"cluster centroids: {cluster.centroids.squeeze(0).squeeze(-1).tolist()}")
    # print(f"cluster weights: {cluster.weights.squeeze(0).tolist()}")
    # print(f"region bounds shape: {tuple(regions.lower.shape)}")
    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()