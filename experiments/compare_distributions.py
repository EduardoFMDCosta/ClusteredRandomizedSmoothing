from __future__ import annotations

import sys
from pathlib import Path

import torch

from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, Parameters, SmoothingParameters

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.clusters import Cluster
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.stochastic import ThreeMode1DGaussianMixture
from randomized_smoothing.regions import HyperRectangle
from randomized_smoothing.smoothing import (
    AlphaTrimmingSmoothing,
)
from plotting import plot_experiment_results


def main():
    torch.manual_seed(0)
    
	# Define stochastic function and inputs x
    x = torch.tensor([[0.0], [5.0]], dtype=torch.float64)
    stochastic_fn = ThreeMode1DGaussianMixture(
        locs=torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64),
        probs=torch.tensor([0.40, 0.20, 0.40], dtype=torch.float64),
        std=0.5,
    )

	# Define smoothing distribution
    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(x.shape[-1], dtype=x.dtype), 
        cov=1.0 * torch.eye(x.shape[-1], dtype=x.dtype)
    )

	# Define parameters
    smoothing_parameters = SmoothingParameters(
        N=50,
        alpha=0.25,
    )
    clustering_parameters = ClusteringParameters(
        num_clusters=3,
        coverage=0.8,
    )
    confidence_parameters = ConfidenceParameters(
        beta=1e-2,
        num_samples=1000,
    )
    
    parameters = Parameters(
        smoothing=smoothing_parameters,
        clustering=clustering_parameters,
        confidence=confidence_parameters,
    )

	# Construct cluster and region of high probability mass based on samples from h_w(x + noise)
    raw_outputs = stochastic_fn.sample(
        x.unsqueeze(1) + smoothing_distribution.sample(size=(x.shape[0], 5000))
    )
    
    alpha_trimming_smoother = AlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        smoothing_distribution=smoothing_distribution,
        parameters=parameters,
    )

	# Plot empirical distributions of raw outputs and smoothed outputs
    alpha_trimming_mc_samples = alpha_trimming_smoother.monte_carlo(num_simulations=5000)

    chart_paths = plot_experiment_results(
        x=x,
        outputs=raw_outputs,
        global_trimmed_mc_samples=alpha_trimming_mc_samples,
        clusterized_mc_samples=alpha_trimming_mc_samples, # TODO: FIX
        region=alpha_trimming_smoother.regions,
        output_dir=ROOT / "outputs",
    )

    # Robustness certification
    # certificate_alpha_trimming = alpha_trimming_smoother.robust_certification(
    #     region_coverage=region_coverage, 
    #     beta=beta, 
    #     num_samples=num_samples, 
    #     radius=0.1
    # )
    # print(f"certificate_alpha_trimming: {certificate_alpha_trimming}")

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