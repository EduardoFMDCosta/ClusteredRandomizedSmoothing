from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch



ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
REPO_ROOT = EXPERIMENTS.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, Parameters, SmoothingParameters
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from stochastic_functions import TrajFlowMeszarosStochasticFunction
from plotting import plot_timestep_rectangles_visualization
from plotting import plot_final_timestep_density


def main():
    torch.manual_seed(0)

    try:
        stochastic_fn = TrajFlowMeszarosStochasticFunction()
    except ImportError as exc:
        print("Could not initialize TrajFlowMeszarosStochasticFunction.")
        print(str(exc))
        return

    # Load real trajectory data: (num_samples, 2 agents, 10 timesteps, 5 features)
    # Keep the model input layout as 2 agents x 10 timesteps x 2 coordinates (= 40 dims).
    data_file = str(EXPERIMENTS) + os.sep + "TrajFlowExperiment" + os.sep + "Data" + os.sep + "Inputs.npy"
    data = np.load(data_file)

    # Parameters
    selected_indices = np.array([480, 114, 129, 665], dtype=int)
    use_position_input = False  # True: smooth in position space x; False: smooth in control space u.
    sigma = 1.0
    coverage = 0.80
    num_plot_samples = 200

    batch_size = selected_indices.shape[0]
    x_input = data[selected_indices, :, :, :2]  # Take selected samples, both agents, (x, y) coords only.
    # Put the target agent in index 0 because TrajFlow predicts from agent slot 0.
    x_input = x_input[:,[1,0]]
    x = torch.from_numpy(x_input.reshape(batch_size, -1)).float()  # Shape: (batch_size, 40)

    if use_position_input:
        model_input = x
        helper = []
    else:
        u, start = stochastic_fn.pos_to_ctrl(x)
        model_input = u
        helper = start

    params = Parameters(
        smoothing=SmoothingParameters(N=10, alpha=0.35),
        clustering=ClusteringParameters(
            max_num_clusters=3,
            coverage=coverage,
            set_type="HyperRectangle",
            dbscan_eps=0.5,
            dbscan_min_samples=10,
        ),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-5),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=sigma,
        parameters=params,
        h=helper,
    )

    smoothed_distribution = smoother.smooth()
    radius = 0.005 * torch.ones((x.shape[0], 1), dtype=x.dtype)
    certificate = smoother.robust_certification(radius=radius)

    # Single-cluster smoother for comparison.
    params_single = Parameters(
        smoothing=SmoothingParameters(N=10, alpha=0.35),
        clustering=ClusteringParameters(
            max_num_clusters=1,
            coverage=coverage,
            set_type="HyperRectangle",
            dbscan_eps=0.5,
            dbscan_min_samples=10,
        ),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-5),
    )

    smoother_single = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=sigma,
        parameters=params_single,
        h=helper,
    )
    smoother_single.smooth()


    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype), 
        cov=sigma ** 2 * torch.eye(model_input.shape[-1], dtype=model_input.dtype)
    )

    # Monte Carlo simulation for the two smoothed methods.
    mc_samples = stochastic_fn.sample_from_noisy_inputs(
        x=model_input,
        smoothing_distribution=smoothing_distribution,
        num_samples=num_plot_samples,
        h=helper,
    )

    # Original means f(x), not f(x + epsilon), for comparison in the plots
    original_inputs = model_input.unsqueeze(1).expand(-1, num_plot_samples, -1)
    if helper:
        original_helper = [h_i.unsqueeze(1).expand(-1, num_plot_samples, -1) for h_i in helper]
        original_mc_samples = stochastic_fn(original_inputs, h=original_helper)
    else:
        original_mc_samples = stochastic_fn(original_inputs)

    single_cluster_mc_samples = smoother_single.monte_carlo(num_simulations=num_plot_samples)
    clusterized_mc_samples = smoother.monte_carlo(num_simulations=num_plot_samples)

    chart_paths = plot_final_timestep_density(
        x=x,
        original_mc_samples=original_mc_samples,
        noisy_mc_samples=mc_samples,
        single_cluster_mc_samples=single_cluster_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        single_cluster_centroids=smoother_single.cluster.centroids,
        clusterized_centroids=smoother.cluster.centroids,
        output_dir=EXPERIMENTS / "data" / "figures",
        num_input_agents=2,
    )

    chart_paths += plot_timestep_rectangles_visualization(
        x=x,
        original_mc_samples=original_mc_samples,
        noisy_mc_samples=mc_samples,
        single_cluster_mc_samples=single_cluster_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        region=smoother.regions,
        single_cluster_region=smoother_single.regions,
        output_dir=EXPERIMENTS / "data" / "figures",
        num_input_agents=2,
    )
    

    print("=== TrajFlow Smoothing Experiment ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"smoothed locs shape: {tuple(smoothed_distribution.locs.shape)}")
    print(f"smoothed probs shape: {tuple(smoothed_distribution.probs.shape)}")
    print(f"monte carlo samples shape: {tuple(mc_samples.shape)}")
    print(f"certificate radius shape: {tuple(certificate.radius.shape)}")
    print(f"certificate probs shape: {tuple(certificate.probs.shape)}")

    print(f"certificate radius: {certificate.radius}")
    print(f"certificate probs: {certificate.probs}")
    print(f"total prob: {certificate.total_probs}")

    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()
