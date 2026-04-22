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
from plotting import plot_timestep_rectangles_visualization, plot_trajectory_samples


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

    batch_size = 4
    x_input = data[:batch_size, :, :, :2]  # Take first 4 samples, both agents, (x, y) coords only.
    # Put the target agent in index 0 because TrajFlow predicts from agent slot 0.
    x_input = x_input[:,[1,0]]
    x = torch.from_numpy(x_input.reshape(batch_size, -1)).float()  # Shape: (batch_size, 40)
    u, start = stochastic_fn.pos_to_ctrl(x)

    sigma = 0.1 # sigma=1 would correspond to the maximum allowable deviation
    params = Parameters(
        smoothing=SmoothingParameters(N=10, alpha=0.35),
        clustering=ClusteringParameters(
            max_num_clusters=3,
            coverage=0.7,
            set_type="TimeStepHyperrectangles",
        ),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-5),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=u,
        sigma=sigma,
        parameters=params,
        h=start,  # Pass the starting positions as helper variables for the stochastic function.
    )

    smoothed_distribution = smoother.smooth()
    radius = 0.001 * torch.ones((x.shape[0], 1), dtype=x.dtype)
    certificate = smoother.robust_certification(radius=radius)

    # Single-cluster smoother for comparison.
    params_single = Parameters(
        smoothing=SmoothingParameters(N=10, alpha=0.35),
        clustering=ClusteringParameters(
            max_num_clusters=1,
            coverage=0.7,
            set_type="TimeStepHyperrectangles",
        ),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-5),
    )

    smoother_single = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=u,
        sigma=sigma,
        parameters=params_single,
        h=start,  # Pass the starting positions as helper variables for the stochastic function.
    )
    smoother_single.smooth()


    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(u.shape[-1], dtype=u.dtype), 
        cov=sigma ** 2 * torch.eye(u.shape[-1], dtype=u.dtype)
    )

    # Monte Carlo simulation
    mc_samples = stochastic_fn.sample_from_noisy_inputs(
        x=u,
        smoothing_distribution=smoothing_distribution,
        num_samples=100,
        h=start,  # Pass the starting positions as helper variables for the stochastic function.
    )

    single_cluster_mc_samples = smoother_single.monte_carlo(num_simulations=100)
    clusterized_mc_samples = smoother.monte_carlo(num_simulations=100)

    chart_paths = plot_trajectory_samples(
        x=x,
        mc_samples=mc_samples,
        single_cluster_mc_samples=single_cluster_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        output_dir=EXPERIMENTS / "data" / "figures",
        num_input_agents=2,
    )

    chart_paths += plot_timestep_rectangles_visualization(
        x=x,
        mc_samples=mc_samples,
        single_cluster_mc_samples=single_cluster_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        region=smoother.regions,
        single_cluster_region=smoother_single.regions,
        output_dir=EXPERIMENTS / "data" / "figures",
        num_input_agents=2,
        x_index=0,
    )

    num_output_timesteps = smoother.regions.lower.shape[-1] // 2
    chart_paths += plot_timestep_rectangles_visualization(
        x=x,
        mc_samples=mc_samples,
        single_cluster_mc_samples=single_cluster_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        region=smoother.regions,
        single_cluster_region=smoother_single.regions,
        output_dir=EXPERIMENTS / "data" / "figures",
        num_input_agents=2,
        x_index=0,
        timesteps=[0, num_output_timesteps - 1],
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
