from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from randomized_smoothing.distributions import MultivariateGaussian


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, Parameters, SmoothingParameters
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from stochastic_functions import TrajFlowMeszarosStochasticFunction
from plotting import plot_trajectory_samples


def main():
    torch.manual_seed(0)

    try:
        stochastic_fn = TrajFlowMeszarosStochasticFunction()
    except ImportError as exc:
        print("Could not initialize TrajFlowMeszarosStochasticFunction.")
        print(str(exc))
        return

    # Load real trajectory data: (323 samples, 2 agents, 10 timesteps, 5 features)
    # Extract first agent's (x,y) coordinates and reshape to (batch, 20)
    data = np.load(ROOT / "trajectory_prediction" / "Data" / "Inputs.npy")
    x_input = data[:4, 0, :, :2]  # Take first 4 samples, first agent, first 2 coords (x,y)
    x = torch.from_numpy(x_input.reshape(4, -1)).float()  # Shape: (4, 20)

    sigma = 1.0
    params = Parameters(
        smoothing=SmoothingParameters(N=20, alpha=0.2),
        clustering=ClusteringParameters(max_num_clusters=10, coverage=0.9),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-3),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=sigma,
        parameters=params,
    )

    smoothed_distribution = smoother.smooth()
    radius = 0.05 * torch.ones((x.shape[0], 1), dtype=x.dtype)
    certificate = smoother.robust_certification(radius=radius)


    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(x.shape[-1], dtype=x.dtype), 
        cov=sigma ** 2 * torch.eye(x.shape[-1], dtype=x.dtype)
    )

    # Monte Carlo simulation
    mc_samples = stochastic_fn.sample_from_noisy_inputs(
        x=x,
        smoothing_distribution=smoothing_distribution,
        num_samples=100
    )

    clusterized_mc_samples = smoother.monte_carlo(num_simulations=100)

    chart_paths = plot_trajectory_samples(
        mc_samples=mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        output_dir=ROOT / "outputs",
        num_timesteps=20,
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

    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()
