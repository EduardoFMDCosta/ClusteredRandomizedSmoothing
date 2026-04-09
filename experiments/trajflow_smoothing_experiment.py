from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.dataclasses import ClusteringParameters, ConfidenceParameters, Parameters, SmoothingParameters
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from stochastic_functions import TrajFlowMeszarosStochasticFunction


def main():
    torch.manual_seed(0)

    try:
        stochastic_fn = TrajFlowMeszarosStochasticFunction()
    except ImportError as exc:
        print("Could not initialize TrajFlowMeszarosStochasticFunction.")
        print(str(exc))
        return

    # One input trajectory has 10 timesteps x 2 coordinates = 20 features.
    x = torch.randn(4, 20, dtype=torch.float32)

    params = Parameters(
        smoothing=SmoothingParameters(N=20, alpha=0.2),
        clustering=ClusteringParameters(max_num_clusters=10, coverage=0.9),
        confidence=ConfidenceParameters(num_samples=5000, beta=1e-2),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=0.5,
        parameters=params,
    )

    smoothed_distribution = smoother.smooth()
    mc_samples = smoother.monte_carlo(num_simulations=10)
    radius = 0.05 * torch.ones((x.shape[0], 1), dtype=x.dtype)
    certificate = smoother.robust_certification(radius=radius)

    print("=== TrajFlow Smoothing Experiment ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"smoothed locs shape: {tuple(smoothed_distribution.locs.shape)}")
    print(f"smoothed probs shape: {tuple(smoothed_distribution.probs.shape)}")
    print(f"monte carlo samples shape: {tuple(mc_samples.shape)}")
    print(f"certificate radius shape: {tuple(certificate.radius.shape)}")
    print(f"certificate probs shape: {tuple(certificate.probs.shape)}")


if __name__ == "__main__":
    main()
