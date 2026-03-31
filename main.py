from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make src/ importable when running `python main.py`.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.clusters import Cluster, ClusterParameters  # noqa: E402
from randomized_smoothing.distributions import MultivariateGaussian  # noqa: E402
from randomized_smoothing.examples import ThreeMode1DGaussianMixture  # noqa: E402
from randomized_smoothing.regions import HyperRectangle  # noqa: E402
from randomized_smoothing.smoothing import (  # noqa: E402
    AlphaTrimmingSmoothing,
    ClusterizedAlphaTrimmingSmoothing,
    SmoothingParameters,
)
from plotting import plot_experiment_results  # noqa: E402


def _draw_empirical_samples(smoother, num_draws: int) -> torch.Tensor:
    draws = [smoother.smooth().sample() for _ in range(num_draws)]
    return torch.stack(draws, dim=1)


def main() -> None:
    torch.manual_seed(0)

    x = torch.tensor([[0.0], [5.0]], dtype=torch.float64)
    stochastic_fn = ThreeMode1DGaussianMixture(
        locs=torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64),
        probs=torch.tensor([0.40, 0.20, 0.40], dtype=torch.float64),
        std=0.5,
    )

    noise_mean = torch.zeros(x.shape[-1], dtype=x.dtype)
    noise_cov = 1.5 * torch.eye(x.shape[-1], dtype=x.dtype)
    smoothing_distribution = MultivariateGaussian(mean=noise_mean, cov=noise_cov)

    alpha = 0.25
    region_mass = 0.50
    num_empirical_samples = 5000
    num_cluster_samples = 5000
    num_smoother_draws = 2000

    raw_noise = smoothing_distribution.sample(size=(x.shape[0], num_empirical_samples))
    raw_outputs = stochastic_fn.sample(x.unsqueeze(1) + raw_noise)

    cluster_noise = smoothing_distribution.sample(size=(x.shape[0], num_cluster_samples))
    cluster_outputs = stochastic_fn.sample(x.unsqueeze(1) + cluster_noise)
    cluster = Cluster(
        outputs=cluster_outputs,
        params=ClusterParameters(num_clusters=3),
    )
    region = HyperRectangle.from_samples(samples=cluster_outputs, cluster=cluster, p=region_mass)

    alpha_params = SmoothingParameters(
        smoothing_distribution=smoothing_distribution,
        N=200,
        alpha=alpha,
    )
    alpha_smoother = AlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        params=alpha_params,
    )

    clusterized_params = SmoothingParameters(
        smoothing_distribution=smoothing_distribution,
        N=200,
        alpha=alpha,
        cluster=cluster,
    )
    clusterized_smoother = ClusterizedAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        params=clusterized_params,
    )

    global_trimmed_mc_samples = _draw_empirical_samples(alpha_smoother, num_smoother_draws)
    clusterized_mc_samples = _draw_empirical_samples(clusterized_smoother, num_smoother_draws)

    chart_paths = plot_experiment_results(
        x=x,
        outputs=raw_outputs,
        global_trimmed_mc_samples=global_trimmed_mc_samples,
        clusterized_mc_samples=clusterized_mc_samples,
        region=region,
        output_dir=ROOT / "outputs",
    )

    print("=== Experiment Summary ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"raw outputs shape: {tuple(raw_outputs.shape)}")
    print(f"alpha-trimming samples shape: {tuple(global_trimmed_mc_samples.shape)}")
    print(f"clusterized samples shape: {tuple(clusterized_mc_samples.shape)}")
    print(f"cluster centroids: {cluster.centroids.squeeze(0).squeeze(-1).tolist()}")
    print(f"cluster weights: {cluster.weights.squeeze(0).tolist()}")
    print(f"region bounds shape: {tuple(region.lower.shape)}")
    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()