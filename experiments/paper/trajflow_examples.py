from __future__ import annotations

import argparse
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

from plotting import plot_final_timestep_examples, plot_trajectory_examples_clustered
from stochastic_functions import (
    TrajFlowMeszarosStochasticFunction,
    TrajFlowMeszarosFullTrajectoryStochasticFunction,
)

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.distributions import MultivariateGaussian
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing


def _parse_indices(text: str) -> np.ndarray:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("selected-indices must contain at least one integer")
    return np.array([int(v) for v in values], dtype=int)


def _extract_target_initial_position(x_positions: torch.Tensor, num_input_agents: int = 2) -> torch.Tensor:
    nx = x_positions.shape[0]
    num_input_timesteps = x_positions.shape[1] // (2 * num_input_agents)
    x_traj = x_positions.reshape(nx, num_input_agents, num_input_timesteps, 2)
    return x_traj[:, 0, 0, :]


def main():
    parser = argparse.ArgumentParser(description="TrajFlow smoothing experiment with plotting.")
    parser.add_argument("--selected-indices", type=str, default="157,203,153,2,41",
                        help="Comma-separated dataset indices to analyze.")
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=("endpoint", "trajectory"),
        default="endpoint",
        help=(
            "Whether to use the 2-D endpoint output (endpoint) or the full "
            "predicted trajectory flattened to 40-D (trajectory).  "
            "Trajectory mode clusters in the full output space without "
            "any endpoint-specific projection."
        ),
    )
    parser.add_argument(
        "--cluster-strategy",
        type=str,
        choices=("dbscan", "behavior_split", "trajectory_dbscan"),
        default=None,
        help=(
            "Clustering strategy for the multi-cluster smoother.  "
            "Defaults to 'behavior_split' for endpoint mode and "
            "'trajectory_dbscan' for trajectory mode."
        ),
    )
    parser.add_argument("--behavior-stay-radius", type=float, default=1.0,
                        help="Stay mode threshold: ||y - y_init|| <= radius.")
    parser.add_argument("--behavior-min-down", type=float, default=0.5,
                        help="Turn mode threshold on axis x2: y2 - y2_init <= -min_down.")
    parser.add_argument("--behavior-min-left", type=float, default=0.0,
                        help="Turn mode threshold on axis x1: y1 - y1_init <= -min_left.")
    parser.add_argument("--dbscan-eps", type=float, default=None,
                        help=(
                            "DBSCAN eps.  Defaults to 0.5 for endpoint mode and "
                            "8.0 for trajectory mode (40-D space)."
                        ))
    parser.add_argument("--dbscan-min-samples", type=int, default=100,
                        help="DBSCAN min_samples used by the smoothing clusterer.")
    parser.add_argument("--num-plot-samples", type=int, default=500,
                        help="Monte Carlo samples used for plotting distributions.")
    parser.add_argument("--confidence-num-samples", type=int, default=100,
                        help="num_samples used to fit cluster boundaries. Clopper-Pearson\n                        bounds are not needed here, so 100 is sufficient.")
    args = parser.parse_args()

    # Apply mode-dependent defaults.
    if args.cluster_strategy is None:
        args.cluster_strategy = "behavior_split" if args.output_mode == "endpoint" else "trajectory_dbscan"
    if args.dbscan_eps is None:
        args.dbscan_eps = 0.5 if args.output_mode == "endpoint" else 8.0

    torch.manual_seed(0)

    try:
        if args.output_mode == "trajectory":
            stochastic_fn = TrajFlowMeszarosFullTrajectoryStochasticFunction()
        else:
            stochastic_fn = TrajFlowMeszarosStochasticFunction()
    except ImportError as exc:
        print("Could not initialize TrajFlow stochastic function.")
        print(str(exc))
        return

    # Load real trajectory data: (num_samples, 2 agents, 10 timesteps, 5 features)
    # Keep the model input layout as 2 agents x 10 timesteps x 2 coordinates (= 40 dims).
    data_file = str(EXPERIMENTS) + os.sep + "TrajFlowExperiment" + os.sep + "Data" + os.sep + "Inputs.npy"
    data = np.load(data_file)

    # Parameters
    selected_indices = _parse_indices(args.selected_indices)
    use_position_input = False  # True: smooth in position space x; False: smooth in control space u.
    sigma = 1.0
    coverage = 0.80
    num_plot_samples = args.num_plot_samples

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
            max_num_clusters=2,
            coverage=coverage,
            strategy=args.cluster_strategy,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            behavior_stay_radius=args.behavior_stay_radius,
            behavior_min_down=args.behavior_min_down,
            behavior_min_left=args.behavior_min_left,
        ),
        confidence=ConfidenceParameters(num_samples=args.confidence_num_samples, beta=1e-5),
    )

    # For trajectory_dbscan we do not need a behavior anchor (endpoint-free).
    cluster_anchor_for_smoother = (
        None if args.cluster_strategy == "trajectory_dbscan"
        else _extract_target_initial_position(x)
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=sigma,
        parameters=params,
        h=helper,
        cluster_anchor=cluster_anchor_for_smoother,
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
            strategy="dbscan",
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
        ),
        confidence=ConfidenceParameters(num_samples=args.confidence_num_samples, beta=1e-5),
    )

    smoother_single = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=sigma,
        parameters=params_single,
        h=helper,
        cluster_anchor=None,
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
    clustered_mc_samples = smoother.monte_carlo(num_simulations=num_plot_samples)

    if args.output_mode == "trajectory":
        chart_paths = plot_trajectory_examples_clustered(
            x=x,
            original_mc_samples=original_mc_samples,
            noisy_mc_samples=mc_samples,
            single_cluster_mc_samples=single_cluster_mc_samples,
            clustered_mc_samples=clustered_mc_samples,
            single_cluster_centroids=smoother_single.cluster.centroids,
            clustered_centroids=smoother.cluster.centroids,
            region=smoother.regions,
            single_cluster_region=smoother_single.regions,
            output_dir=EXPERIMENTS / "data" / "figures",
            num_input_agents=2,
        )
    else:
        chart_paths = plot_final_timestep_examples(
            x=x,
            original_mc_samples=original_mc_samples,
            noisy_mc_samples=mc_samples,
            single_cluster_mc_samples=single_cluster_mc_samples,
            clustered_mc_samples=clustered_mc_samples,
            single_cluster_centroids=smoother_single.cluster.centroids,
            clustered_centroids=smoother.cluster.centroids,
            region=smoother.regions,
            single_cluster_region=smoother_single.regions,
            output_dir=EXPERIMENTS / "data" / "figures",
            num_input_agents=2,
        )
    

    print("=== TrajFlow Smoothing Experiment ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"smoothed locs shape: {tuple(smoothed_distribution.locs.shape)}")
    print(f"smoothed probs shape: {tuple(smoothed_distribution.probs.shape)}")
    print(f"output mode: {args.output_mode}")
    print(f"cluster strategy: {args.cluster_strategy}")
    print(f"dbscan k_per_row: {smoother.cluster.k_per_row.tolist()}")
    print(f"monte carlo samples shape: {tuple(mc_samples.shape)}")
    print(f"certificate radius shape: {tuple(certificate.radius.shape)}")
    print(f"certificate probs shape: {tuple(certificate.probs.shape)}")

    print(f"certificate radius: {certificate.radius}")
    print(f"certificate probs: {certificate.probs}")
    print(f"total prob: {certificate.total_probs}")
    if args.cluster_strategy == "behavior_split":
        print("behavior-split partition probabilities [stay, turn-left/down]:")
        for i, dataset_idx in enumerate(selected_indices.tolist()):
            p_stay = float(smoother.cluster.probs_partition[i, 0].item())
            p_turn = float(smoother.cluster.probs_partition[i, 1].item())
            print(f"  idx={dataset_idx}: [{p_stay:.3f}, {p_turn:.3f}]")

    print("")
    print("Saved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()
