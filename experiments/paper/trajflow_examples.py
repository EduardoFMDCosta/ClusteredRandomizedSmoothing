from __future__ import annotations

import argparse
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
from experiment_configs import EXAMPLES as cfg


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
    """All smoothing/model parameters are in experiment_configs.py (EXAMPLES)."""
    parser = argparse.ArgumentParser(
        description=(
            "TrajFlow smoothing visualisation — per-scenario plots comparing\n"
            "the single-cluster and multi-cluster smoothed distributions.\n"
            "Edit experiment_configs.EXAMPLES to change model/smoothing parameters."
        )
    )
    parser.add_argument(
        "--selected-indices",
        type=str,
        default=cfg.selected_indices,
        help="Comma-separated dataset indices to visualise (default: from config).",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "figures"),
        help="Directory where output figures are saved.",
    )
    args = parser.parse_args()

    torch.manual_seed(cfg.seed)

    try:
        if cfg.output_mode == "trajectory":
            stochastic_fn = TrajFlowMeszarosFullTrajectoryStochasticFunction()
        else:
            stochastic_fn = TrajFlowMeszarosStochasticFunction()
    except ImportError as exc:
        print("Could not initialise TrajFlow stochastic function.")
        print(str(exc))
        return

    # Load real trajectory data: (num_samples, 2 agents, 10 timesteps, 5 features).
    data_file = EXPERIMENTS / "TrajFlowExperiment" / "Data" / "Inputs.npy"
    data = np.load(data_file)

    selected_indices = _parse_indices(args.selected_indices)
    batch_size = selected_indices.shape[0]
    x_input = data[selected_indices, :, :, :2]   # both agents, (x, y) coords only
    x_input = x_input[:, [1, 0]]                  # target agent in slot 0
    x = torch.from_numpy(x_input.reshape(batch_size, -1)).float()  # (batch_size, 40)

    if cfg.input_mode == "position":
        model_input = x
        helper = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x)

    params = Parameters(
        smoothing=SmoothingParameters(N=cfg.smoothing_N, alpha=cfg.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=2,
            coverage=cfg.coverage,
            strategy=cfg.cluster_strategy,
            dbscan_eps=cfg.dbscan_eps,
            dbscan_min_samples=cfg.dbscan_min_samples,
            behavior_stay_radius=cfg.behavior_stay_radius,
            behavior_min_down=cfg.behavior_min_down,
            behavior_min_left=cfg.behavior_min_left,
        ),
        confidence=ConfidenceParameters(num_samples=cfg.confidence_num_samples, beta=1e-5),
    )

    # For endpoint_dbscan we do not need a behavior anchor (data-driven from
    # model outputs). behavior_split requires the initial position of the target agent.
    cluster_anchor_for_smoother = (
        None if cfg.cluster_strategy == "endpoint_dbscan"
        else _extract_target_initial_position(x)
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=params,
        h=helper,
        cluster_anchor=cluster_anchor_for_smoother,
    )

    smoothed_distribution = smoother.smooth()
    radius = 0.005 * torch.ones((x.shape[0], 1), dtype=x.dtype)
    certificate = smoother.robust_certification(radius=radius)

    # Single-cluster smoother for comparison.
    params_single = Parameters(
        smoothing=SmoothingParameters(N=cfg.smoothing_N, alpha=cfg.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=1,
            coverage=cfg.coverage,
            strategy="dbscan",
            dbscan_eps=cfg.dbscan_eps,
            dbscan_min_samples=cfg.dbscan_min_samples,
        ),
        confidence=ConfidenceParameters(num_samples=cfg.confidence_num_samples, beta=1e-5),
    )
    smoother_single = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=cfg.sigma,
        parameters=params_single,
        h=helper,
        cluster_anchor=None,
    )
    smoother_single.smooth()

    smoothing_distribution = MultivariateGaussian(
        mean=torch.zeros(model_input.shape[-1], dtype=model_input.dtype),
        cov=cfg.sigma ** 2 * torch.eye(model_input.shape[-1], dtype=model_input.dtype),
    )

    # Monte Carlo simulation for the two smoothed methods.
    mc_samples = stochastic_fn.sample_from_noisy_inputs(
        x=model_input,
        smoothing_distribution=smoothing_distribution,
        num_samples=cfg.num_plot_samples,
        h=helper,
    )

    # Original means f(x), not f(x + epsilon), for comparison in the plots.
    original_inputs = model_input.unsqueeze(1).expand(-1, cfg.num_plot_samples, -1)
    if helper:
        original_helper = [h_i.unsqueeze(1).expand(-1, cfg.num_plot_samples, -1) for h_i in helper]
        original_mc_samples = stochastic_fn(original_inputs, h=original_helper)
    else:
        original_mc_samples = stochastic_fn(original_inputs)

    single_cluster_mc_samples = smoother_single.monte_carlo(num_simulations=cfg.num_plot_samples)
    clustered_mc_samples = smoother.monte_carlo(num_simulations=cfg.num_plot_samples)

    figures_dir = Path(args.figures_dir)
    if cfg.output_mode == "trajectory":
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
            output_dir=figures_dir,
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
            output_dir=figures_dir,
            num_input_agents=2,
        )

    print("=== TrajFlow Smoothing Experiment ===")
    print(f"x shape: {tuple(x.shape)}")
    print(f"smoothed locs shape: {tuple(smoothed_distribution.locs.shape)}")
    print(f"smoothed probs shape: {tuple(smoothed_distribution.probs.shape)}")
    print(f"output mode: {cfg.output_mode}")
    print(f"cluster strategy: {cfg.cluster_strategy}")
    print(f"dbscan k_per_row: {smoother.cluster.k_per_row.tolist()}")
    print(f"monte carlo samples shape: {tuple(mc_samples.shape)}")
    print(f"certificate radius shape: {tuple(certificate.radius.shape)}")
    print(f"certificate probs shape: {tuple(certificate.probs.shape)}")
    print(f"certificate radius: {certificate.radius}")
    print(f"certificate probs: {certificate.probs}")
    print(f"total prob: {certificate.total_probs}")
    if cfg.cluster_strategy == "behavior_split":
        print("behavior-split partition probabilities [stay, turn-left/down]:")
        for i, dataset_idx in enumerate(selected_indices.tolist()):
            p_stay = float(smoother.cluster.probs_partition[i, 0].item())
            p_turn = float(smoother.cluster.probs_partition[i, 1].item())
            print(f"  idx={dataset_idx}: [{p_stay:.3f}, {p_turn:.3f}]")

    print("\nSaved charts:")
    for path in chart_paths:
        print(path)


if __name__ == "__main__":
    main()
