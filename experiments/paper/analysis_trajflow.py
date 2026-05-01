from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from randomized_smoothing.dataclasses import (
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing
from stochastic_functions import TrajFlowMeszarosStochasticFunction


@dataclass
class Setting:
    alpha: float
    N: int
    max_num_clusters: int
    coverage: float
    sigma: float
    r: float


def load_inputs(batch_size: int | None = None) -> torch.Tensor:
    data_file = EXPERIMENTS / "TrajFlowExperiment" / "Data" / "Inputs.npy"
    data = np.load(str(data_file))

    if batch_size is None:
        batch_size = len(data)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(data):
        raise ValueError(f"batch_size={batch_size} exceeds available samples={len(data)}")

    x_input = data[:batch_size, :, :, :2]
    x_input = x_input[:, [1, 0]]  # Put the target agent in index 0.
    return torch.from_numpy(x_input.reshape(batch_size, -1)).float()


def load_ground_truth_outputs(batch_size: int | None = None) -> torch.Tensor:
    data_file = EXPERIMENTS / "TrajFlowExperiment" / "Data" / "Outputs_gt.npy"
    data = np.load(str(data_file))

    if batch_size is None:
        batch_size = len(data)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(data):
        raise ValueError(f"batch_size={batch_size} exceeds available GT samples={len(data)}")

    data = data[:batch_size]

    # Accept common GT layouts and extract the target agent at the final timestep.
    if data.ndim == 4 and data.shape[-1] == 2:
        # (num_samples, num_agents, num_timesteps, 2)
        if data.shape[1] < 1:
            raise ValueError("Outputs_gt.npy has no agent dimension entries")
        gt_final = data[:, 0, -1, :]
    elif data.ndim == 3 and data.shape[-1] == 2:
        # (num_samples, num_timesteps, 2)
        gt_final = data[:, -1, :]
    elif data.ndim == 2 and data.shape[-1] == 2:
        # (num_samples, 2)
        gt_final = data
    else:
        raise ValueError(
            "Unsupported Outputs_gt.npy shape. Expected (..., 2) with time and optional agent axes, "
            f"got {tuple(data.shape)}"
        )

    return torch.from_numpy(gt_final).float()


def build_settings() -> list[Setting]:
    alphas = [0.0, 0.2, 0.4]
    Ns = [10, 20, 50]
    max_clusters = [1, 3]
    coverages = [0.7, 0.9]
    sigmas = [0.1, 0.5, 1.0]

    settings: list[Setting] = []
    for alpha, N, max_num_clusters, coverage, sigma in itertools.product(
        alphas,
        Ns,
        max_clusters,
        coverages,
        sigmas,
    ):
        for denom in (10,):
            settings.append(
                Setting(
                    alpha=alpha,
                    N=N,
                    max_num_clusters=max_num_clusters,
                    coverage=coverage,
                    sigma=sigma,
                    r=sigma / denom,
                )
            )

    return settings


def evaluate_setting(
    setting: Setting,
    stochastic_fn: TrajFlowMeszarosStochasticFunction,
    model_input: torch.Tensor,
    helper: list[torch.Tensor],
    gt_output_final: torch.Tensor,
    error_num_samples: int,
    confidence_num_samples: int,
    confidence_beta: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one smoothing/certification setting and return serializable metrics."""
    if error_num_samples <= 0:
        raise ValueError("error_num_samples must be positive")

    torch.manual_seed(seed)

    params = Parameters(
        smoothing=SmoothingParameters(N=setting.N, alpha=setting.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=setting.max_num_clusters,
            coverage=setting.coverage,
        ),
        confidence=ConfidenceParameters(
            num_samples=confidence_num_samples,
            beta=confidence_beta,
        ),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=model_input,
        sigma=setting.sigma,
        parameters=params,
        h=helper,
    )

    radius = torch.full(
        (model_input.shape[0], 1),
        fill_value=setting.r,
        dtype=model_input.dtype,
        device=model_input.device,
    )
    certificate = smoother.robust_certification(radius=radius)

    smoothed_samples = smoother.monte_carlo(num_simulations=error_num_samples)
    if smoothed_samples.shape[-1] != gt_output_final.shape[-1]:
        raise ValueError(
            "Dimension mismatch between smoothed samples and GT outputs: "
            f"{smoothed_samples.shape[-1]} vs {gt_output_final.shape[-1]}"
        )
    # Per Monte Carlo draw: Euclidean residual ||y_hat - y_gt||_2, then aggregate over draws.
    l2_errors = (smoothed_samples - gt_output_final.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
    mean_l2_error_to_gt = l2_errors.mean(dim=1).detach().cpu()                  # (nx,)
    std_l2_error_to_gt = l2_errors.std(dim=1, unbiased=False).detach().cpu()    # (nx,)

    certified_radius = certificate.radius.squeeze(-1).detach().cpu()    # (nx,)
    certified_probs = certificate.probs.detach().cpu()                   # (nx, num_clusters)
    total_certified_prob = certificate.total_probs.detach().cpu()         # (nx,)

    # The output layout is assumed to be [x1, y1, x2, y2, ..., xT, yT].
    # Compute area per timestep from even/odd coordinate widths, sum over
    # clusters, then aggregate over timesteps.
    widths = smoother.regions.width.to(torch.float64)                           # (nx, k, 2T)
    if widths.shape[-1] % 2 != 0:
        raise ValueError("Expected even output dimension arranged as [x1,y1,...,xT,yT]")

    x_widths = widths[..., 0::2]                                                # (nx, k, T)
    y_widths = widths[..., 1::2]                                                # (nx, k, T)
    areas_per_cluster = x_widths * y_widths                                     # (nx, k, T)
    # T can be 1 (final timestep only) or larger (full trajectory output).
    areas_per_timestep = areas_per_cluster.sum(dim=1).detach().cpu()           # (nx, T)
    average_area = areas_per_timestep.mean(dim=-1)                             # (nx,)
    minimum_area = areas_per_timestep.min(dim=-1).values                       # (nx,)
    maximum_area = areas_per_timestep.max(dim=-1).values                       # (nx,)

    nx = model_input.shape[0]
    xs = [
        {
            "x_index": i,
            "certified_radius": float(certified_radius[i].item()),
            "minimum_prob": float(certified_probs[i].min().item()),
            "maximum_prob": float(certified_probs[i].max().item()),
            "total_certified_prob": float(total_certified_prob[i].item()),
            "mean_l2_error_to_gt": float(mean_l2_error_to_gt[i].item()),
            "std_l2_error_to_gt": float(std_l2_error_to_gt[i].item()),
            "average_area": float(average_area[i].item()),
            "minimum_area": float(minimum_area[i].item()),
            "maximum_area": float(maximum_area[i].item()),
        }
        for i in range(nx)
    ]

    return {
        "setting": asdict(setting),
        "xs": xs,
    }



def _sanitize_json_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_numbers(v) for v in obj]
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    return obj


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json_numbers(payload), f, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter analysis for TrajFlow randomized smoothing.")
    parser.add_argument("--batch-size", type=int, default=None, help="How many input trajectories to evaluate (default: all).")
    parser.add_argument("--sample-batch-size", type=int, default=16, help="Number of samples processed per forward pass (default: 16).")
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=("control", "position"),
        default="control",
        help="Input representation for smoothing: 'control' (default) or 'position'.",
    )
    parser.add_argument(
        "--confidence-num-samples",
        type=int,
        default=5000,
        help="num_samples used in confidence (Clopper-Pearson) estimation.",
    )
    parser.add_argument(
        "--confidence-beta",
        type=float,
        default=1e-5,
        help="beta used in confidence bounds.",
    )
    parser.add_argument(
        "--error-num-samples",
        type=int,
        default=100,
        help="Number of Monte Carlo samples from the smoothed model used to compute average Euclidean error against GT.",
    )
    parser.add_argument(
        "--rmse-num-samples",
        dest="error_num_samples",
        type=int,
        help="Deprecated alias for --error-num-samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "data" / "analysis"),
        help="Directory where JSON/CSV files are written.",
    )
    parser.add_argument(
        "--max-settings",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the settings list across this many independent shards/jobs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to run (must be in [0, num_shards)).",
    )
    parser.add_argument(
        "--num-random-seeds",
        type=int,
        default=10,
        help="Number of random seeds to average over per batch (default: 10).",
    )

    args = parser.parse_args()

    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    stochastic_fn = TrajFlowMeszarosStochasticFunction()
    x = load_inputs(batch_size=args.batch_size)
    y_gt_final = load_ground_truth_outputs(batch_size=args.batch_size)
    if y_gt_final.shape[0] != x.shape[0]:
        raise ValueError(
            "Inputs.npy and Outputs_gt.npy have inconsistent number of samples "
            f"({x.shape[0]} vs {y_gt_final.shape[0]})"
        )

    if args.input_mode == "position":
        model_input = x
        helper: list[torch.Tensor] = []
    else:
        model_input, helper = stochastic_fn.pos_to_ctrl(x)

    settings = build_settings()
    if args.max_settings is not None:
        settings = settings[: args.max_settings]
    all_settings_count = len(settings)
    settings = [s for i, s in enumerate(settings) if i % args.num_shards == args.shard_index]

    print("=== TrajFlow Hyperparameter Analysis ===")
    print(f"batch_size: {x.shape[0]} (all)" if args.batch_size is None else f"batch_size: {args.batch_size}")
    print(f"input_mode: {args.input_mode}")
    print(f"num_settings_total: {all_settings_count}")
    print(f"num_settings_this_shard: {len(settings)}")
    print(f"shard: {args.shard_index}/{args.num_shards}")
    print(f"num_random_seeds: {args.num_random_seeds}")
    print(f"error_num_samples: {args.error_num_samples}")

    nx = x.shape[0]
    all_results: list[dict[str, Any]] = []
    for idx, setting in enumerate(settings, start=1):
        print(
            f"[{idx}/{len(settings)}] "
            f"alpha={setting.alpha}, N={setting.N}, "
            f"max_num_clusters={setting.max_num_clusters}, coverage={setting.coverage}, "
            f"sigma={setting.sigma}, r={setting.r}"
        )

        _METRICS = [
            "certified_radius",
            "minimum_prob",
            "maximum_prob",
            "total_certified_prob",
            "mean_l2_error_to_gt",
            "std_l2_error_to_gt",
            "average_area",
            "minimum_area",
            "maximum_area",
        ]

        xs_all: list[dict[str, Any]] = []
        for batch_start in range(0, nx, args.sample_batch_size):
            batch_end = min(batch_start + args.sample_batch_size, nx)
            batch_n = batch_end - batch_start
            seed_xs: list[list[dict[str, Any]]] = []
            for seed in range(args.num_random_seeds):
                print(
                    f"  points {batch_start + 1}-{batch_end}/{nx}  seed {seed + 1}/{args.num_random_seeds}",
                    end="\r",
                    flush=True,
                )
                result_i = evaluate_setting(
                    setting=setting,
                    stochastic_fn=stochastic_fn,
                    model_input=model_input[batch_start:batch_end],
                    helper=[h_i[batch_start:batch_end] for h_i in helper],
                    gt_output_final=y_gt_final[batch_start:batch_end],
                    error_num_samples=args.error_num_samples,
                    confidence_num_samples=args.confidence_num_samples,
                    confidence_beta=args.confidence_beta,
                    seed=seed,
                )
                seed_xs.append(result_i["xs"])
            for local_i in range(batch_n):
                aggregated: dict[str, Any] = {"x_index": batch_start + local_i}
                for metric in _METRICS:
                    vals = [seed_xs[s][local_i][metric] for s in range(args.num_random_seeds)]
                    aggregated[f"{metric}_mean"] = float(np.mean(vals))
                    aggregated[f"{metric}_std"] = float(np.std(vals, ddof=1))
                xs_all.append(aggregated)
        if nx > 1:
            print()  # newline after the \r progress

        all_results.append({"setting": asdict(setting), "xs": xs_all})

    output_dir = Path(args.output_dir)
    if args.num_shards == 1:
        json_path = output_dir / "results.json"
    else:
        json_path = output_dir / f"results_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"

    save_json(json_path, all_results)

    print("")
    print("Saved file:")
    print(json_path)


if __name__ == "__main__":
    main()
