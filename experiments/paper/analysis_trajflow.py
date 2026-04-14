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


def load_inputs(batch_size: int) -> torch.Tensor:
    data_file = EXPERIMENTS / "TrajFlowExperiment" / "Data" / "Inputs.npy"
    data = np.load(str(data_file))

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(data):
        raise ValueError(f"batch_size={batch_size} exceeds available samples={len(data)}")

    x_input = data[:batch_size, :, :, :2]
    x_input = x_input[:, [1, 0]]  # Put the target agent in index 0.
    return torch.from_numpy(x_input.reshape(batch_size, -1)).float()


def build_settings() -> list[Setting]:
    alphas = [0.0]
    Ns = [10]
    max_clusters = [1]
    coverages = [0.9]
    sigmas = [0.1]

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
    x: torch.Tensor,
    confidence_num_samples: int,
    confidence_beta: float,
) -> dict[str, Any]:
    """Run one smoothing/certification setting and return serializable metrics."""
    torch.manual_seed(0)

    stochastic_fn = TrajFlowMeszarosStochasticFunction()
    params = Parameters(
        smoothing=SmoothingParameters(N=setting.N, alpha=setting.alpha),
        clustering=ClusteringParameters(
            max_num_clusters=setting.max_num_clusters,
            coverage=setting.coverage,
            set_type="TimeStepHyperrectangles",
        ),
        confidence=ConfidenceParameters(
            num_samples=confidence_num_samples,
            beta=confidence_beta,
        ),
    )

    smoother = GaussianAlphaTrimmingSmoothing(
        stochastic_function=stochastic_fn,
        x=x,
        sigma=setting.sigma,
        parameters=params,
    )

    radius = torch.full((x.shape[0], 1), fill_value=setting.r, dtype=x.dtype, device=x.device)
    certificate = smoother.robust_certification(radius=radius)

    certified_radius = certificate.radius.squeeze(-1).detach().cpu()    # (nx,)
    certified_probs = certificate.probs.detach().cpu()                   # (nx, num_clusters)
    total_certified_prob = certificate.probs.sum(dim=-1).detach().cpu()  # (nx,)

    # The output layout is assumed to be [x1, y1, x2, y2, ..., xT, yT].
    # Compute area per timestep from even/odd coordinate widths, sum over
    # clusters, then aggregate over timesteps.
    widths = smoother.regions.width.to(torch.float64)                           # (nx, k, 2T)
    if widths.shape[-1] % 2 != 0:
        raise ValueError("Expected even output dimension arranged as [x1,y1,...,xT,yT]")

    x_widths = widths[..., 0::2]                                                # (nx, k, T)
    y_widths = widths[..., 1::2]                                                # (nx, k, T)
    areas_per_cluster = x_widths * y_widths                                     # (nx, k, T)
    areas_per_timestep = areas_per_cluster.sum(dim=1).detach().cpu()           # (nx, 20)
    average_area = areas_per_timestep.mean(dim=-1)                             # (nx,)
    minimum_area = areas_per_timestep.min(dim=-1).values                       # (nx,)
    maximum_area = areas_per_timestep.max(dim=-1).values                       # (nx,)

    nx = x.shape[0]
    xs = [
        {
            "x_index": i,
            "certified_radius": float(certified_radius[i].item()),
            "minimum_prob": float(certified_probs[i].min().item()),
            "maximum_prob": float(certified_probs[i].max().item()),
            "total_certified_prob": float(total_certified_prob[i].item()),
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
    parser.add_argument("--batch-size", type=int, default=4, help="How many input trajectories to evaluate.")
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

    args = parser.parse_args()

    x = load_inputs(batch_size=args.batch_size)
    settings = build_settings()
    if args.max_settings is not None:
        settings = settings[: args.max_settings]

    print("=== TrajFlow Hyperparameter Analysis ===")
    print(f"batch_size: {args.batch_size}")
    print(f"num_settings: {len(settings)}")

    all_results: list[dict[str, Any]] = []
    for idx, setting in enumerate(settings, start=1):
        print(
            f"[{idx}/{len(settings)}] "
            f"alpha={setting.alpha}, N={setting.N}, "
            f"max_num_clusters={setting.max_num_clusters}, coverage={setting.coverage}, "
            f"sigma={setting.sigma}, r={setting.r}"
        )

        result = evaluate_setting(
            setting=setting,
            x=x,
            confidence_num_samples=args.confidence_num_samples,
            confidence_beta=args.confidence_beta,
        )
        all_results.append(result)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "results.json"

    save_json(json_path, all_results)

    print("")
    print("Saved file:")
    print(json_path)


if __name__ == "__main__":
    main()
