from __future__ import annotations

import argparse
import csv
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
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.dataclasses import (  # noqa: E402
    ClusteringParameters,
    ConfidenceParameters,
    Parameters,
    SmoothingParameters,
)
from randomized_smoothing.smoothing import GaussianAlphaTrimmingSmoothing  # noqa: E402
from stochastic_functions import TrajFlowMeszarosStochasticFunction  # noqa: E402


@dataclass
class Setting:
    alpha: float
    N: int
    max_num_clusters: int
    coverage: float
    sigma: float
    r: float


def load_inputs(batch_size: int) -> torch.Tensor:
    """Load TrajFlow inputs and return shape (batch_size, 40)."""
    data_file = ROOT / "TrajFlow_exp" / "Data" / "Inputs.npy"
    data = np.load(str(data_file))

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > len(data):
        raise ValueError(f"batch_size={batch_size} exceeds available samples={len(data)}")

    x_input = data[:batch_size, :, :, :2]
    x_input = x_input[:, [1, 0]]  # Put the target agent in index 0.
    return torch.from_numpy(x_input.reshape(batch_size, -1)).float()


def build_settings() -> list[Setting]:
    alphas = [0.0, 0.2, 0.4]
    Ns = [10, 30]
    max_clusters = [1, 3, 10]
    coverages = [0.9, 0.99]
    sigmas = [0.1, 0.5]

    settings: list[Setting] = []
    for alpha, N, max_num_clusters, coverage, sigma in itertools.product(
        alphas,
        Ns,
        max_clusters,
        coverages,
        sigmas,
    ):
        for denom in (10, 50):
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

    total_certified_prob = certificate.probs.sum(dim=-1)
    regions = smoother.regions
    total_volume_per_x = regions.volume.sum(dim=-1)
    total_volume_all_x = regions.total_volume

    return {
        "setting": asdict(setting),
        "certified_radius": certificate.radius.squeeze(-1).detach().cpu().tolist(),
        "certified_probs": certificate.probs.detach().cpu().tolist(),
        "total_certified_prob": total_certified_prob.detach().cpu().tolist(),
        "total_volume_per_x": total_volume_per_x.detach().cpu().tolist(),
        "total_volume_all_x": float(total_volume_all_x.detach().cpu().item()),
    }


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert setting-level results to one row per (setting, x_index) for tables."""
    rows: list[dict[str, Any]] = []

    for setting_id, result in enumerate(results):
        setting = result["setting"]
        certified_radius = result["certified_radius"]
        certified_probs = result["certified_probs"]
        total_certified_prob = result["total_certified_prob"]
        total_volume_per_x = result["total_volume_per_x"]
        total_volume_all_x = result["total_volume_all_x"]

        for x_index in range(len(certified_radius)):
            total_volume = float(total_volume_per_x[x_index])
            row = {
                "setting_id": setting_id,
                **setting,
                "x_index": x_index,
                "certified_radius": float(certified_radius[x_index]),
                "total_certified_prob": float(total_certified_prob[x_index]),
                "total_volume": total_volume,
                "total_volume_all_x": float(total_volume_all_x),
                "total_volume_is_finite": math.isfinite(total_volume),
                # Keep cluster-wise probabilities in a JSON-compatible cell.
                "certified_probs": json.dumps(certified_probs[x_index]),
            }
            rows.append(row)

    return rows


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


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to save")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        default=str(ROOT / "outputs" / "analysis"),
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

    rows = flatten_results(all_results)

    output_dir = Path(args.output_dir)
    json_path = output_dir / "results_by_setting.json"
    csv_path = output_dir / "results_long_table.csv"

    save_json(json_path, all_results)
    save_csv(csv_path, rows)

    print("")
    print("Saved files:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
