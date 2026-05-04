from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plotting import plot_bound_clustered_grid

from randomized_smoothing.optimization import robustness_bound


def compute_curve(p_values: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    # Rekavandi setting: single cluster with p_partition = 1.
    values = []
    for pi in p_values:
        probs_region = torch.tensor([[float(pi.item())]], dtype=torch.float64)
        probs_partition_lb = torch.ones((1, 1), dtype=torch.float64)
        probs_partition_ub = torch.ones((1, 1), dtype=torch.float64)
        bound = robustness_bound(
            probs_region=probs_region,
            probs_partition_lb=probs_partition_lb,
            probs_partition_ub=probs_partition_ub,
            num_samples=num_samples,
            alpha=alpha,
        )
        values.append(float(bound.squeeze().item()))
    return torch.tensor(values, dtype=torch.float64)


def compute_curve_ours_per_region(
    ratio_values: torch.Tensor,
    p_v: float,
    num_samples: int,
    alpha: float,
) -> torch.Tensor:
    values = []
    for ratio in ratio_values:
        probs_partition_lb = torch.tensor([[p_v]], dtype=torch.float64)
        probs_partition_ub = torch.tensor([[p_v]], dtype=torch.float64)
        probs_region = ratio * probs_partition_ub
        bound = robustness_bound(
            probs_region=probs_region,
            probs_partition_lb=probs_partition_lb,
            probs_partition_ub=probs_partition_ub,
            num_samples=num_samples,
            alpha=alpha,
        )
        values.append(float(bound.squeeze().item()))
    return torch.tensor(values, dtype=torch.float64)


def main() -> None:
    output_dir = EXPERIMENTS / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.0, 0.2, 0.4]
    num_samples_list = [10, 25, 50]

    # Professional scientific paper colors
    colors = {
        10: "#0173B2",  # Blue
        25: "#DE8F05",  # Orange
        50: "#CC78BC",  # Purple
    }

    # Per-region bound versus ratio p_R / p_V.
    p_v_values = [0.2, 0.5, 0.8, 1.0]
    ratio_values = torch.linspace(0.0, 1.0, 300, dtype=torch.float64)

    y_data: dict[tuple[float, float, int], list[float]] = {}
    for alpha in alphas:
        for p_v in p_v_values:
            for n in num_samples_list:
                y = compute_curve_ours_per_region(
                    ratio_values=ratio_values,
                    p_v=p_v,
                    num_samples=n,
                    alpha=alpha,
                )
                y_data[(alpha, p_v, n)] = y.numpy()

    output_path_ours = plot_bound_clustered_grid(
        ratio_values=ratio_values.numpy(),
        alphas=alphas,
        p_v_values=p_v_values,
        num_samples_list=num_samples_list,
        y_data=y_data,
        colors=colors,
        output_dir=output_dir,
    )

    print(f"Saved plot to: {output_path_ours}")


if __name__ == "__main__":
    main()
