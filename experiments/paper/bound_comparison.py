from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

TEXTWIDTH = 5.5  # inches

plt.rcParams.update(
	{
		"font.family": "serif",
		"mathtext.fontset": "cm",
		"font.size": 10,
		"axes.labelsize": 10,
		"axes.titlesize": 10,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
		"legend.fontsize": 9,
		"legend.framealpha": 0.95,
	}
)

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
SRC = EXPERIMENTS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.utils import robustness_bound


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
    _w = TEXTWIDTH  # full textwidth for the grid
    fig2, axes2 = plt.subplots(len(alphas), len(p_v_values), figsize=(_w, 0.60 * _w), sharex=True, sharey=True)

    for i, alpha in enumerate(alphas):
        for j, p_v in enumerate(p_v_values):
            ax = axes2[i, j]
            for n in num_samples_list:
                y = compute_curve_ours_per_region(
                    ratio_values=ratio_values,
                    p_v=p_v,
                    num_samples=n,
                    alpha=alpha,
                )
                ax.plot(
                    ratio_values.numpy(),
                    y.numpy(),
                    linewidth=2.0,
                    color=colors[n],
                    label=rf"$N={n}$",
                )

            ax.set_title(rf"$\alpha={alpha},\ p_{{\mathcal{{V}}}}={p_v}$")
            ax.grid(alpha=0.3, linestyle=":")
            ax.set_yticks([0.0, 0.5, 1.0])

    # Single shared axis labels.
    fig2.supylabel("Certified probability", fontsize=plt.rcParams["axes.labelsize"])

    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.22)  # room for x-label + legend

    # Place shared x-label and legend at explicit vertical positions so they
    # don't overlap.  Both use figure-fraction coordinates.
    fig2.text(0.5, 0.12, r"Ratio $p_{\mathcal{R}} / p_{\mathcal{V}}$",
              ha="center", va="center", fontsize=plt.rcParams["axes.labelsize"])

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="lower center", ncol=len(num_samples_list),
                bbox_to_anchor=(0.5, 0.0))

    output_path_ours = output_dir / "bound_clusterized_grid.pdf"
    fig2.savefig(output_path_ours, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved plot to: {output_path_ours}")


if __name__ == "__main__":
    main()
