from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Make src/ importable when running this script from the repository root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from randomized_smoothing.utils import robustness_bound


def compute_curve(p_values: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    # Rekavandi setting: single cluster with p_partition = 1.
    values = []
    for pi in p_values:
        probs_region = torch.tensor([[float(pi.item())]], dtype=torch.float64)
        probs_partition = torch.ones((1, 1), dtype=torch.float64)
        bound = robustness_bound(
            probs_region=probs_region,
            probs_partition=probs_partition,
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
        probs_partition = torch.tensor([[p_v]], dtype=torch.float64)
        probs_region = ratio * probs_partition
        bound = robustness_bound(
            probs_region=probs_region,
            probs_partition=probs_partition,
            num_samples=num_samples,
            alpha=alpha,
        )
        values.append(float(bound.squeeze().item()))
    return torch.tensor(values, dtype=torch.float64)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [0.0, 0.2, 0.4]
    num_samples_list = [10, 25, 50]

    p_values = torch.linspace(0.01, 0.99, 300, dtype=torch.float64)

    fig, axes = plt.subplots(1, len(alphas), figsize=(18, 5), sharey=True)

    colors = {
        10: "#1f77b4",
        25: "#ff7f0e",
        50: "#2ca02c",
    }

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        for n in num_samples_list:
            y = compute_curve(p_values=p_values, num_samples=n, alpha=alpha)
            ax.plot(p_values.numpy(), y.numpy(), linewidth=2.0, color=colors[n], label=f"N={n}")

        ax.set_title(rf"$\alpha={alpha}$")
        ax.set_xlabel("p")
        ax.grid(alpha=0.3, linestyle=":")

    axes[0].set_ylabel("Certified probability")
    axes[-1].legend(loc="best")

    fig.suptitle(r"Rekavandi et al. (2024) bound across $p$, $\alpha$, and sample size $N$", fontsize=13)
    fig.tight_layout()

    output_path = output_dir / "bound_rekavandi_grid.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot to: {output_path}")

    # Additional test: per-region bound versus ratio p_R / p_V.
    p_v_values = [0.2, 0.5, 0.8]
    ratio_values = torch.linspace(0.0, 1.0, 300, dtype=torch.float64)
    fig2, axes2 = plt.subplots(len(alphas), len(p_v_values), figsize=(18, 12), sharex=True, sharey=True)

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

            ax.set_title(rf"$\alpha={alpha},\ p_{{V}}={p_v}$")
            ax.grid(alpha=0.3, linestyle=":")
            if i == len(alphas) - 1:
                ax.set_xlabel(r"$p_{R} / p_{V}$")
            if j == 0:
                ax.set_ylabel("Certified probability")

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper center", ncol=3)
    fig2.suptitle(r"Ours per-region bound across $p_{R} / p_{V}$, $p_{V}$, $\alpha$, and sample size $N$", fontsize=13)
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path_ours = output_dir / "bound_ours_per_region_grid.png"
    fig2.savefig(output_path_ours, dpi=180)
    plt.close(fig2)

    print(f"Saved plot to: {output_path_ours}")


if __name__ == "__main__":
    main()
