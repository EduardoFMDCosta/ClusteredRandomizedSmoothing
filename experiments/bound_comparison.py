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

from randomized_smoothing.utils import _bound_ours_per_region, _bound_rekavandi_et_al_2024


def compute_curve(p_values: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    values = [
        float(_bound_rekavandi_et_al_2024(p=pi, num_samples=num_samples, alpha=alpha).item())
        for pi in p_values
    ]
    return torch.tensor(values, dtype=torch.float64)


def compute_curve_ours_per_region(
    p0_values: torch.Tensor,
    p1: float,
    num_samples: int,
    alpha: float,
) -> torch.Tensor:
    values = []
    for p0 in p0_values:
        # Build p as a (2, 1) tensor and flatten for the utility function API.
        p_pair = torch.tensor([[float(p0.item())], [p1]], dtype=torch.float64)
        bound = _bound_ours_per_region(p=p_pair.reshape(-1), num_samples=num_samples, alpha=alpha)
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

    # Additional test: _bound_ours_per_region with p = [p0, p1]
    p1_values = [0.2, 0.5, 0.8]
    p0_values = p_values
    fig2, axes2 = plt.subplots(len(alphas), len(p1_values), figsize=(18, 12), sharex=True, sharey=True)

    for i, alpha in enumerate(alphas):
        for j, p1 in enumerate(p1_values):
            ax = axes2[i, j]
            for n in num_samples_list:
                y = compute_curve_ours_per_region(p0_values=p0_values, p1=p1, num_samples=n, alpha=alpha)
                ax.plot(
                    p0_values.numpy(),
                    y.numpy(),
                    linewidth=2.0,
                    color=colors[n],
                    label=rf"$N={n}$",
                )

            ax.set_title(rf"$\alpha={alpha},\ p_1={p1}$")
            ax.grid(alpha=0.3, linestyle=":")
            if i == len(alphas) - 1:
                ax.set_xlabel(r"$p_0$")
            if j == 0:
                ax.set_ylabel("Certified probability")

    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper center", ncol=3)
    fig2.suptitle(r"Ours per-region bound across $p_0$, $p_1$, $\alpha$, and sample size $N$", fontsize=13)
    fig2.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path_ours = output_dir / "bound_ours_per_region_grid.png"
    fig2.savefig(output_path_ours, dpi=180)
    plt.close(fig2)

    print(f"Saved plot to: {output_path_ours}")


if __name__ == "__main__":
    main()
