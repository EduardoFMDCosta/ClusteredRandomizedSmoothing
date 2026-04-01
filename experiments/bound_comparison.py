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

from randomized_smoothing.utils import _bound_rekavandi_et_al_2024


def compute_curve(p_values: torch.Tensor, num_samples: int, alpha: float) -> torch.Tensor:
    values = [
        float(_bound_rekavandi_et_al_2024(p=pi, num_samples=num_samples, alpha=alpha).item())
        for pi in p_values
    ]
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


if __name__ == "__main__":
    main()
