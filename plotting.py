from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x: torch.Tensor) -> np.ndarray:
	return x.detach().cpu().numpy()


def plot_experiment_results(
	x: torch.Tensor,
	outputs: torch.Tensor,
	global_trimmed_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
) -> list[Path]:
	"""Create and save charts for the smoothing experiment."""
	output_dir.mkdir(parents=True, exist_ok=True)

	x_np = _to_numpy(x[:, 0])
	outputs_np = _to_numpy(outputs[:, :, 0])
	global_mc_np = _to_numpy(global_trimmed_mc_samples[:, :, 0])
	clusterized_mc_np = _to_numpy(clusterized_mc_samples[:, :, 0])

	m = outputs_np.shape[0]
	colors = {
		"raw": "#111111",
		"global_mc": "#E31A1C",
		"cluster_mc": "#1F78B4",
	}

	# Figure 1: per-input distribution comparison.
	fig1, axes = plt.subplots(m, 1, figsize=(10, 3.4 * m), sharex=False)
	if m == 1:
		axes = [axes]

	for i, ax in enumerate(axes):
		ax.hist(
			outputs_np[i],
			bins=60,
			density=True,
			histtype="step",
			linewidth=2.2,
			color=colors["raw"],
			label="Raw stochastic outputs",
		)
		ax.hist(
			global_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["global_mc"],
			edgecolor=colors["global_mc"],
			linewidth=1.4,
			label="MC global trimming samples",
		)
		ax.hist(
			clusterized_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["cluster_mc"],
			edgecolor=colors["cluster_mc"],
			linewidth=1.4,
			label="MC clusterized trimming samples",
		)

		ax.set_title(f"Input x={x_np[i]:.3f}")
		ax.set_xlabel("Output y")
		ax.set_ylabel("Density")
		ax.grid(alpha=0.25, linestyle=":")
		ax.legend(loc="upper left", fontsize=9)

	fig1.suptitle("Single-Run vs Monte Carlo Smoothing Distributions", fontsize=13)
	fig1.tight_layout()
	fig1_path = output_dir / "distribution_comparison.png"
	fig1.savefig(fig1_path, dpi=180)
	plt.close(fig1)

	return [fig1_path]

