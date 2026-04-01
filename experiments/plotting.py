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
	region=None,
) -> list[Path]:
	"""Create and save charts for the smoothing experiment."""
	output_dir.mkdir(parents=True, exist_ok=True)

	x_np = _to_numpy(x[:, 0])
	outputs_np = _to_numpy(outputs[:, :, 0])
	global_mc_np = _to_numpy(global_trimmed_mc_samples[:, :, 0])
	clusterized_mc_np = _to_numpy(clusterized_mc_samples[:, :, 0])
	region_lower_np = _to_numpy(region.lower[:, :, 0]) if region is not None else None
	region_upper_np = _to_numpy(region.upper[:, :, 0]) if region is not None else None

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

		if region_lower_np is not None and region_upper_np is not None:
			k = region_lower_np.shape[1]
			region_color = "#7A7A7A"
			for j in range(k):
				left = region_lower_np[i, j]
				right = region_upper_np[i, j]
				ax.axvspan(left, right, color=region_color, alpha=0.14, lw=0)
				ax.axvline(left, color=region_color, linestyle="--", linewidth=1.0, alpha=0.9)
				ax.axvline(right, color=region_color, linestyle="--", linewidth=1.0, alpha=0.9)
				ax.plot([], [], color=region_color, linewidth=3, alpha=0.7)

		ax.set_title(f"Input x={x_np[i]:.3f}")
		ax.set_xlabel("Output y")
		ax.set_ylabel("Density")
		ax.grid(alpha=0.25, linestyle=":")
		ax.legend(loc="upper left", fontsize=9)

	fig1.suptitle("Single-Run vs Monte Carlo Smoothing Distributions with Cluster Regions", fontsize=13)
	fig1.tight_layout()
	fig1_path = output_dir / "distribution_comparison.png"
	fig1.savefig(fig1_path, dpi=180)
	plt.close(fig1)

	return [fig1_path]

