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
	rekavandi_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
	region=None,
) -> list[Path]:
	"""Create and save charts for the smoothing experiment."""
	output_dir.mkdir(parents=True, exist_ok=True)

	x_np = _to_numpy(x[:, 0])
	outputs_np = _to_numpy(outputs[:, :, 0])
	rekavandi_mc_np = _to_numpy(rekavandi_mc_samples[:, :, 0])
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
			rekavandi_mc_np[i],
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


def _reshape_trajectory_samples(samples: torch.Tensor) -> np.ndarray:
	arr = _to_numpy(samples)
	if arr.ndim != 3:
		raise ValueError("samples must have shape (nx, num_samples, 2 * num_timesteps)")

	nx, num_samples, flat_dim = arr.shape
	if flat_dim % 2 != 0:
		raise ValueError(
			f"Expected an even last dimension encoding [x1, y1, x2, y2, ...], got {flat_dim}."
		)
	num_timesteps = flat_dim // 2

	return arr.reshape(nx, num_samples, num_timesteps, 2)


def plot_trajectory_samples(
	x: torch.Tensor,
	mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
) -> list[Path]:
	"""Plot trajectory samples for each x point.

	Args:
		x: Tensor (nx, 2 * num_input_timesteps) for dashed input trajectory overlay.
		mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		clusterized_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		output_dir: Folder to save charts.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = _reshape_trajectory_samples(mc_samples)
	clusterized = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)
	if x_np.ndim != 2 or x_np.shape[1] % 2 != 0:
		raise ValueError("x must have shape (nx, 2 * num_input_timesteps)")
	x_traj = x_np.reshape(x_np.shape[0], x_np.shape[1] // 2, 2)

	if raw.shape != clusterized.shape:
		raise ValueError("mc_samples and clusterized_mc_samples must have identical shapes")
	if raw.shape[0] != x_traj.shape[0]:
		raise ValueError("x and trajectory samples must have the same first dimension")

	nx, num_samples, _, _ = raw.shape
	fig, axes = plt.subplots(nx, 2, figsize=(12, 3.2 * nx), sharex="row", sharey="row")
	if nx == 1:
		axes = np.array([axes])

	for i in range(nx):
		ax_raw = axes[i, 0]
		ax_cluster = axes[i, 1]

		for j in range(num_samples):
			traj = raw[i, j]
			ax_raw.plot(traj[:, 0], traj[:, 1], color="#222222", alpha=0.12, linewidth=0.8)

			traj_cluster = clusterized[i, j]
			ax_cluster.plot(traj_cluster[:, 0], traj_cluster[:, 1], color="#1F78B4", alpha=0.12, linewidth=0.8)

		xi = x_traj[i]
		ax_raw.plot(xi[:, 0], xi[:, 1], color="#FF7F00", linestyle="--", linewidth=1.8)
		ax_cluster.plot(xi[:, 0], xi[:, 1], color="#FF7F00", linestyle="--", linewidth=1.8)

		all_x = np.concatenate((raw[i, :, :, 0].reshape(-1), clusterized[i, :, :, 0].reshape(-1), xi[:, 0]))
		all_y = np.concatenate((raw[i, :, :, 1].reshape(-1), clusterized[i, :, :, 1].reshape(-1), xi[:, 1]))
		x_min, x_max = np.min(all_x), np.max(all_x)
		y_min, y_max = np.min(all_y), np.max(all_y)
		x_pad = max((x_max - x_min) * 0.05, 1e-6)
		y_pad = max((y_max - y_min) * 0.05, 1e-6)
		for ax in (ax_raw, ax_cluster):
			ax.set_xlim(x_min - x_pad, x_max + x_pad)
			ax.set_ylim(y_min - y_pad, y_max + y_pad)

		ax_raw.set_title(f"x[{i}] raw MC trajectories")
		ax_cluster.set_title(f"x[{i}] clusterized MC trajectories")

		for ax in (ax_raw, ax_cluster):
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.grid(alpha=0.25, linestyle=":")

	fig.suptitle("Trajectory Samples from Stochastic and Clusterized Smoothing", fontsize=13)
	fig.tight_layout()
	fig_path = output_dir / "trajectory_samples_comparison.png"
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)

	return [fig_path]

