from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
import torch


# Use LaTeX-like typography without requiring a full LaTeX installation.
plt.rcParams.update(
	{
		"font.family": "serif",
		"mathtext.fontset": "cm",
	}
)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
	return x.detach().cpu().numpy()


def plot_experiment_results(
	x: torch.Tensor,
	outputs: torch.Tensor,
	rekavandi_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
	region=None,
	plot_region_shading: bool = False,
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
	fig1, axes = plt.subplots(m, 1, figsize=(7, 4.5 * m), sharex=False)
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
			label="Samples from original model",
		)
		ax.hist(
			rekavandi_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["global_mc"],
			edgecolor=colors["global_mc"],
			linewidth=1.4,
			label=r"Samples from $\alpha$-smoothing (Rekavandi et al, 2024)",
		)
		ax.hist(
			clusterized_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["cluster_mc"],
			edgecolor=colors["cluster_mc"],
			linewidth=1.4,
			label=r"Samples from our clusterized $\alpha$-smoothing",
		)

		if plot_region_shading and region_lower_np is not None and region_upper_np is not None:
			k = region_lower_np.shape[1]
			region_color = "#7A7A7A"
			for j in range(k):
				left = region_lower_np[i, j]
				right = region_upper_np[i, j]
				ax.axvspan(left, right, color=region_color, alpha=0.14, lw=0)
				ax.axvline(left, color=region_color, linestyle="--", linewidth=1.0, alpha=0.9)
				ax.axvline(right, color=region_color, linestyle="--", linewidth=1.0, alpha=0.9)
				ax.plot([], [], color=region_color, linewidth=3, alpha=0.7)

		ax.set_xlabel("Output")
		ax.set_ylabel("Frequency")
		ax.grid(alpha=0.25, linestyle=":")
		ax.legend(loc="upper left", fontsize=9)

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


def _plot_street_marks(ax: plt.Axes, x_extent: float, y_extent: float) -> None:
	"""Overlay static street boundaries and lane center lines."""
	x_extent = max(float(x_extent), 12.0)
	y_extent = max(float(y_extent), 12.0)

	lines_solid = []
	lines_solid.append(np.array([[-x_extent, -4], [-4, -4], [-4, -y_extent]]))
	lines_solid.append(np.array([[x_extent, -4], [4, -4], [4, -y_extent]]))
	lines_solid.append(np.array([[-x_extent, 4], [-4, 4], [-4, y_extent]]))
	lines_solid.append(np.array([[x_extent, 4], [4, 4], [4, y_extent]]))

	lines_dashed = []
	lines_dashed.append(np.array([[0, 6], [0, y_extent]]))
	lines_dashed.append(np.array([[0, -6], [0, -y_extent]]))
	lines_dashed.append(np.array([[6, 0], [x_extent, 0]]))
	lines_dashed.append(np.array([[-6, 0], [-x_extent, 0]]))

	for line in lines_solid:
		ax.plot(
			line[:, 0],
			line[:, 1],
			color="#4C4C4C",
			linewidth=1.6,
			solid_capstyle="round",
			zorder=0,
		)

	for line in lines_dashed:
		ax.plot(
			line[:, 0],
			line[:, 1],
			color="#6B6B6B",
			linewidth=1.2,
			linestyle=(0, (5, 5)),
			solid_capstyle="round",
			zorder=0,
		)


def _plot_car_icon(
	ax: plt.Axes,
	start: np.ndarray,
	next_point: np.ndarray,
	color: str,
	car_length: float,
) -> None:
	"""Draw a simple rotated rectangle to indicate vehicle pose at trajectory start."""
	heading = float(np.arctan2(next_point[1] - start[1], next_point[0] - start[0]))
	car_width = car_length * 0.55
	rect = mpatches.Rectangle(
		(start[0] - 0.5 * car_length, start[1] - 0.5 * car_width),
		car_length,
		car_width,
		facecolor=color,
		edgecolor="#FFFFFF",
		linewidth=0.6,
		zorder=7,
	)
	rect.set_transform(mtransforms.Affine2D().rotate_around(start[0], start[1], heading) + ax.transData)
	ax.add_patch(rect)


def plot_trajectory_samples(
	x: torch.Tensor,
	mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot trajectory samples for each x point.

	Args:
		x: Tensor (nx, num_input_agents * num_input_timesteps * 2) containing
		   flattened input trajectories used for dashed overlays.
		mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		clusterized_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		output_dir: Folder to save charts.
		num_input_agents: Number of agents encoded in x.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = _reshape_trajectory_samples(mc_samples)
	clusterized = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)
	if num_input_agents < 1:
		raise ValueError("num_input_agents must be >= 1")
	if x_np.ndim != 2 or x_np.shape[1] % (2 * num_input_agents) != 0:
		raise ValueError(
			"x must have shape (nx, num_input_agents * num_input_timesteps * 2)"
		)
	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	if raw.shape != clusterized.shape:
		raise ValueError("mc_samples and clusterized_mc_samples must have identical shapes")
	if raw.shape[0] != x_traj.shape[0]:
		raise ValueError("x and trajectory samples must have the same first dimension")

	nx, num_samples, _, _ = raw.shape
	fig, axes = plt.subplots(nx, 2, figsize=(12, 3.2 * nx), sharex=True, sharey=True)
	if nx == 1:
		axes = np.array([axes])

	all_x = np.concatenate(
		(
			raw[:, :, :, 0].reshape(-1),
			clusterized[:, :, :, 0].reshape(-1),
			x_traj[:, :, :, 0].reshape(-1),
		)
	)
	all_y = np.concatenate(
		(
			raw[:, :, :, 1].reshape(-1),
			clusterized[:, :, :, 1].reshape(-1),
			x_traj[:, :, :, 1].reshape(-1),
		)
	)
	x_min, x_max = np.min(all_x), np.max(all_x)
	y_min, y_max = np.min(all_y), np.max(all_y)
	x_pad = max((x_max - x_min) * 0.05, 1e-6)
	y_pad = max((y_max - y_min) * 0.05, 1e-6)
	x_lim = (x_min - x_pad, x_max + x_pad)
	y_lim = (y_min - y_pad, y_max + y_pad)

	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))
	car_size = 0.018 * max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
	car_size = max(car_size, 0.2)

	for i in range(nx):
		ax_raw = axes[i, 0]
		ax_cluster = axes[i, 1]

		_plot_street_marks(ax_raw, x_extent=x_extent, y_extent=y_extent)
		_plot_street_marks(ax_cluster, x_extent=x_extent, y_extent=y_extent)

		for j in range(num_samples):
			traj = raw[i, j]
			ax_raw.plot(
				traj[:, 0],
				traj[:, 1],
				color="#1B5E20",
				alpha=0.16,
				linewidth=0.9,
				zorder=1,
				label="Samples from original model" if j == 0 else None,
			)

			traj_cluster = clusterized[i, j]
			ax_cluster.plot(
				traj_cluster[:, 0],
				traj_cluster[:, 1],
				color="#66BB6A",
				alpha=0.35,
				linewidth=1.0,
				zorder=1,
				label="Samples from smoothed model" if j == 0 else None,
			)

		agent_colors = ("#1B5E20", "#0B3C5D")
		for agent_idx in range(num_input_agents):
			agent_traj = x_traj[i, agent_idx]
			color = agent_colors[agent_idx % len(agent_colors)]
			label = "Input (Target agent)" if agent_idx == 0 else "Input (Ego agent)"
			ax_raw.plot(
				agent_traj[:, 0],
				agent_traj[:, 1],
				color=color,
				linestyle="None",
				marker="o",
				markersize=2.1,
				markerfacecolor=color,
				markeredgecolor=color,
				markeredgewidth=0.2,
				zorder=6,
				label=label,
			)
			ax_cluster.plot(
				agent_traj[:, 0],
				agent_traj[:, 1],
				color=color,
				linestyle="None",
				marker="o",
				markersize=2.0,
				markerfacecolor=color,
				markeredgecolor=color,
				markeredgewidth=0.2,
				zorder=5,
			)

			next_idx = 1 if agent_traj.shape[0] > 1 else 0
			_plot_car_icon(
				ax_raw,
				start=agent_traj[0],
				next_point=agent_traj[next_idx],
				color=color,
				car_length=car_size,
			)
			_plot_car_icon(
				ax_cluster,
				start=agent_traj[0],
				next_point=agent_traj[next_idx],
				color=color,
				car_length=car_size,
			)

		for ax in (ax_raw, ax_cluster):
			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_xlabel("x")
			ax.grid(alpha=0.25, linestyle=":")

		ax_raw.set_ylabel("y")
		ax_cluster.set_ylabel("")
		ax_cluster.tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			ax_raw.legend(loc="upper left", fontsize=8)
			ax_cluster.legend(loc="upper left", fontsize=8)

	fig.tight_layout()
	fig_path = output_dir / "trajectory_samples_comparison.png"
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)

	return [fig_path]

