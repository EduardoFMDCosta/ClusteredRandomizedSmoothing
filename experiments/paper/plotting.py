from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.transforms as mtransforms
import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from randomized_smoothing.clusters import Cluster


TEXTWIDTH = 5.5  # inches
HALFWIDTH = 0.50 * TEXTWIDTH  # for \begin{minipage}{0.50\textwidth}

plt.rcParams.update(
	{
		"font.family": "serif",
		"mathtext.fontset": "cm",
		"font.size": 10,
		"axes.labelsize": 10,
		"axes.titlesize": 10,
		"xtick.labelsize": 7,
		"ytick.labelsize": 7,
		"legend.fontsize": 9,
	}
)

CLUSTER_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


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
	_w = 0.80 * TEXTWIDTH
	fig1, axes = plt.subplots(m, 1, figsize=(_w, 0.8 * _w * m), sharex=False)
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
			label="Original model",
		)
		ax.hist(
			rekavandi_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["global_mc"],
			edgecolor=colors["global_mc"],
			linewidth=1.4,
			label=r"$\alpha$-smoothing (Rekavandi et al, 2024)",
		)
		ax.hist(
			clusterized_mc_np[i],
			bins=45,
			density=True,
			alpha=0.30,
			color=colors["cluster_mc"],
			edgecolor=colors["cluster_mc"],
			linewidth=1.4,
			label=r"Clusterized $\alpha$-smoothing",
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
		ax.legend(loc="upper left")

	fig1.tight_layout()
	fig1_path = output_dir / "distribution_comparison.pdf"
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


def plot_final_timestep_density(
	x: torch.Tensor,
	original_mc_samples: torch.Tensor,
	noisy_mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	single_cluster_centroids: torch.Tensor,
	clusterized_centroids: torch.Tensor,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot final-step endpoint samples colored by smoother-induced clusters.

	Columns are Original / Noisy / single-cluster smoother / clusterized smoother.
	Samples are colored by nearest smoother centroids (Voronoi attribution).

	Args:
		x: Tensor (nx, num_input_agents * num_input_timesteps * 2) – input
		   trajectory context, used only for spatial reference markers.
		original_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		noisy_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		single_cluster_mc_samples: same shape, single-cluster smoother.
		clusterized_mc_samples: same shape, multi-cluster smoother.
		single_cluster_centroids: Tensor (nx, k1, 2) for single-cluster smoother.
		clusterized_centroids: Tensor (nx, k2, 2) for multi-cluster smoother.
		output_dir: Directory to save the chart.
		num_input_agents: Number of agents encoded in x.
	"""

	output_dir.mkdir(parents=True, exist_ok=True)

	original = _reshape_trajectory_samples(original_mc_samples)
	noisy = _reshape_trajectory_samples(noisy_mc_samples)
	single = _reshape_trajectory_samples(single_cluster_mc_samples)
	clust = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)
	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	# Last-timestep endpoints: (nx, N, 2)
	original_end = original[:, :, -1, :]
	noisy_end = noisy[:, :, -1, :]
	single_end = single[:, :, -1, :]
	clust_end = clust[:, :, -1, :]

	nx = original.shape[0]

	# Shared axis limits derived from all endpoints and input trajectories.
	all_x_coords = np.concatenate([
		original_end[..., 0].reshape(-1),
		noisy_end[..., 0].reshape(-1),
		single_end[..., 0].reshape(-1),
		clust_end[..., 0].reshape(-1),
		x_traj[..., 0].reshape(-1),
	])
	all_y_coords = np.concatenate([
		original_end[..., 1].reshape(-1),
		noisy_end[..., 1].reshape(-1),
		single_end[..., 1].reshape(-1),
		clust_end[..., 1].reshape(-1),
		x_traj[..., 1].reshape(-1),
	])
	x_pad = max((float(all_x_coords.max()) - float(all_x_coords.min())) * 0.12, 0.5)
	y_pad = max((float(all_y_coords.max()) - float(all_y_coords.min())) * 0.12, 0.5)
	x_lim = (float(all_x_coords.min()) - x_pad, float(all_x_coords.max()) + x_pad)
	y_lim = (float(all_y_coords.min()) - y_pad, float(all_y_coords.max()) + y_pad)
	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))

	col_titles = (r"$h_{w}(x)$", r"$h_{w}(x+\varepsilon)$", r"$\alpha$-smoothing [2]", r"Clustered $\alpha$-smoothing")
	agent_colors = ("#000000", "#0B3C5D")
	agent_labels = ("Agent", "Ego agent")

	def _assign_to_centroids(points_np: np.ndarray, centroids_np: np.ndarray) -> np.ndarray:
		diff = points_np[:, None, :] - centroids_np[None, :, :]
		d2 = np.sum(diff * diff, axis=-1)
		return np.argmin(d2, axis=1)

	_w = TEXTWIDTH
	fig, axes = plt.subplots(nx, 4, figsize=(_w, 0.25 * _w * nx), sharex=True, sharey=True, constrained_layout=True)
	if nx == 1:
		axes = np.array([axes])

	for i in range(nx):
		single_centroids_i = _to_numpy(single_cluster_centroids[i])
		clust_centroids_i = _to_numpy(clusterized_centroids[i])

		column_data = (
			(original_end[i], None),
			(noisy_end[i], None),
			(single_end[i], single_centroids_i),
			(clust_end[i], clust_centroids_i),
		)

		for col_idx, (endpoints, centroids_ref) in enumerate(column_data):
			ax = axes[i, col_idx]
			_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)

			if centroids_ref is None:
				labels_np = np.zeros(endpoints.shape[0], dtype=np.int64)
				num_cls = 1
			else:
				labels_np = _assign_to_centroids(endpoints, centroids_ref)
				num_cls = int(centroids_ref.shape[0])

			for j in range(num_cls):
				mask_j = labels_np == j
				if not np.any(mask_j):
					continue
				color_j = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
				ax.scatter(
					endpoints[mask_j, 0],
					endpoints[mask_j, 1],
					s=8,
					alpha=0.45,
					color=color_j,
					linewidths=0,
					zorder=2,
				)

			# Input-trajectory markers for spatial context.
			for agent_idx in range(num_input_agents):
				agent_traj = x_traj[i, agent_idx]
				color = agent_colors[agent_idx % len(agent_colors)]
				label = None
				if i == 0 and col_idx == 0 and agent_idx < len(agent_labels):
					label = agent_labels[agent_idx]
				ax.plot(
					agent_traj[:, 0],
					agent_traj[:, 1],
					color=color,
					linestyle="None",
					marker="o",
					markersize=2.0,
					markerfacecolor=color,
					markeredgecolor=color,
					markeredgewidth=0.2,
					zorder=6,
					label=label,
				)

			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_aspect("equal", adjustable="box")

		for y_col in (1, 2, 3):
			axes[i, y_col].tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			for col_idx in range(4):
				axes[i, col_idx].set_title(col_titles[col_idx], fontsize=8)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, fontsize=8)
	fig.supxlabel("x (m)", fontsize=8, y=-0.02)
	fig.supylabel("y (m)", fontsize=8)
	fig_path = output_dir / "final_timestep_density.pdf"
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)

	return [fig_path]


def plot_final_timestep_examples(
	x: torch.Tensor,
	original_mc_samples: torch.Tensor,
	noisy_mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	single_cluster_centroids: torch.Tensor,
	clusterized_centroids: torch.Tensor,
	region,
	single_cluster_region,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot final-step endpoint samples and certified rectangles in one chart.

	Columns are Original / Noisy / single-cluster smoother / clusterized smoother.
	Certified final-step rectangles are overlaid on the smoother columns.
	"""

	output_dir.mkdir(parents=True, exist_ok=True)

	original = _reshape_trajectory_samples(original_mc_samples)
	noisy = _reshape_trajectory_samples(noisy_mc_samples)
	single = _reshape_trajectory_samples(single_cluster_mc_samples)
	clust = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)
	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	original_end = original[:, :, -1, :]
	noisy_end = noisy[:, :, -1, :]
	single_end = single[:, :, -1, :]
	clust_end = clust[:, :, -1, :]

	# Final-step rectangle slices: (nx, k, 2)
	if region.lower.shape[-1] % 2 != 0 or single_cluster_region.lower.shape[-1] % 2 != 0:
		raise ValueError("Expected region output dimension arranged as [x1,y1,...,xT,yT]")

	region_lower_final = _to_numpy(region.lower[..., -2:])
	region_upper_final = _to_numpy(region.upper[..., -2:])
	sc_lower_final = _to_numpy(single_cluster_region.lower[..., -2:])
	sc_upper_final = _to_numpy(single_cluster_region.upper[..., -2:])

	nx = original.shape[0]

	all_x_coords = np.concatenate([
		original_end[..., 0].reshape(-1),
		noisy_end[..., 0].reshape(-1),
		single_end[..., 0].reshape(-1),
		clust_end[..., 0].reshape(-1),
		x_traj[..., 0].reshape(-1),
		region_lower_final[..., 0].reshape(-1),
		region_upper_final[..., 0].reshape(-1),
		sc_lower_final[..., 0].reshape(-1),
		sc_upper_final[..., 0].reshape(-1),
	])
	all_y_coords = np.concatenate([
		original_end[..., 1].reshape(-1),
		noisy_end[..., 1].reshape(-1),
		single_end[..., 1].reshape(-1),
		clust_end[..., 1].reshape(-1),
		x_traj[..., 1].reshape(-1),
		region_lower_final[..., 1].reshape(-1),
		region_upper_final[..., 1].reshape(-1),
		sc_lower_final[..., 1].reshape(-1),
		sc_upper_final[..., 1].reshape(-1),
	])
	x_pad = max((float(all_x_coords.max()) - float(all_x_coords.min())) * 0.12, 0.5)
	y_pad = max((float(all_y_coords.max()) - float(all_y_coords.min())) * 0.12, 0.5)
	x_lim = (float(all_x_coords.min()) - x_pad, float(all_x_coords.max()) + x_pad)
	y_lim = (float(all_y_coords.min()) - y_pad, float(all_y_coords.max()) + y_pad)
	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))

	col_titles = (r"$h_{w}(x)$", r"$h_{w}(x+\varepsilon)$", r"$\alpha$-smoothing", r"Clustered $\alpha$-smoothing")
	agent_colors = ("#000000", "#0B3C5D")
	agent_labels = ("Agent", "Ego agent")

	def _assign_to_centroids(points_np: np.ndarray, centroids_np: np.ndarray) -> np.ndarray:
		diff = points_np[:, None, :] - centroids_np[None, :, :]
		d2 = np.sum(diff * diff, axis=-1)
		return np.argmin(d2, axis=1)

	_w = TEXTWIDTH
	fig, axes = plt.subplots(nx, 4, figsize=(_w, 0.25 * _w * nx), sharex=True, sharey=True, constrained_layout=True)
	if nx == 1:
		axes = np.array([axes])

	for i in range(nx):
		single_centroids_i = _to_numpy(single_cluster_centroids[i])
		clust_centroids_i = _to_numpy(clusterized_centroids[i])

		column_data = (
			(original_end[i], None),
			(noisy_end[i], None),
			(single_end[i], single_centroids_i),
			(clust_end[i], clust_centroids_i),
		)

		for col_idx, (endpoints, centroids_ref) in enumerate(column_data):
			ax = axes[i, col_idx]
			_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)

			if centroids_ref is None:
				labels_np = np.zeros(endpoints.shape[0], dtype=np.int64)
				num_cls = 1
			else:
				labels_np = _assign_to_centroids(endpoints, centroids_ref)
				num_cls = int(centroids_ref.shape[0])

			for j in range(num_cls):
				mask_j = labels_np == j
				if not np.any(mask_j):
					continue
				color_j = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
				ax.scatter(
					endpoints[mask_j, 0],
					endpoints[mask_j, 1],
					s=8,
					alpha=0.45,
					color=color_j,
					linewidths=0,
					zorder=2,
				)

			# Overlay certified final-step rectangles on smoother columns.
			if col_idx == 2:
				sc_k = int(single_cluster_region.lower.shape[1])
				for c in range(sc_k):
					color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = sc_lower_final[i, c]
					x_hi, y_hi = sc_upper_final[i, c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo),
						x_hi - x_lo,
						y_hi - y_lo,
						edgecolor=color,
						facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.8,
						linestyle="-",
						zorder=3,
					)
					ax.add_patch(rect)
			elif col_idx == 3:
				k = int(region.lower.shape[1])
				for c in range(k):
					color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = region_lower_final[i, c]
					x_hi, y_hi = region_upper_final[i, c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo),
						x_hi - x_lo,
						y_hi - y_lo,
						edgecolor=color,
						facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.8,
						linestyle="-",
						zorder=3,
					)
					ax.add_patch(rect)

			for agent_idx in range(num_input_agents):
				agent_traj = x_traj[i, agent_idx]
				color = agent_colors[agent_idx % len(agent_colors)]
				label = None
				if i == 0 and col_idx == 0 and agent_idx < len(agent_labels):
					label = agent_labels[agent_idx]
				ax.plot(
					agent_traj[:, 0],
					agent_traj[:, 1],
					color=color,
					linestyle="None",
					marker="o",
					markersize=2.0,
					markerfacecolor=color,
					markeredgecolor=color,
					markeredgewidth=0.2,
					zorder=6,
					label=label,
				)

			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)

		for y_col in (1, 2, 3):
			axes[i, y_col].tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			for col_idx in range(4):
				axes[i, col_idx].set_title(col_titles[col_idx], fontsize=8)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, fontsize=8)
	fig.supxlabel("x (m)", fontsize=8, y=-0.02)
	fig.supylabel("y (m)", fontsize=8)
	fig_path = output_dir / "final_timestep_examples.pdf"
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)

	return [fig_path]


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
	car_width = car_length * 0.40
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
	single_cluster_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot trajectory samples for each x point.

	Args:
		x: Tensor (nx, num_input_agents * num_input_timesteps * 2) containing
		   flattened input trajectories used for dashed overlays.
		mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		single_cluster_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps)
		   from a smoother with max_num_clusters=1.
		clusterized_mc_samples: Tensor (nx, num_samples, 2 * num_timesteps).
		output_dir: Folder to save charts.
		num_input_agents: Number of agents encoded in x.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = _reshape_trajectory_samples(mc_samples)
	single_cluster = _reshape_trajectory_samples(single_cluster_mc_samples)
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
	_w = TEXTWIDTH
	fig, axes = plt.subplots(nx, 3, figsize=(_w, 0.30 * _w * nx), sharex=True, sharey=True)
	if nx == 1:
		axes = np.array([axes])

	all_x = np.concatenate(
		(
			raw[:, :, :, 0].reshape(-1),
			single_cluster[:, :, :, 0].reshape(-1),
			clusterized[:, :, :, 0].reshape(-1),
			x_traj[:, :, :, 0].reshape(-1),
		)
	)
	all_y = np.concatenate(
		(
			raw[:, :, :, 1].reshape(-1),
			single_cluster[:, :, :, 1].reshape(-1),
			clusterized[:, :, :, 1].reshape(-1),
			x_traj[:, :, :, 1].reshape(-1),
		)
	)
	x_min, x_max = np.min(all_x), np.max(all_x)
	y_min, y_max = np.min(all_y), np.max(all_y)
	x_pad = max((x_max - x_min) * 0.05, 1e-6)
	y_pad = max((y_max - y_min) * 0.05, 1e-6)
	x_lim = (x_min - x_pad, x_max + x_pad)
	y_lim = (y_min - y_pad - 5.0, y_max + y_pad + 5.0)

	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))
	car_size = 0.022 * max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
	car_size = max(car_size, 0.25)

	for i in range(nx):
		ax_raw = axes[i, 0]
		ax_single = axes[i, 1]
		ax_cluster = axes[i, 2]

		_plot_street_marks(ax_raw, x_extent=x_extent, y_extent=y_extent)
		_plot_street_marks(ax_single, x_extent=x_extent, y_extent=y_extent)
		_plot_street_marks(ax_cluster, x_extent=x_extent, y_extent=y_extent)

		for j in range(num_samples):
			traj = raw[i, j]
			ax_raw.plot(
				traj[:, 0],
				traj[:, 1],
				color="#1B2A4A",
				alpha=0.16,
				linewidth=0.9,
				zorder=1,
				label=None,
			)

			traj_single = single_cluster[i, j]
			ax_single.plot(
				traj_single[:, 0],
				traj_single[:, 1],
				color="#D55E00",
				alpha=0.35,
				linewidth=1.0,
				zorder=1,
				label=None,
			)

			traj_cluster = clusterized[i, j]
			ax_cluster.plot(
				traj_cluster[:, 0],
				traj_cluster[:, 1],
				color="#009E73",
				alpha=0.35,
				linewidth=1.0,
				zorder=1,
				label=None,
			)

		agent_colors = ("#000000", "#0B3C5D")
		for agent_idx in range(num_input_agents):
			agent_traj = x_traj[i, agent_idx]
			color = agent_colors[agent_idx % len(agent_colors)]
			label = "Target agent" if agent_idx == 0 else "Ego agent"
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
			for ax_mid in (ax_single, ax_cluster):
				ax_mid.plot(
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
			for ax_car in (ax_raw, ax_single, ax_cluster):
				_plot_car_icon(
					ax_car,
					start=agent_traj[0],
					next_point=agent_traj[next_idx],
					color=color,
					car_length=car_size,
				)

		for ax in (ax_raw, ax_single, ax_cluster):
			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)

		ax_single.tick_params(axis="y", which="both", left=False, labelleft=False)
		ax_cluster.tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			ax_raw.set_title("Original", fontsize=8)
			ax_single.set_title("Rekavandi et al, 2024", fontsize=8)
			ax_cluster.set_title("Ours", fontsize=8)

	fig.supxlabel("x", fontsize=8)
	fig.supylabel("y", fontsize=8)
	fig.tight_layout()
	fig_path = output_dir / "trajectory_samples_comparison.pdf"
	fig.savefig(fig_path, dpi=180, bbox_inches="tight")
	plt.close(fig)

	return [fig_path]


def plot_timestep_rectangles_visualization(
	x: torch.Tensor,
	original_mc_samples: torch.Tensor,
	noisy_mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	region,
	single_cluster_region,
	output_dir: Path,
	num_input_agents: int = 2,
	timesteps: list[int] | None = None,
) -> list[Path]:
	"""Plot timestep-wise certified rectangles with one row per input trajectory.

	Four columns are shown per row: Original, Noisy, single-cluster smoother,
	and multi-cluster smoother. Certified rectangles are shown on smoother columns.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	original = _reshape_trajectory_samples(original_mc_samples)
	noisy = _reshape_trajectory_samples(noisy_mc_samples)
	single_cluster = _reshape_trajectory_samples(single_cluster_mc_samples)
	clusterized = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)

	if x_np.ndim != 2 or x_np.shape[1] % (2 * num_input_agents) != 0:
		raise ValueError("x must have shape (nx, num_input_agents * num_input_timesteps * 2)")

	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	if region.lower.shape[-1] % 2 != 0:
		raise ValueError("Expected region output dimension arranged as [x1,y1,...,xT,yT]")
	num_timesteps = region.lower.shape[-1] // 2

	if timesteps is None:
		timesteps_to_plot = list(range(num_timesteps))
	else:
		timesteps_to_plot = timesteps
		for t in timesteps_to_plot:
			if t < 0 or t >= num_timesteps:
				raise ValueError(f"timestep {t} out of range [0, {num_timesteps - 1}]")

	plt.rcParams.update({
		"text.usetex": True,
		"font.serif": ["Computer Modern Roman"],
		"font.size": 10,
		"axes.labelsize": 10,
		"axes.titlesize": 10,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
		"legend.fontsize": 9,
	})

	nx = original.shape[0]
	_w = TEXTWIDTH
	fig, axes = plt.subplots(nx, 2, figsize=(_w, 0.25 * _w * nx), sharex=True, sharey=True, constrained_layout=True)
	if nx == 1:
		axes = np.array([axes])

	# Compute limits using all data (not just x_index) to match trajectory_samples figure
	all_x = np.concatenate((original[:, :, :, 0].reshape(-1), noisy[:, :, :, 0].reshape(-1), single_cluster[:, :, :, 0].reshape(-1), clusterized[:, :, :, 0].reshape(-1), x_traj[:, :, :, 0].reshape(-1)))
	all_y = np.concatenate((original[:, :, :, 1].reshape(-1), noisy[:, :, :, 1].reshape(-1), single_cluster[:, :, :, 1].reshape(-1), clusterized[:, :, :, 1].reshape(-1), x_traj[:, :, :, 1].reshape(-1)))
	x_min, x_max = np.min(all_x), np.max(all_x)
	y_min, y_max = np.min(all_y), np.max(all_y)
	x_pad = max((x_max - x_min) * 0.05, 1e-6)
	y_pad = max((y_max - y_min) * 0.05, 1e-6)
	x_lim = (x_min - x_pad, x_max + x_pad)
	y_lim = (y_min - y_pad - 5.0, y_max + y_pad + 5.0)

	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))

	for i in range(nx):
		ax_single, ax_cluster = axes[i]

		for ax in (ax_single, ax_cluster):
			_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)

		single_cluster_i = single_cluster[i]
		clusterized_i = clusterized[i]
		x_traj_i = x_traj[i]

		lower = _to_numpy(region.lower[i]).reshape(region.lower.shape[1], -1, 2)
		upper = _to_numpy(region.upper[i]).reshape(region.upper.shape[1], -1, 2)
		num_clusters = lower.shape[0]

		sc_lower = _to_numpy(single_cluster_region.lower[i]).reshape(single_cluster_region.lower.shape[1], -1, 2)
		sc_upper = _to_numpy(single_cluster_region.upper[i]).reshape(single_cluster_region.upper.shape[1], -1, 2)
		sc_num_clusters = sc_lower.shape[0]

		for j in range(single_cluster_i.shape[0]):
			traj_single = single_cluster_i[j]
			ax_single.plot(traj_single[:, 0], traj_single[:, 1], color="#D55E00", alpha=0.20, linewidth=0.9, zorder=1)
			traj_cluster = clusterized_i[j]
			ax_cluster.plot(traj_cluster[:, 0], traj_cluster[:, 1], color="#009E73", alpha=0.20, linewidth=0.9, zorder=1)

		# Draw multi-cluster rectangles on right panel only.
		for c in range(num_clusters):
			color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
			for t in timesteps_to_plot:
				x_lo, y_lo = lower[c, t]
				x_hi, y_hi = upper[c, t]
				w, h = x_hi - x_lo, y_hi - y_lo
				face = to_rgba(color, alpha=0.10)
				lw = 0.8 if len(timesteps_to_plot) <= 2 else 0.5
				rect = mpatches.Rectangle(
					(x_lo, y_lo), w, h,
					edgecolor=color, facecolor=face,
					linewidth=lw, linestyle="-", zorder=2,
				)
				ax_cluster.add_patch(rect)

		# Draw single-cluster rectangles on middle panel.
		for c in range(sc_num_clusters):
			color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
			for t in timesteps_to_plot:
				x_lo, y_lo = sc_lower[c, t]
				x_hi, y_hi = sc_upper[c, t]
				w, h = x_hi - x_lo, y_hi - y_lo
				face = to_rgba(color, alpha=0.10)
				lw = 0.8 if len(timesteps_to_plot) <= 2 else 0.5
				rect = mpatches.Rectangle(
					(x_lo, y_lo), w, h,
					edgecolor=color, facecolor=face,
					linewidth=lw, linestyle="-", zorder=2,
				)
				ax_single.add_patch(rect)

		agent_colors = ("#000000", "#0B3C5D")
		agent_labels = ("Agent", "Ego agent")
		for agent_idx in range(num_input_agents):
			agent_traj = x_traj_i[agent_idx]
			color = agent_colors[agent_idx % len(agent_colors)]
			label = agent_labels[agent_idx] if (i == 0 and agent_idx < len(agent_labels)) else None
			for ax in (ax_single, ax_cluster):
				ax.plot(agent_traj[:, 0], agent_traj[:, 1], linestyle="None", marker="o", markersize=2.2, color=color, zorder=6, label=label if ax is ax_single else None)

		for ax in (ax_single, ax_cluster):
			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)

		axes[i, 1].tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			ax_single.set_title("Rekavandi et al, 2024", fontsize=8)
			ax_cluster.set_title("Ours", fontsize=8)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, fontsize=8)
	fig.supxlabel("x", fontsize=8, y=-0.02)
	fig.supylabel("y", fontsize=8)

	suffix = "" if timesteps is None else "_t" + "_".join(str(t) for t in timesteps_to_plot)
	fig_path = output_dir / f"timestep_rectangles_visualization{suffix}.pdf"
	fig.savefig(fig_path, dpi=180)
	plt.close(fig)

	return [fig_path]


def _add_region_patch(ax: plt.Axes, cluster: Cluster, cluster_index: int, color: str) -> None:
	region = cluster.regions
	centroid = cluster.centroids[0, cluster_index].detach().cpu().numpy()

	if hasattr(region, "radius"):
		radius = float(region.radius[0, cluster_index].detach().cpu().item())
		patch = mpatches.Circle(
			centroid,
			radius=radius,
			fill=False,
			edgecolor=color,
			linewidth=1.4,
			linestyle="-",
		)
	else:
		lower = region.lower[0, cluster_index].detach().cpu().numpy()
		upper = region.upper[0, cluster_index].detach().cpu().numpy()
		patch = mpatches.Rectangle(
			lower,
			*(upper - lower),
			fill=False,
			edgecolor=color,
			linewidth=1.4,
			linestyle="-",
		)

	ax.add_patch(patch)


def plot_comparison(
	outputs: torch.Tensor,
	clusters_by_key: dict[tuple[str, float], Cluster],
	set_types: list[str],
	coverages: list[float],
	output_dir: Path,
) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	samples_np = outputs[0].detach().cpu().numpy()
	x_pad = 0.8
	y_pad = 0.8

	all_x = samples_np[:, 0]
	all_y = samples_np[:, 1]
	x_lim = (float(all_x.min() - x_pad), float(all_x.max() + x_pad))
	y_lim = (float(all_y.min() - y_pad), float(all_y.max() + y_pad))

	plt.rcParams.update({
		"text.usetex": True,
		"font.serif": ["Computer Modern Roman"],
		"font.size": 8,
		"axes.labelsize": 8,
		"axes.titlesize": 8,
		"xtick.labelsize": 7,
		"ytick.labelsize": 7,
		"legend.fontsize": 7,
	})

	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
	paths: list[Path] = []

	for coverage in coverages:
		_w = HALFWIDTH
		fig, axes = plt.subplots(
			1, len(set_types),
			figsize=(_w, 0.55 * _w),
			sharex=True,
			sharey=True,
		)

		if len(set_types) == 1:
			axes = [axes]

		for col, set_type in enumerate(set_types):
			ax = axes[col]
			cluster = clusters_by_key[(set_type, coverage)]
			assignments = cluster.assignments[0].detach().cpu().numpy()
			active_k = int(cluster.k_per_row[0].item())

			for j in range(active_k):
				cluster_points = samples_np[assignments == j]
				ax.scatter(
					cluster_points[:, 0],
					cluster_points[:, 1],
					s=8,
					alpha=0.05,
					color=colors[j % len(colors)],
				)
				centroid = cluster.centroids[0, j].detach().cpu().numpy()
				ax.scatter(
					centroid[0],
					centroid[1],
					marker="x",
					s=12,
					linewidths=1.2,
					color=colors[j % len(colors)],
				)
				_add_region_patch(ax=ax, cluster=cluster, cluster_index=j, color=colors[j % len(colors)])

			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_aspect("equal", adjustable="box")
			ax.grid(alpha=0.25, linestyle=":")

		fig.subplots_adjust(wspace=0.05)
		output_path = output_dir / f"axis_aligned_hyperrectangle_cov{coverage:.2f}.pdf"
		fig.savefig(output_path, dpi=180, bbox_inches="tight")
		plt.close(fig)
		paths.append(output_path)

	return paths


def plot_bound_clusterized_grid(
	ratio_values: np.ndarray,
	alphas: list[float],
	p_v_values: list[float],
	num_samples_list: list[int],
	y_data: dict[tuple[float, float, int], np.ndarray],
	colors: dict[int, str],
	output_dir: Path,
) -> Path:
	output_dir.mkdir(parents=True, exist_ok=True)

	_w = TEXTWIDTH
	fig, axes = plt.subplots(
		len(alphas), len(p_v_values),
		figsize=(_w, 0.60 * _w),
		sharex=True, sharey=True,
	)

	for i, alpha in enumerate(alphas):
		for j, p_v in enumerate(p_v_values):
			ax = axes[i, j]
			for n in num_samples_list:
				y = y_data[(alpha, p_v, n)]
				ax.plot(
					ratio_values, y,
					linewidth=1.5,
					color=colors[n],
					label=rf"$N={n}$",
				)

			ax.set_title(rf"$\alpha={alpha},\ p_{{\mathcal{{V}}}}={p_v}$", fontsize=8)
			ax.grid(alpha=0.3, linestyle=":")
			ax.set_yticks([0.0, 0.5, 1.0])

	fig.supylabel("Certified probability", fontsize=8)

	fig.tight_layout()
	fig.subplots_adjust(bottom=0.22)

	fig.text(
		0.5, 0.12,
		r"Ratio $p_{\mathcal{R}} / p_{\mathcal{V}}$",
		ha="center", va="center",
		fontsize=8,
	)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(
		handles, labels,
		loc="lower center",
		ncol=len(num_samples_list),
		bbox_to_anchor=(0.5, 0.0),
		framealpha=0.95,
	)

	output_path = output_dir / "bound_clusterized_grid.pdf"
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)

	return output_path

