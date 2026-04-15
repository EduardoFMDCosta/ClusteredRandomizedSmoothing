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
	fig, axes = plt.subplots(nx, 3, figsize=(_w, 0.55 * _w * nx), sharex=True, sharey=True)
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
	y_lim = (y_min - y_pad, y_max + y_pad)

	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))
	car_size = 0.018 * max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
	car_size = max(car_size, 0.2)

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
				color="#1B5E20",
				alpha=0.16,
				linewidth=0.9,
				zorder=1,
				label="Original model" if j == 0 else None,
			)

			traj_single = single_cluster[i, j]
			ax_single.plot(
				traj_single[:, 0],
				traj_single[:, 1],
				color="#FFA726",
				alpha=0.35,
				linewidth=1.0,
				zorder=1,
				label="Smoothed (1 cluster)" if j == 0 else None,
			)

			traj_cluster = clusterized[i, j]
			ax_cluster.plot(
				traj_cluster[:, 0],
				traj_cluster[:, 1],
				color="#66BB6A",
				alpha=0.35,
				linewidth=1.0,
				zorder=1,
				label="Smoothed model" if j == 0 else None,
			)

		agent_colors = ("#1B5E20", "#0B3C5D")
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
			ax.grid(alpha=0.25, linestyle=":")

		ax_single.tick_params(axis="y", which="both", left=False, labelleft=False)
		ax_cluster.tick_params(axis="y", which="both", left=False, labelleft=False)

		if i == 0:
			ax_raw.legend(loc="upper left", fontsize=7, handlelength=1.5, handletextpad=0.4)
			ax_single.legend(loc="upper left", fontsize=7, handlelength=1.5, handletextpad=0.4)
			ax_cluster.legend(loc="upper left", fontsize=7, handlelength=1.5, handletextpad=0.4)

	fig.supxlabel("x")
	fig.supylabel("y")
	fig.tight_layout()
	fig_path = output_dir / "trajectory_samples_comparison.pdf"
	fig.savefig(fig_path, dpi=180, bbox_inches="tight")
	plt.close(fig)

	return [fig_path]


def plot_timestep_rectangles_visualization(
	x: torch.Tensor,
	mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clusterized_mc_samples: torch.Tensor,
	region,
	single_cluster_region,
	output_dir: Path,
	num_input_agents: int = 2,
	x_index: int = 0,
	timesteps: list[int] | None = None,
) -> list[Path]:
	"""Plot timestep-wise certified rectangles for a single input trajectory.

	Three columns are shown: raw trajectories with multi-cluster rectangles (left),
	single-cluster smoothed trajectories with single-cluster rectangles (middle),
	and multi-cluster smoothed trajectories with multi-cluster rectangles (right).
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	raw = _reshape_trajectory_samples(mc_samples)
	single_cluster = _reshape_trajectory_samples(single_cluster_mc_samples)
	clusterized = _reshape_trajectory_samples(clusterized_mc_samples)
	x_np = _to_numpy(x)

	if not (0 <= x_index < raw.shape[0]):
		raise ValueError(f"x_index must be in [0, {raw.shape[0] - 1}]")
	if x_np.ndim != 2 or x_np.shape[1] % (2 * num_input_agents) != 0:
		raise ValueError("x must have shape (nx, num_input_agents * num_input_timesteps * 2)")

	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	if region.lower.shape[-1] % 2 != 0:
		raise ValueError("Expected region output dimension arranged as [x1,y1,...,xT,yT]")

	# Use one input to keep rectangles readable.
	raw_i = raw[x_index]
	single_cluster_i = single_cluster[x_index]
	clusterized_i = clusterized[x_index]
	x_traj_i = x_traj[x_index]

	# Region boxes: (k, 2T) -> (k, T, 2)
	lower = _to_numpy(region.lower[x_index]).reshape(region.lower.shape[1], -1, 2)
	upper = _to_numpy(region.upper[x_index]).reshape(region.upper.shape[1], -1, 2)
	num_clusters = lower.shape[0]
	num_timesteps = lower.shape[1]

	sc_lower = _to_numpy(single_cluster_region.lower[x_index]).reshape(single_cluster_region.lower.shape[1], -1, 2)
	sc_upper = _to_numpy(single_cluster_region.upper[x_index]).reshape(single_cluster_region.upper.shape[1], -1, 2)
	sc_num_clusters = sc_lower.shape[0]

	if timesteps is None:
		timesteps_to_plot = list(range(num_timesteps))
	else:
		timesteps_to_plot = timesteps

	plt.rcParams.update({
		"text.usetex": True,
		"font.serif": ["Computer Modern Roman"],
	})

	_w = 1.20 * TEXTWIDTH
	fig, axes = plt.subplots(1, 3, figsize=(_w, 0.33 * _w), sharex=True, sharey=True)
	ax_left, ax_mid, ax_right = axes

	all_x = np.concatenate((raw_i[:, :, 0].reshape(-1), single_cluster_i[:, :, 0].reshape(-1), clusterized_i[:, :, 0].reshape(-1), x_traj_i[:, :, 0].reshape(-1)))
	all_y = np.concatenate((raw_i[:, :, 1].reshape(-1), single_cluster_i[:, :, 1].reshape(-1), clusterized_i[:, :, 1].reshape(-1), x_traj_i[:, :, 1].reshape(-1)))
	x_min, x_max = np.min(all_x), np.max(all_x)
	y_min, y_max = np.min(all_y), np.max(all_y)
	x_pad = max((x_max - x_min) * 0.05, 1e-6)
	y_pad = max((y_max - y_min) * 0.05, 1e-6)
	x_lim = (x_min - x_pad, x_max + x_pad)
	y_lim = (y_min - y_pad, y_max + y_pad)

	x_extent = max(abs(x_lim[0]), abs(x_lim[1]))
	y_extent = max(abs(y_lim[0]), abs(y_lim[1]))

	for ax in (ax_left, ax_mid, ax_right):
		_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)

	for j in range(raw_i.shape[0]):
		traj = raw_i[j]
		ax_left.plot(traj[:, 0], traj[:, 1], color="#1B5E20", alpha=0.09, linewidth=0.8, zorder=1)
		traj_single = single_cluster_i[j]
		ax_mid.plot(traj_single[:, 0], traj_single[:, 1], color="#FFA726", alpha=0.20, linewidth=0.9, zorder=1)
		traj_cluster = clusterized_i[j]
		ax_right.plot(traj_cluster[:, 0], traj_cluster[:, 1], color="#66BB6A", alpha=0.20, linewidth=0.9, zorder=1)

	cluster_colors = ["#E31A1C", "#1F78B4", "#33A02C", "#FDBF6F", "#FF7F00"]

	# Draw multi-cluster rectangles on left and right panels.
	for c in range(num_clusters):
		color = cluster_colors[c % len(cluster_colors)]
		for t in timesteps_to_plot:
			x_lo, y_lo = lower[c, t]
			x_hi, y_hi = upper[c, t]
			w, h = x_hi - x_lo, y_hi - y_lo
			face = to_rgba(color, alpha=0.10)
			lw = 1.8 if len(timesteps_to_plot) <= 2 else 1.0
			for ax in (ax_left, ax_right):
				rect = mpatches.Rectangle(
					(x_lo, y_lo), w, h,
					edgecolor=color, facecolor=face,
					linewidth=lw, linestyle="-", zorder=2,
				)
				ax.add_patch(rect)

	# Draw single-cluster rectangles on middle panel.
	for c in range(sc_num_clusters):
		color = cluster_colors[c % len(cluster_colors)]
		for t in timesteps_to_plot:
			x_lo, y_lo = sc_lower[c, t]
			x_hi, y_hi = sc_upper[c, t]
			w, h = x_hi - x_lo, y_hi - y_lo
			face = to_rgba(color, alpha=0.10)
			lw = 1.8 if len(timesteps_to_plot) <= 2 else 1.0
			rect = mpatches.Rectangle(
				(x_lo, y_lo), w, h,
				edgecolor=color, facecolor=face,
				linewidth=lw, linestyle="-", zorder=2,
			)
			ax_mid.add_patch(rect)

	# Input trajectory markers for context.
	agent_colors = ("#1B5E20", "#0B3C5D")
	for agent_idx in range(num_input_agents):
		agent_traj = x_traj_i[agent_idx]
		color = agent_colors[agent_idx % len(agent_colors)]
		for ax in (ax_left, ax_mid, ax_right):
			ax.plot(agent_traj[:, 0], agent_traj[:, 1], linestyle="None", marker="o", markersize=2.2, color=color, zorder=6)

	for ax in (ax_left, ax_mid, ax_right):
		ax.set_xlim(*x_lim)
		ax.set_ylim(*y_lim)
		ax.grid(alpha=0.25, linestyle=":")

	fig.tight_layout()
	suffix = "" if timesteps is None else "_t" + "_".join(str(t) for t in timesteps_to_plot)
	fig_path = output_dir / f"timestep_rectangles_visualization{suffix}.pdf"
	fig.savefig(fig_path, dpi=180, bbox_inches="tight")
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
			linewidth=2.0,
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
			linewidth=2.0,
			linestyle="-",
		)

	ax.add_patch(patch)


def plot_comparison(
	outputs: torch.Tensor,
	clusters_by_key: dict[tuple[str, float], Cluster],
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
	})

	colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
	paths: list[Path] = []

	for coverage in coverages:
		_w = 0.80 * TEXTWIDTH
		fig, axes = plt.subplots(
			1, 2,
			figsize=(_w, 0.50 * _w),
			sharex=True,
			sharey=True,
		)

		for col, set_type in enumerate(["HyperRectangle", "EuclideanBall"]):
			ax = axes[col]
			cluster = clusters_by_key[(set_type, coverage)]
			assignments = cluster.assignments[0].detach().cpu().numpy()
			active_k = int(cluster.k_per_row[0].item())

			for j in range(active_k):
				cluster_points = samples_np[assignments == j]
				ax.scatter(
					cluster_points[:, 0],
					cluster_points[:, 1],
					s=12,
					alpha=0.1,
					color=colors[j % len(colors)],
				)
				centroid = cluster.centroids[0, j].detach().cpu().numpy()
				ax.scatter(
					centroid[0],
					centroid[1],
					marker="x",
					s=18,
					linewidths=1.2,
					color=colors[j % len(colors)],
				)
				_add_region_patch(ax=ax, cluster=cluster, cluster_index=j, color=colors[j % len(colors)])

			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_aspect("equal", adjustable="box")
			ax.grid(alpha=0.25, linestyle=":")

		fig.subplots_adjust(wspace=0.05)
		output_path = output_dir / f"hyperrectangle_vs_ball_cov{coverage:.2f}.pdf"
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
					linewidth=2.0,
					color=colors[n],
					label=rf"$N={n}$",
				)

			ax.set_title(rf"$\alpha={alpha},\ p_{{\mathcal{{V}}}}={p_v}$")
			ax.grid(alpha=0.3, linestyle=":")
			ax.set_yticks([0.0, 0.5, 1.0])

	fig.supylabel("Certified probability", fontsize=plt.rcParams["axes.labelsize"])

	fig.tight_layout()
	fig.subplots_adjust(bottom=0.22)

	fig.text(
		0.5, 0.12,
		r"Ratio $p_{\mathcal{R}} / p_{\mathcal{V}}$",
		ha="center", va="center",
		fontsize=plt.rcParams["axes.labelsize"],
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

