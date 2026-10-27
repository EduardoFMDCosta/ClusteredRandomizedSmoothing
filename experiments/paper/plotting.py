from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
	from randomized_smoothing.clusters import Cluster


TEXTWIDTH = 5.5  # inches
HALFWIDTH = 0.50 * TEXTWIDTH  # for \begin{minipage}{0.50\textwidth}

plt.rcParams.update(
	{
		"text.usetex": True,
		"font.family": "serif",
		"font.serif": ["Computer Modern Roman"],
		"mathtext.fontset": "cm",
		"font.size": 10,
		"axes.labelsize": 10,
		"axes.titlesize": 10,
		"xtick.labelsize": 9,
		"ytick.labelsize": 9,
		"legend.fontsize": 10,
	}
)

CLUSTER_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _resolve_trajflow_gt_file(data_dir: Path) -> Path:
	candidates = [
		data_dir / "Outputs_gt.npy",
		data_dir / "Output_gt.npy",
	]
	for path in candidates:
		if path.exists():
			return path
	raise FileNotFoundError(
		"Could not find ground-truth output file. Expected one of: "
		f"{', '.join(str(p) for p in candidates)}"
	)


def _load_trajflow_target_displacements() -> np.ndarray:
	data_dir = Path(__file__).resolve().parent.parent / "TrajFlowExperiment" / "Data"
	inputs = np.load(str(data_dir / "Inputs.npy"))
	outputs_gt = np.load(str(_resolve_trajflow_gt_file(data_dir)))

	if outputs_gt.ndim != 4 or outputs_gt.shape[-1] != 2:
		raise ValueError(
			"Expected GT outputs with shape (num_samples, num_agents, num_timesteps, 2); "
			f"got {tuple(outputs_gt.shape)}"
		)
	if outputs_gt.shape[1] != 1:
		raise ValueError(
			"Expected GT outputs to contain exactly one target-agent trajectory; "
			f"got shape {tuple(outputs_gt.shape)}"
		)
	if inputs.shape[0] != outputs_gt.shape[0]:
		raise ValueError(
			"Inputs and GT outputs have different sample counts: "
			f"Inputs={inputs.shape[0]}, GT={outputs_gt.shape[0]}"
		)

	# Raw TrajFlow inputs store the tracked/target vehicle at agent index 1.
	target_initial = inputs[:, 1, 0, :2]
	target_final = outputs_gt[:, 0, -1, :]
	finite_mask = np.isfinite(target_initial).all(axis=1) & np.isfinite(target_final).all(axis=1)
	displacements = np.full(inputs.shape[0], np.inf, dtype=float)
	displacements[finite_mask] = np.linalg.norm(target_final[finite_mask] - target_initial[finite_mask], axis=1)
	return displacements


def _to_numpy(x: torch.Tensor) -> np.ndarray:
	return x.detach().cpu().numpy()


def _draw_smoother_distribution_panel(
	ax,
	*,
	i: int,
	outputs: torch.Tensor,
	outputs_np: np.ndarray,
	smoother_mc_np: np.ndarray,
	region,
	region_lower_np: np.ndarray,
	region_upper_np: np.ndarray,
	raw_bins: np.ndarray | int,
	mc_bins: np.ndarray | int,
	smoother_label: str,
	smoother_color: str,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
	show_ylabel: bool = True,
) -> None:
	"""Render one smoother-vs-original distribution panel onto ``ax``.

	Shared between the single-panel and combined plots so the visual
	encoding (cluster step histograms, region shading, Voronoi splits,
	legend) stays identical across figures.
	"""
	raw_color = "#111111"
	region_color = "#7A7A7A"
	split_color = "#D62728"

	ax.tick_params(axis="both", labelsize=9)
	k_regions = region_lower_np.shape[1]
	if isinstance(raw_bins, np.ndarray):
		shared_bins = np.asarray(raw_bins, dtype=float)
	else:
		shared_bins = np.histogram_bin_edges(outputs_np[i], bins=raw_bins)
	bin_width = float(shared_bins[1] - shared_bins[0])
	n_total = int(outputs_np[i].size)

	# Order regions left-to-right by their center for consistent coloring.
	region_centers_i = 0.5 * (region_lower_np[i] + region_upper_np[i])
	region_order = np.argsort(region_centers_i)

	if k_regions >= 2:
		# Local palette tuned to contrast with both smoother colours
		# (red ``#E31A1C`` and blue ``#1F78B4``) and to read clearly as
		# step outlines over a filled smoothed-predictor histogram.
		original_palette = ("#222222", "#D55E00", "#009E73", "#CC79A7", "#9467BD")
		region_distances = _to_numpy(region.distance(outputs[i])[i])
		assignments = np.argmin(region_distances, axis=-1)
		for rank, region_idx in enumerate(region_order):
			cluster_mask = assignments == region_idx
			if not np.any(cluster_mask):
				continue
			cluster_color = original_palette[rank % len(original_palette)]
			n_cluster = int(cluster_mask.sum())
			# Per-cluster weights ``1 / (n_total * bin_width)`` give the
			# same scale as ``density=True`` over the full data, so the
			# stack across clusters reproduces the full histogram.
			ax.hist(
				outputs_np[i][cluster_mask],
				bins=shared_bins,
				density=False,
				weights=np.full(n_cluster, 1.0 / (n_total * bin_width)),
				histtype="step",
				linewidth=2.0,
				edgecolor=cluster_color,
				label=f"Original (cluster {rank + 1})",
			)
	else:
		ax.hist(
			outputs_np[i],
			bins=shared_bins,
			density=True,
			histtype="step",
			linewidth=2.2,
			color=raw_color,
			label="Original model",
		)

	ax.hist(
		smoother_mc_np[i],
		bins=mc_bins,
		density=True,
		alpha=0.30,
		color=smoother_color,
		edgecolor=smoother_color,
		linewidth=1.4,
		label=smoother_label,
	)

	for j in range(k_regions):
		left = region_lower_np[i, j]
		right = region_upper_np[i, j]
		ax.axvspan(left, right, color=region_color, alpha=0.15, lw=0)
		ax.axvline(left, color=region_color, linestyle="--", linewidth=1.0, alpha=0.7)
		ax.axvline(right, color=region_color, linestyle="--", linewidth=1.0, alpha=0.7)

	legend_extras: list = [
		mpatches.Patch(
			facecolor=to_rgba(region_color, alpha=0.15),
			edgecolor=region_color,
			linestyle="--",
			label="Robustness region",
		)
	]
	legend_extra_labels: list[str] = ["Robustness region"]

	if k_regions >= 2:
		# Explicit (on, off) dash pattern: matplotlib's built-in ``"--"``
		# scales with linewidth and reads as nearly solid at width 1.0.
		split_dashes = (0, (5, 3))
		for c_idx in range(len(region_order) - 1):
			left_idx = int(region_order[c_idx])
			right_idx = int(region_order[c_idx + 1])
			split_x = 0.5 * (
				region_upper_np[i, left_idx] + region_lower_np[i, right_idx]
			)
			ax.axvline(
				split_x,
				color=split_color,
				linestyle=split_dashes,
				linewidth=1.0,
				alpha=0.95,
				zorder=5,
			)
		legend_extras.append(
			plt.Line2D([0], [0], color=split_color, linewidth=1.0, linestyle=split_dashes, label="Voronoi cell split"),
		)
		legend_extra_labels.append("Voronoi cell split")

	handles, labels = ax.get_legend_handles_labels()
	handles.extend(legend_extras)
	labels.extend(legend_extra_labels)
	ax.legend(handles=handles, labels=labels, loc="upper right")

	ax.set_xlabel("Output", fontsize=10)
	if show_ylabel:
		ax.set_ylabel("Density", fontsize=10)
	ax.grid(alpha=0.25, linestyle=":")
	if xlim is not None:
		ax.set_xlim(*xlim)
	if ylim is not None:
		ax.set_ylim(*ylim)


def plot_combined_smoother_distributions(
	x: torch.Tensor,
	outputs: torch.Tensor,
	panels: list[tuple[torch.Tensor, Any, str, str]],
	output_path: Path,
	raw_bins: np.ndarray | int = 60,
	mc_bins: np.ndarray | int = 45,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
) -> Path:
	"""Render multiple smoothers side-by-side, sharing the original-model histogram.

	``panels`` is a list of ``(smoother_mc_samples, region, label, color)``
	tuples; one column per panel, each tagged ``(a)``, ``(b)``, ... in the
	top-left so they can be cross-referenced from the paper text.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if not panels:
		raise ValueError("plot_combined_smoother_distributions requires at least one panel")

	outputs_np = _to_numpy(outputs[:, :, 0])
	m = outputs_np.shape[0]
	if m != 1:
		raise NotImplementedError(
			"plot_combined_smoother_distributions currently supports a single input row"
		)
	n_panels = len(panels)

	_panel_w = 0.75 * TEXTWIDTH
	fig, axes = plt.subplots(
		1,
		n_panels,
		figsize=(_panel_w * n_panels, 0.55 * 0.80 * TEXTWIDTH),
		sharex=False,
		sharey=True,
	)
	if n_panels == 1:
		axes = [axes]

	for col, ((smoother_mc_samples, region, smoother_label, smoother_color), ax) in enumerate(
		zip(panels, axes)
	):
		smoother_mc_np = _to_numpy(smoother_mc_samples[:, :, 0])
		region_lower_np = _to_numpy(region.lower[:, :, 0])
		region_upper_np = _to_numpy(region.upper[:, :, 0])
		_draw_smoother_distribution_panel(
			ax,
			i=0,
			outputs=outputs,
			outputs_np=outputs_np,
			smoother_mc_np=smoother_mc_np,
			region=region,
			region_lower_np=region_lower_np,
			region_upper_np=region_upper_np,
			raw_bins=raw_bins,
			mc_bins=mc_bins,
			smoother_label=smoother_label,
			smoother_color=smoother_color,
			xlim=xlim,
			ylim=ylim,
			show_ylabel=(col == 0),
		)
		panel_tag = f"({chr(ord('a') + col)})"
		ax.text(
			0.025,
			0.955,
			panel_tag,
			transform=ax.transAxes,
			fontsize=10,
			fontweight="bold",
			va="top",
			ha="left",
		)

	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)
	return output_path


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


def plot_final_timestep_examples(
	x: torch.Tensor,
	original_mc_samples: torch.Tensor,
	noisy_mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clustered_mc_samples: torch.Tensor,
	single_cluster_centroids: torch.Tensor,
	clustered_centroids: torch.Tensor,
	region,
	single_cluster_region,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot final-step endpoint samples and certified rectangles in one chart.

	Columns are Original / Noisy / single-cluster smoother / clustered smoother.
	Certified final-step rectangles are overlaid on the smoother columns.
	"""

	output_dir.mkdir(parents=True, exist_ok=True)

	original = _reshape_trajectory_samples(original_mc_samples)
	noisy = _reshape_trajectory_samples(noisy_mc_samples)
	single = _reshape_trajectory_samples(single_cluster_mc_samples)
	clust = _reshape_trajectory_samples(clustered_mc_samples)
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

	# Hard-clip the view to [-10, 10] in both axes.
	x_lim = (-10.0, 10.0)
	y_lim = (-10.0, 10.0)
	x_extent = 10.0
	y_extent = 10.0

	col_titles = (r"$h_{w}(x)$", r"$h_{w}(x+\varepsilon)$", r"$\alpha$-smoothing", r"Clustered $\alpha$-smoothing")
	title_fontsize = 16
	axis_label_fontsize = 15
	tick_label_fontsize = 14
	major_ticks = np.arange(-10.0, 10.1, 5.0)
	# Map flat column index to (row, col) in the 1x4 grid
	_grid_pos = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3)}
	agent_colors = ("#008A2E",)
	agent_labels = ("Agent",)

	def _assign_to_centroids(points_np: np.ndarray, centroids_np: np.ndarray) -> np.ndarray:
		diff = points_np[:, None, :] - centroids_np[None, :, :]
		d2 = np.sum(diff * diff, axis=-1)
		return np.argmin(d2, axis=1)

	# Figure size suitable for a scientific article (full-width 2-column figure)
	_fig_w = 16.0
	_fig_h = 4.5

	chart_paths: list[Path] = []

	for i in range(nx):
		fig, axes = plt.subplots(1, 4, figsize=(_fig_w, _fig_h), sharex=True, sharey=True, constrained_layout=False)
		fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.12, hspace=0.0, wspace=0.05)
		fig.patch.set_facecolor("#FFFFFF")

		single_centroids_i = _to_numpy(single_cluster_centroids[i])
		clust_centroids_i = _to_numpy(clustered_centroids[i])

		column_data = (
			(original_end[i], None),
			(noisy_end[i], None),
			(single_end[i], single_centroids_i),
			(clust_end[i], clust_centroids_i),
		)

		for col_idx, (endpoints, centroids_ref) in enumerate(column_data):
			ax = axes[col_idx]
			ax.set_facecolor("#FFFFFF")
			_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)
			ax.set_aspect("equal", adjustable="box")

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
					s=10,
					alpha=0.45,
					color=color_j,
					linewidths=0,
					zorder=2,
				)

			# Overlay certified final-step rectangles on smoother columns.
			if col_idx == 2:
				sc_k = int(single_cluster_region.lower.shape[1])
				for k_c in range(sc_k):
					color = CLUSTER_PALETTE[k_c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = sc_lower_final[i, k_c]
					x_hi, y_hi = sc_upper_final[i, k_c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo),
						x_hi - x_lo,
						y_hi - y_lo,
						edgecolor=color,
						facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.9,
						linestyle="-",
						zorder=3,
					)
					ax.add_patch(rect)
			elif col_idx == 3:
				k = int(region.lower.shape[1])
				for k_c in range(k):
					color = CLUSTER_PALETTE[k_c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = region_lower_final[i, k_c]
					x_hi, y_hi = region_upper_final[i, k_c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo),
						x_hi - x_lo,
						y_hi - y_lo,
						edgecolor=color,
						facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.9,
						linestyle="-",
						zorder=3,
					)
					ax.add_patch(rect)

			agent_traj = x_traj[i, 0]
			color = agent_colors[0]

			next_idx = 1 if agent_traj.shape[0] > 1 else 0
			_plot_car_icon(
				ax,
				start=agent_traj[0],
				next_point=agent_traj[next_idx],
				color=color,
				car_length=1.55,
				alpha=0.3,
				edge_width=2.2,
				road_aligned=True,
			)

			ax.set_title(col_titles[col_idx], fontsize=title_fontsize)
			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_xticks(major_ticks)
			ax.set_yticks(major_ticks)
			ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

		# Suppress tick labels on shared edges to keep the grid seamless.
		for c in range(1, 4):
			axes[c].tick_params(axis="y", which="both", left=False, labelleft=False)

		# x-labels on all columns, y-label on left column only
		for c in range(4):
			axes[c].set_xlabel(r"$x_1$", fontsize=axis_label_fontsize)
		axes[0].set_ylabel(r"$x_2$", fontsize=axis_label_fontsize)

		# Legend only on the top-left panel of the first figure
		if i == 0:
			handles = [
				mpatches.Patch(facecolor=to_rgba(agent_colors[0], alpha=0.3), edgecolor=agent_colors[0], label=agent_labels[0]),
			]
			axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=10)

		fig_path = output_dir / f"final_timestep_examples_{i}.pdf"
		fig.savefig(fig_path, dpi=180)
		plt.close(fig)
		chart_paths.append(fig_path)

	return chart_paths


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
			color="#1A1A1A",
			linewidth=1.8,
			solid_capstyle="round",
			zorder=0,
		)

	for line in lines_dashed:
		ax.plot(
			line[:, 0],
			line[:, 1],
			color="#A2A2A2",
			linewidth=1.4,
			linestyle=(0, (4, 4)),
			solid_capstyle="round",
			zorder=0,
		)


def _plot_car_icon(
	ax: plt.Axes,
	start: np.ndarray,
	next_point: np.ndarray,
	color: str,
	car_length: float,
	alpha: float = 0.3,
	edge_width: float = 1.8,
	road_aligned: bool = False,
) -> None:
	"""Draw a simple rotated rectangle to indicate vehicle pose at trajectory start."""
	heading = float(np.arctan2(next_point[1] - start[1], next_point[0] - start[0]))
	if road_aligned:
		# Snap to nearest road axis: 0, pi/2, pi, -pi/2.
		axes_headings = np.array([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi])
		delta = np.arctan2(np.sin(heading - axes_headings), np.cos(heading - axes_headings))
		heading = float(axes_headings[np.argmin(np.abs(delta))])
	car_width = car_length * 0.45
	rect = mpatches.Rectangle(
		(start[0] - 0.5 * car_length, start[1] - 0.5 * car_width),
		car_length,
		car_width,
		facecolor=to_rgba(color, alpha=alpha),
		edgecolor=color,
		linewidth=edge_width,
		zorder=7,
		joinstyle="round",
	)
	rect.set_transform(mtransforms.Affine2D().rotate_around(start[0], start[1], heading) + ax.transData)
	ax.add_patch(rect)


def plot_bound_clustered_grid(
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
		figsize=(_w, 0.45 * _w),
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

			ax.grid(alpha=0.3, linestyle=":")
			ax.set_xticks([0.0, 0.5, 1.0])
			ax.set_yticks([0.0, 0.5, 1.0])

			if i == 0:
				ax.set_title(rf"$p_{{\mathcal{{V}}}}={p_v}$", fontsize=8)
			if j == len(p_v_values) - 1:
				twin = ax.twinx()
				twin.set_yticks([])
				twin.set_ylabel(
					rf"$\alpha={alpha}$",
					fontsize=8, rotation=270, labelpad=15,
				)

	supy = fig.supylabel("Certified probability", fontsize=8)
	supx = fig.supxlabel(r"Ratio $p_{\mathcal{R}} / p_{\mathcal{V}}$", fontsize=8)

	fig.tight_layout()

	bottom_ax_pos = axes[-1, 0].get_position()
	y_offset = 0.15
	supx.set_y(max(bottom_ax_pos.y0 - y_offset, 0.005))

	# Match visual distance horizontally given the figure aspect ratio.
	fig_w, fig_h = fig.get_size_inches()
	x_offset = y_offset * fig_h / fig_w
	supy.set_x(max(bottom_ax_pos.x0 - x_offset - 0.02, 0.005))

	top_ax_pos = axes[0, 0].get_position()
	supy.set_y((bottom_ax_pos.y0 + top_ax_pos.y1) / 2)

	middle_ax_pos = axes[len(alphas) // 2, -1].get_position()
	y_center = (middle_ax_pos.y0 + middle_ax_pos.y1) / 2

	handles, labels = axes[0, 0].get_legend_handles_labels()
	fig.legend(
		handles, labels,
		loc="center left",
		ncol=1,
		bbox_to_anchor=(1.0, y_center),
		bbox_transform=fig.transFigure,
		framealpha=0.95,
	)

	output_path = output_dir / "bound_clustered_grid.pdf"
	fig.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)

	return output_path


def _configure_trajflow_hist_plot_style() -> None:
	plt.rcParams.update(
		{
			"text.usetex": False,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 10,
			"axes.titlesize": 10,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
		}
	)


def _unit_interval_bins(min_v: float, max_v: float) -> np.ndarray:
	bin_width = 0.5
	left = np.floor(min_v / bin_width) * bin_width
	right = np.ceil(max_v / bin_width) * bin_width
	if np.isclose(left, right):
		right = left + bin_width
	return np.arange(left, right + bin_width, bin_width, dtype=float)


def _gaussian_kde_curve(data: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
	if len(data) < 2:
		return None
	std = float(np.std(data, ddof=1))
	if not np.isfinite(std) or std <= 0.0:
		return None
	bandwidth = 1.06 * std * (len(data) ** (-1.0 / 5.0))
	bandwidth = max(bandwidth, 1e-3)
	x_grid = np.linspace(float(bins[0]), float(bins[-1]), 400)
	diff = (x_grid[:, None] - data[None, :]) / bandwidth
	kernel = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
	density = kernel.mean(axis=1) / bandwidth
	return x_grid, density


def _plot_trajflow_metric_histogram(
	values: dict[str, np.ndarray],
	xlabel: str,
	output_path: Path,
	tail_quantile: float,
	adversarial_titles: bool,
) -> None:
	_configure_trajflow_hist_plot_style()

	colors = {
		"original": "#222222",
		"noisy_input": "#E4572E",
		"smoothed_k1": "#4C78A8",
		"smoothed_k2": "#2A9D8F",
	}
	mean_color = "#1B1B1B"
	tail_color = "#C73E1D"
	if adversarial_titles:
		titles = {
			"original": r"$h_w(x + \delta)$",
			"noisy_input": r"$h_w(x + \delta + \varepsilon)$",
			"smoothed_k1": r"$\alpha$-smoothing",
			"smoothed_k2": r"Clustered $\alpha$-smoothing",
		}
	else:
		titles = {
			"original": r"$h_w(x)$",
			"noisy_input": r"$h_w(x + \varepsilon)$",
			"smoothed_k1": r"$\alpha$-smoothing",
			"smoothed_k2": r"Clustered $\alpha$-smoothing",
		}

	ordered_keys = ("original", "noisy_input", "smoothed_k1", "smoothed_k2")
	clean_values = {key: values[key] for key in ordered_keys}
	all_values = np.concatenate([clean_values[key] for key in ordered_keys])
	min_v = float(np.min(all_values))
	max_v = float(np.max(all_values))
	bins = _unit_interval_bins(min_v, max_v)

	fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True, sharey=True)
	fig.subplots_adjust(left=0.1, right=0.98, bottom=0.11, top=0.95, hspace=0.28, wspace=0.16)

	for panel_index, (ax, key) in enumerate(zip(axes.flat, ordered_keys)):
		data = clean_values[key]
		ax.hist(
			data,
			bins=bins,
			density=True,
			alpha=0.35,
			linewidth=1.1,
			edgecolor=colors[key],
			histtype="stepfilled",
			color=colors[key],
		)
		kde_curve = _gaussian_kde_curve(data, bins)
		if kde_curve is not None:
			x_grid, density = kde_curve
			ax.plot(x_grid, density, color=colors[key], linewidth=1.5, alpha=0.4)
		mean_line = ax.axvline(
			float(np.mean(data)),
			color=mean_color,
			linestyle="--",
			linewidth=1.1,
			alpha=0.95,
			label="Mean",
		)
		cvar_q = float(np.percentile(data, tail_quantile * 100))
		cvar_tail = data[data >= cvar_q]
		cvar_value = float(cvar_tail.mean()) if len(cvar_tail) > 0 else cvar_q
		tail_line = ax.axvline(
			cvar_value,
			color=tail_color,
			linestyle="-.",
			linewidth=1.1,
			alpha=0.95,
			label=f"CVaR {tail_quantile:.0%}",
		)
		ax.set_title(titles[key])
		if panel_index >= 2:
			ax.set_xlabel(xlabel)
		if panel_index % 2 == 0:
			ax.set_ylabel("Density")
		ax.grid(True, linestyle=":", alpha=0.35)

		if panel_index == 0:
			ax.legend([mean_line, tail_line], ["Mean", f"CVaR {tail_quantile:.0%}"], loc="upper left", fontsize=8, frameon=True)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)


def plot_trajflow_wasserstein_histograms(
	w_noisy_vs_k1: np.ndarray,
	w_noisy_vs_k2: np.ndarray,
	output_path: Path,
	bins: int = 24,
	w_noisy_vs_rsreg: np.ndarray | None = None,
) -> Path:
	"""Plot Wasserstein histograms for RS-Reg (optional), k=1 and k=2 smoothers.

	When ``w_noisy_vs_rsreg`` is supplied the figure shows three side-by-side
	panels in the order RS-Reg → alpha-smoothing → clustered alpha-smoothing,
	sharing a common x-axis range across all three distributions.  When it is
	omitted the classic two-panel layout is produced.
	"""
	if bins <= 0:
		raise ValueError("bins must be positive")
	if len(w_noisy_vs_k1) == 0 or len(w_noisy_vs_k2) == 0:
		raise ValueError("Wasserstein arrays must be non-empty")

	output_path.parent.mkdir(parents=True, exist_ok=True)

	arrays_for_range = [w_noisy_vs_k1, w_noisy_vs_k2]
	if w_noisy_vs_rsreg is not None:
		arrays_for_range.append(w_noisy_vs_rsreg)
	combined = np.concatenate(arrays_for_range).astype(float)
	min_v = float(np.min(combined))
	max_v = float(np.max(combined))
	if np.isclose(min_v, max_v):
		eps = 1e-6 if abs(min_v) < 1e-3 else abs(min_v) * 1e-3
		bin_edges = np.linspace(min_v - eps, max_v + eps, bins + 1)
	else:
		bin_edges = np.linspace(min_v, max_v, bins + 1)

	with plt.rc_context(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 10,
			"axes.titlesize": 10,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
			"legend.fontsize": 9,
		}
	):
		if w_noisy_vs_rsreg is not None:
			fig, axes = plt.subplots(1, 3, figsize=(13.5, 2.4), sharey=True)
			fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.28, wspace=0.14)
			panels = (
				(
					axes[0],
					w_noisy_vs_rsreg.astype(float),
					r"RS-Reg",
					"#E76F51",
				),
				(
					axes[1],
					w_noisy_vs_k1.astype(float),
					r"$\alpha$-smoothing",
					"#4C78A8",
				),
				(
					axes[2],
					w_noisy_vs_k2.astype(float),
					r"Clustered $\alpha$-smoothing",
					"#2A9D8F",
				),
			)
		else:
			fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.4), sharey=True)
			fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.28, wspace=0.14)
			panels = (
				(
					axes[0],
					w_noisy_vs_k1.astype(float),
					r"$\alpha$-smoothing",
					"#4C78A8",
				),
				(
					axes[1],
					w_noisy_vs_k2.astype(float),
					r"Clustered $\alpha$-smoothing",
					"#2A9D8F",
				),
			)
		for ax, values, subtitle, color in panels:
			ax.set_title(subtitle, pad=8)
			ax.hist(
				values,
				bins=bin_edges,
				density=False,
				color=color,
				alpha=0.35,
				edgecolor=color,
				linewidth=1.0,
			)
			mean_value = float(np.mean(values))
			ax.axvline(
				mean_value,
				color="#202020",
				linestyle="--",
				linewidth=1.2,
				label=f"Mean: {mean_value:.1f}",
			)
			ax.set_xlabel("2-Wasserstein distance")
			ax.grid(True, linestyle=":", alpha=0.35)
			ax.legend(loc="upper right", frameon=True)

		axes[0].set_ylabel("Frequency")
		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	return output_path


def plot_trajectory_examples_clustered(
	x: torch.Tensor,
	original_mc_samples: torch.Tensor,
	noisy_mc_samples: torch.Tensor,
	single_cluster_mc_samples: torch.Tensor,
	clustered_mc_samples: torch.Tensor,
	single_cluster_centroids: torch.Tensor,
	clustered_centroids: torch.Tensor,
	region,
	single_cluster_region,
	output_dir: Path,
	num_input_agents: int = 2,
) -> list[Path]:
	"""Plot full-trajectory samples with cluster-colouring for the clustered smoother.

	Like ``plot_final_timestep_examples`` but uses trajectory lines instead of
	endpoint scatter plots.  Trajectories in columns 3-4 (smoothers) are coloured
	by their nearest centroid in the full output space, so the stay / go separation
	is visible even in high dimensions without relying on endpoint-only projection.

	Certified final-step rectangles are overlaid on the smoother columns.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)

	original = _reshape_trajectory_samples(original_mc_samples)
	noisy = _reshape_trajectory_samples(noisy_mc_samples)
	single = _reshape_trajectory_samples(single_cluster_mc_samples)
	clust = _reshape_trajectory_samples(clustered_mc_samples)
	x_np = _to_numpy(x)
	num_input_timesteps = x_np.shape[1] // (2 * num_input_agents)
	x_traj = x_np.reshape(x_np.shape[0], num_input_agents, num_input_timesteps, 2)

	if region.lower.shape[-1] % 2 != 0 or single_cluster_region.lower.shape[-1] % 2 != 0:
		raise ValueError("Expected region output dimension arranged as [x1,y1,...,xT,yT]")

	region_lower_final = _to_numpy(region.lower[..., -2:])
	region_upper_final = _to_numpy(region.upper[..., -2:])
	sc_lower_final = _to_numpy(single_cluster_region.lower[..., -2:])
	sc_upper_final = _to_numpy(single_cluster_region.upper[..., -2:])

	nx, num_samples, num_timesteps, _ = original.shape

	def _assign_by_centroid_flat(
		samples_flat: np.ndarray,
		centroids_flat: np.ndarray,
	) -> np.ndarray:
		"""Nearest-centroid assignment in the full output space (L2)."""
		diff = samples_flat[:, None, :] - centroids_flat[None, :, :]
		d2 = np.sum(diff * diff, axis=-1)
		return np.argmin(d2, axis=1)

	x_lim = (-10.0, 10.0)
	y_lim = (-10.0, 10.0)
	x_extent = 10.0
	y_extent = 10.0

	col_titles = (
		r"$h_{w}(x)$",
		r"$h_{w}(x+\varepsilon)$",
		r"$\alpha$-smoothing",
		r"Clustered $\alpha$-smoothing",
	)
	title_fontsize = 16
	axis_label_fontsize = 15
	tick_label_fontsize = 14
	major_ticks = np.arange(-10.0, 10.1, 5.0)
	agent_colors = ("#008A2E",)
	agent_labels = ("Agent",)
	_fig_w = 16.0
	_fig_h = 4.5
	chart_paths: list[Path] = []

	for i in range(nx):
		fig, axes = plt.subplots(
			1, 4, figsize=(_fig_w, _fig_h), sharex=True, sharey=True,
			constrained_layout=False,
		)
		fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.12, hspace=0.0, wspace=0.05)
		fig.patch.set_facecolor("#FFFFFF")

		single_centroids_flat = _to_numpy(single_cluster_centroids[i])  # (k1, out_dim)
		clust_centroids_flat = _to_numpy(clustered_centroids[i])  # (k2, out_dim)

		single_labels = _assign_by_centroid_flat(
			single[i].reshape(num_samples, -1), single_centroids_flat
		)
		clust_labels = _assign_by_centroid_flat(
			clust[i].reshape(num_samples, -1), clust_centroids_flat
		)

		columns = (
			(original[i], None),
			(noisy[i], None),
			(single[i], single_labels),
			(clust[i], clust_labels),
		)

		for col_idx, (traj_data, labels) in enumerate(columns):
			ax = axes[col_idx]
			ax.set_facecolor("#FFFFFF")
			_plot_street_marks(ax, x_extent=x_extent, y_extent=y_extent)
			ax.set_aspect("equal", adjustable="box")

			if labels is None:
				for j in range(num_samples):
					traj = traj_data[j]
					ax.plot(
						traj[:, 0], traj[:, 1],
						color=CLUSTER_PALETTE[0], alpha=0.10, linewidth=0.7, zorder=1,
					)
			else:
				for j in range(num_samples):
					c = int(labels[j])
					color = CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]
					traj = traj_data[j]
					ax.plot(
						traj[:, 0], traj[:, 1],
						color=color, alpha=0.15, linewidth=0.8, zorder=1,
					)

			# Overlay certified final-step rectangles on smoother columns.
			if col_idx == 2:
				sc_k = sc_lower_final.shape[1]
				for k_c in range(sc_k):
					color = CLUSTER_PALETTE[k_c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = sc_lower_final[i, k_c]
					x_hi, y_hi = sc_upper_final[i, k_c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
						edgecolor=color, facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.9, linestyle="-", zorder=3,
					)
					ax.add_patch(rect)
			elif col_idx == 3:
				k = region_lower_final.shape[1]
				for k_c in range(k):
					color = CLUSTER_PALETTE[k_c % len(CLUSTER_PALETTE)]
					x_lo, y_lo = region_lower_final[i, k_c]
					x_hi, y_hi = region_upper_final[i, k_c]
					rect = mpatches.Rectangle(
						(x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
						edgecolor=color, facecolor=to_rgba(color, alpha=0.12),
						linewidth=0.9, linestyle="-", zorder=3,
					)
					ax.add_patch(rect)

			agent_traj = x_traj[i, 0]
			color = agent_colors[0]
			next_idx = 1 if agent_traj.shape[0] > 1 else 0
			_plot_car_icon(
				ax,
				start=agent_traj[0],
				next_point=agent_traj[next_idx],
				color=color,
				car_length=1.55,
				alpha=0.3,
				edge_width=2.2,
				road_aligned=True,
			)

			ax.set_title(col_titles[col_idx], fontsize=title_fontsize)
			ax.set_xlim(*x_lim)
			ax.set_ylim(*y_lim)
			ax.set_xticks(major_ticks)
			ax.set_yticks(major_ticks)
			ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)

		for c in range(1, 4):
			axes[c].tick_params(axis="y", which="both", left=False, labelleft=False)
		for c in range(4):
			axes[c].set_xlabel(r"$x_1$", fontsize=axis_label_fontsize)
		axes[0].set_ylabel(r"$x_2$", fontsize=axis_label_fontsize)

		if i == 0:
			handles = [
				mpatches.Patch(
					facecolor=to_rgba(agent_colors[0], alpha=0.3),
					edgecolor=agent_colors[0],
					label=agent_labels[0],
				)
			]
			axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=10)

		fig_path = output_dir / f"trajectory_examples_clustered_{i}.pdf"
		fig.savefig(fig_path, dpi=180)
		plt.close(fig)
		chart_paths.append(fig_path)

	return chart_paths


# ---------------------------------------------------------------------------
# Per-timestep Wasserstein plot
# ---------------------------------------------------------------------------

def plot_wasserstein_per_timestep(
	w2_k1: np.ndarray,
	w2_k2: np.ndarray,
	output_path: Path,
	timesteps: np.ndarray | None = None,
	w2_rsreg: np.ndarray | None = None,
) -> Path:
	"""Plot per-timestep 2-Wasserstein distances with mean ± 1 std band.

	Args:
		w2_k1: Array of shape (num_inputs, num_eval_steps) – W2 per (input, step)
		        for the alpha-smoothing (k=1) smoother versus h_w(x).
		w2_k2: Same shape – W2 for clustered alpha-smoothing (k=2) versus h_w(x).
		output_path: Where to save the figure.
		timesteps: 1-D integer array of the actual 0-based prediction timestep
		           indices corresponding to the columns of w2_k1/w2_k2.  When
		           omitted, defaults to 0, 1, …, num_eval_steps-1.
		w2_rsreg: Optional array of the same shape as w2_k1 – W2 for RS-Reg
		          (alpha=0, k=1) versus h_w(x).  When supplied a third line is
		          drawn as a baseline, plotted below the other two lines.
	"""
	if w2_k1.ndim != 2 or w2_k2.ndim != 2:
		raise ValueError("w2_k1 and w2_k2 must be 2-D arrays (num_inputs, num_eval_steps)")
	if w2_k1.shape != w2_k2.shape:
		raise ValueError("w2_k1 and w2_k2 must have the same shape")
	if w2_rsreg is not None and w2_rsreg.shape != w2_k1.shape:
		raise ValueError("w2_rsreg must have the same shape as w2_k1 when provided")

	num_timesteps = w2_k1.shape[1]
	if timesteps is None:
		timesteps = np.arange(num_timesteps)
	timesteps = np.asarray(timesteps, dtype=int)
	mean_k1 = w2_k1.mean(axis=0)
	std_k1 = w2_k1.std(axis=0, ddof=0)
	mean_k2 = w2_k2.mean(axis=0)
	std_k2 = w2_k2.std(axis=0, ddof=0)
	if w2_rsreg is not None:
		mean_rsreg = w2_rsreg.mean(axis=0)
		std_rsreg = w2_rsreg.std(axis=0, ddof=0)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with plt.rc_context(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 10,
			"axes.titlesize": 10,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
			"legend.fontsize": 9,
		}
	):
		fig, ax = plt.subplots(figsize=(5.5, 3.2))

		if w2_rsreg is not None:
			ax.fill_between(
				timesteps, mean_rsreg - std_rsreg, mean_rsreg + std_rsreg,
				alpha=0.20, color="#E76F51",
			)
			ax.plot(
				timesteps, mean_rsreg,
				marker=".", linewidth=1.4, color="#E76F51",
				label=r"RS-Reg",
			)

		ax.fill_between(
			timesteps, mean_k1 - std_k1, mean_k1 + std_k1,
			alpha=0.20, color="#4C78A8",
		)
		ax.plot(
			timesteps, mean_k1,
			marker=".", linewidth=1.4, color="#4C78A8",
			label=r"$\alpha$-smoothing",
		)

		ax.fill_between(
			timesteps, mean_k2 - std_k2, mean_k2 + std_k2,
			alpha=0.20, color="#2A9D8F",
		)
		ax.plot(
			timesteps, mean_k2,
			marker=".", linewidth=1.4, color="#2A9D8F",
			label=r"Clustered $\alpha$-smoothing",
		)

		ax.set_xlabel(r"Prediction timestep $t$")
		ax.set_ylabel(r"Mean $W_2$ distance to $h_w(x)$")
		ax.set_xticks(timesteps)
		ax.grid(True, linestyle=":", alpha=0.35)
		ax.legend(loc="best", frameon=True)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	return output_path


def plot_wasserstein_trajectory_barchart(
	w2_traj_k1: np.ndarray,
	w2_traj_k2: np.ndarray,
	output_path: Path,
	w2_traj_rsreg: np.ndarray | None = None,
) -> Path:
	"""Bar chart of mean full-trajectory (R^d) 2-Wasserstein distances.

	Each bar shows the mean W2 over all evaluated inputs; error bars span ±1
	standard deviation.  The three smoothers are RS-Reg (optional),
	alpha-smoothing (k=1) and clustered alpha-smoothing (k=2).
	"""
	if len(w2_traj_k1) == 0 or len(w2_traj_k2) == 0:
		raise ValueError("Wasserstein arrays must be non-empty")

	labels: list[str] = []
	means: list[float] = []
	stds: list[float] = []
	colors_bar: list[str] = []

	if w2_traj_rsreg is not None:
		labels.append(r"RS-Reg")
		means.append(float(np.mean(w2_traj_rsreg)))
		stds.append(float(np.std(w2_traj_rsreg, ddof=0)))
		colors_bar.append("#E76F51")
	labels.append(r"$\alpha$-smoothing")
	means.append(float(np.mean(w2_traj_k1)))
	stds.append(float(np.std(w2_traj_k1, ddof=0)))
	colors_bar.append("#4C78A8")
	labels.append(r"Clustered $\alpha$-smoothing")
	means.append(float(np.mean(w2_traj_k2)))
	stds.append(float(np.std(w2_traj_k2, ddof=0)))
	colors_bar.append("#2A9D8F")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with plt.rc_context(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 10,
			"axes.titlesize": 10,
			"xtick.labelsize": 9,
			"ytick.labelsize": 9,
		}
	):
		fig, ax = plt.subplots(figsize=(5.0, 3.2))
		x = np.arange(len(labels))
		bars = ax.bar(
			x, means, yerr=stds,
			color=colors_bar, alpha=0.75, edgecolor="black", linewidth=0.8,
			capsize=5, error_kw={"linewidth": 1.0},
		)
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.set_ylabel(r"Mean $W_2$ distance to $h_w(x)$")
		ax.set_title(r"Full-trajectory $W_2$ ($\mathbf{R}^d$, mean $\pm$ std)")
		ax.grid(True, axis="y", linestyle=":", alpha=0.35)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	return output_path


# ---------------------------------------------------------------------------
# Bounds-only robustness plot
# ---------------------------------------------------------------------------

def plot_robustness_bounds_only(
	bounds_by_alpha: "dict[float, list[dict]]",
	output_path: Path,
) -> Path:
	"""Plot clustered alpha-smoothing certified bounds vs robustness radius.

	A single chart with one line per alpha value. Each line shows the mean
	per-input clustered certified lower bound (k=2) across all evaluated inputs;
	the shaded band spans ±1 std dev.

	Each row in the per-alpha list must have:
	    ``{"radius": float, "cert_k2_mean": float, "cert_k2_std": float}``
	"""
	alpha_values = sorted(bounds_by_alpha.keys())
	if not alpha_values:
		raise ValueError("bounds_by_alpha must be non-empty")

	colors  = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]
	markers = ["o", "s", "^", "D", "v"]

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with plt.rc_context(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 11,
			"axes.titlesize": 11,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"legend.fontsize": 10,
		}
	):
		fig, ax = plt.subplots(figsize=(5.0, 3.4))

		for i, alpha in enumerate(alpha_values):
			rows = sorted(bounds_by_alpha[alpha], key=lambda r: float(r["radius"]))
			radii = np.array([float(r["radius"]) for r in rows])
			cert_k2_mean = np.array([float(r["cert_k2_mean"]) for r in rows])
			cert_k2_std = np.array([float(r.get("cert_k2_std", 0.0)) for r in rows])
			color  = colors[i % len(colors)]
			marker = markers[i % len(markers)]

			ax.fill_between(
				radii,
				cert_k2_mean - cert_k2_std,
				cert_k2_mean + cert_k2_std,
				alpha=0.15, color=color,
			)
			ax.plot(
				radii, cert_k2_mean,
				marker=marker, linewidth=1.4, color=color,
				label=rf"$\alpha = {alpha:g}$",
			)

		ax.set_xlabel(r"Robustness radius $r$")
		ax.set_ylabel("Certified lower-bound")
		ax.grid(True, linestyle=":", alpha=0.35)
		ax.legend(loc="best", frameon=True)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	return output_path


# ---------------------------------------------------------------------------
# Safety analysis plot
# ---------------------------------------------------------------------------

def plot_safety_analysis(
	p_stay_k1: np.ndarray,
	p_stay_k2: np.ndarray,
	gt_stays: np.ndarray,
	threshold: float,
	output_path: Path,
	p_stay_rsreg: np.ndarray | None = None,
) -> Path:
	"""Plot decision-making safety analysis comparing smoothers.

	When ``p_stay_rsreg`` is supplied the figure shows four panels:
	[RS-Reg strip, alpha-smoothing strip, clustered alpha-smoothing strip, bar chart].
	Otherwise the classic three-panel layout is produced.

	For each scenario the ego vehicle decides to *go* when its predicted
	P(other stays) >= ``threshold``.  Risk occurs when the ego goes AND the
	other vehicle also goes in the ground truth.

	Args:
		p_stay_k1: P(other stays) predicted by alpha-smoothing (k=1).
		p_stay_k2: P(other stays) predicted by clustered smoothing (k=2).
		gt_stays: Boolean array, True when the other vehicle actually stays.
		threshold: Decision threshold; ego goes if predicted P(stays) >= threshold.
		output_path: Where to save the figure.
		p_stay_rsreg: Optional P(other stays) from RS-Reg (alpha=0, k=1).  When
		    supplied it is plotted as the leftmost strip panel.
	"""
	gt_stays = np.asarray(gt_stays, dtype=bool)
	gt_goes = ~gt_stays
	ego_go_k1 = p_stay_k1 >= threshold
	ego_go_k2 = p_stay_k2 >= threshold
	risk_k1 = float(np.mean(ego_go_k1 & gt_goes))
	risk_k2 = float(np.mean(ego_go_k2 & gt_goes))

	rng = np.random.default_rng(0)
	jitter = rng.uniform(-0.12, 0.12, len(p_stay_k1))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with plt.rc_context(
		{
			"text.usetex": True,
			"font.family": "serif",
			"font.serif": ["Computer Modern Roman"],
			"mathtext.fontset": "cm",
			"axes.labelsize": 13,
			"axes.titlesize": 13,
			"xtick.labelsize": 12,
			"ytick.labelsize": 12,
			"legend.fontsize": 12,
		}
	):
		if p_stay_rsreg is not None:
			nego_go_rsreg = p_stay_rsreg >= threshold
			risk_rsreg = float(np.mean(nego_go_rsreg & gt_goes))
			fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.45]})
			fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.20, wspace=0.12)
			strip_specs = [
				(axes[0], p_stay_rsreg, r"RS-Reg", "#6A3D9A", risk_rsreg),
				(axes[1], p_stay_k1, r"$\alpha$-smoothing", "#4C78A8", risk_k1),
				(axes[2], p_stay_k2, r"Clustered $\alpha$-smoothing", "#F58518", risk_k2),
			]
			ax_bar = axes[3]
			methods = [r"RS-Reg", r"$\alpha$-smooth.", r"Clustered $\alpha$-smooth."]
			risks = [risk_rsreg, risk_k1, risk_k2]
			bar_colors = ["#6A3D9A", "#4C78A8", "#F58518"]
		else:
			fig, axes = plt.subplots(1, 3, figsize=(10.9, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})
			fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.20, wspace=0.14)
			strip_specs = [
				(axes[0], p_stay_k1, r"$\alpha$-smoothing", "#4C78A8", risk_k1),
				(axes[1], p_stay_k2, r"Clustered $\alpha$-smoothing", "#F58518", risk_k2),
			]
			ax_bar = axes[2]
			methods = [r"$\alpha$-smooth.", r"Clustered $\alpha$-smooth."]
			risks = [risk_k1, risk_k2]
			bar_colors = ["#4C78A8", "#F58518"]

		for strip_idx, (ax, p_stay, title, color, risk) in enumerate(strip_specs):
			gt_x = np.where(gt_stays, 0.0, 1.0)
			# Keep scatter colours fixed across methods for comparability.
			point_colors = np.where(gt_stays, "#4C78A8", "#C73E1D")
			ax.scatter(gt_x + jitter, p_stay, c=point_colors, alpha=0.55, s=20, linewidths=0, zorder=2)
			ax.axhline(
				threshold, color="#333333", linestyle="--", linewidth=1.1, zorder=3,
				label=rf"Threshold $= {threshold}$",
			)
			# Shade the risky quadrant: ego goes (p_stay >= threshold) AND other goes (right half)
			ax.axhspan(threshold, 1.05, xmin=0.5, color="#C73E1D", alpha=0.08, zorder=0)
			ax.set_xticks([0.0, 1.0])
			ax.set_xticklabels(["Vehicle\nStays", "Vehicle\nGoes"])
			if strip_idx == 0:
				ax.set_ylabel(r"$P(\mbox{vehicle stays})$")
			else:
				ax.tick_params(axis="y", labelleft=False)
			ax.set_ylim(-0.05, 1.05)
			ax.set_title(title)
			ax.grid(True, linestyle=":", alpha=0.35)

		bars = ax_bar.bar(methods, risks, color=bar_colors, alpha=0.75, edgecolor="black", linewidth=0.8)
		ax_bar.set_ylabel("")
		ax_bar.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
		ax_bar.tick_params(axis="x", labelsize=10)
		ylim_top = min(1.0, max(max(risks, default=0.1) * 1.35, 0.1))
		ax_bar.set_ylim(0.0, ylim_top)
		for bar, risk in zip(bars, risks):
			risk_pct = f"{risk * 100:.1f}\\%"
			ax_bar.text(
				bar.get_x() + bar.get_width() / 2,
				bar.get_height() + 0.005,
				risk_pct,
				ha="center", va="bottom", fontsize=11,
			)
		ax_bar.grid(True, linestyle=":", alpha=0.35, axis="y")
		ax_bar.set_title("Risk rate")

		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	return output_path

