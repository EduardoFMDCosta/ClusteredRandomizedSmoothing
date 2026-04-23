import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from dynamics import UAV3D
from torch.distributions import Normal
from train_vi_bnn_rl import VariationalPolicy, normalize_state

ActionSampler = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_policy(checkpoint_path: Path) -> tuple[VariationalPolicy, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise ValueError("Expected checkpoint dict with model_state_dict.")

    state_dim = int(checkpoint.get("state_dim", 6))
    action_dim = int(checkpoint.get("action_dim", 3))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    prior_std = float(checkpoint.get("prior_std", 1.0))
    action_scale = float(checkpoint.get("action_scale", 4.0))
    num_components = int(checkpoint.get("num_components", 1))
    env_type = checkpoint.get("env_type", "uav_3d")
    model_state_dict = checkpoint["model_state_dict"]

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        prior_std=prior_std,
        action_scale=action_scale,
        num_components=num_components,
    )
    policy.load_state_dict(model_state_dict)
    policy.eval()
    return policy, env_type


def policy_uncertainty_grid(policy: VariationalPolicy, env_type: str = "uav_3d", num_samples: int = 80):
    if env_type != "uav_3d":
        raise ValueError(f"Unsupported env_type for uncertainty grid: {env_type}")
    # Project UAV3D uncertainty on the x-y plane at a fixed z slice.
    xs = np.linspace(-14.0, 15.0, 12)
    ys = np.linspace(-9.0, 9.0, 10)
    grid_h, grid_w = len(ys), len(xs)

    epistemic = np.zeros((grid_h, grid_w), dtype=np.float32)
    aleatoric = np.zeros((grid_h, grid_w), dtype=np.float32)

    with torch.no_grad():
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                state = np.array([x, 0.0, y, 0.0, -6.0, 0.0], dtype=np.float32)
                state_t = torch.tensor(normalize_state(state, env_type), dtype=torch.float32).unsqueeze(0)

                # Component-level parameters: exact expected mean & variance
                # of the mixture can be computed without Monte Carlo.
                component_means, component_log_stds, log_mix_w = policy.distribution_params(state_t)
                component_means = (torch.tanh(component_means) * policy.action_scale).squeeze(0).cpu().numpy()  # [K, A]
                component_vars = (torch.exp(component_log_stds) ** 2).squeeze(0).cpu().numpy()  # [K, A]
                mix_w = log_mix_w.exp().squeeze(0).cpu().numpy()  # [K]

                mixture_mean = (mix_w[:, None] * component_means).sum(axis=0)  # [A]
                # Epistemic-ish: spread of component means weighted by gate.
                epistemic[yi, xi] = float(
                    (mix_w[:, None] * (component_means - mixture_mean) ** 2).sum(axis=0).mean()
                )
                # Aleatoric: gate-weighted expected within-component variance.
                aleatoric[yi, xi] = float((mix_w[:, None] * component_vars).sum(axis=0).mean())

    return xs, ys, epistemic, aleatoric


def sample_action_and_next_state_distributions(
    action_sampler: ActionSampler,
    env,
    state: np.ndarray,
    samples_per_state: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    state_dim = len(state)

    first_action = action_sampler(state, rng)
    action_dim = int(first_action.shape[0])

    actions = np.zeros((samples_per_state, action_dim), dtype=np.float32)
    next_states = np.zeros((samples_per_state, state_dim), dtype=np.float32)

    actions[0] = first_action.astype(np.float32)
    np_state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    env._state = state.copy()
    _, _, next_state = env.step(first_action)
    np.random.set_state(np_state)
    next_states[0] = next_state[:state_dim].astype(np.float32)

    for i in range(1, samples_per_state):
        action = action_sampler(state, rng)
        actions[i] = action

        np_state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        env._state = state.copy()
        _, _, next_state = env.step(action)
        np.random.set_state(np_state)

        next_states[i] = next_state[:state_dim].astype(np.float32)

    return actions, next_states


def _plot_distribution_grid(
    data_per_state: list[np.ndarray],
    states: list[np.ndarray],
    grid_x: int,
    grid_y: int,
    bins: int,
    value_range: list[list[float]],
    title: str,
    x_label: str,
    y_label: str,
    cmap: str,
    out_path: Path,
    show_state_marker: bool = False,
) -> None:
    fig, axes = plt.subplots(
        grid_y,
        grid_x,
        figsize=(2.7 * grid_x, 2.7 * grid_y),
        sharex=True,
        sharey=True,
    )
    if isinstance(axes, np.ndarray):
        axes_flat = axes.reshape(-1)
    else:
        axes_flat = np.array([axes])

    image = None
    for idx, ax in enumerate(axes_flat):
        state = states[idx]
        samples = data_per_state[idx]

        _, _, _, image = ax.hist2d(
            samples[:, 0],
            samples[:, 1],
            bins=bins,
            range=value_range,
            cmap=cmap,
            cmin=1,
        )
        ax.set_title(f"s=({state[0]:.2f}, {state[1]:.2f})", fontsize=8)
        ax.set_xlim(value_range[0][0], value_range[0][1])
        ax.set_ylim(value_range[1][0], value_range[1][1])
        ax.tick_params(axis="both", labelsize=7)

        if show_state_marker:
            ax.scatter(state[0], state[1], marker="x", color="white", s=20, linewidths=1.0)

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.suptitle(title)

    fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.97))
    if image is not None:
        cbar_ax = fig.add_axes((0.92, 0.12, 0.015, 0.76))
        fig.colorbar(image, cax=cbar_ax, label="Sample count")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved state-grid distribution visualization to {out_path}")


def plot_state_grid_distributions(
    action_sampler: ActionSampler,
    out_dir: Path,
    prefix: str,
    grid_x: int,
    grid_y: int,
    samples_per_state: int,
    bins: int,
    seed: int,
    env_type: str = "uav_3d",
) -> None:
    if env_type != "uav_3d":
        raise ValueError(f"Unsupported env_type: {env_type}")

    xs = np.linspace(-14.0, 15.0, grid_x, dtype=np.float32)
    ys = np.linspace(-9.0, 9.0, grid_y, dtype=np.float32)
    states = [np.array([x, 0.0, y, 0.0, -6.0, 0.0], dtype=np.float32) for y in ys for x in xs]
    env = UAV3D(gaussian_noise=True, noise_std=0.05)
    next_state_range = [[-15.0, 16.0], [-10.0, 10.0]]
    next_state_labels = ("next_x", "next_y")
    action_range = [[-4.0, 4.0], [-4.0, 4.0]]
    action_labels = ("action_fx", "action_fy")
    action_title = "Policy Action Distribution Per State (x-y force projection)"

    rng = np.random.default_rng(seed)
    action_data = []
    next_state_data = []
    for state in states:
        actions, next_states = sample_action_and_next_state_distributions(
            action_sampler=action_sampler,
            env=env,
            state=state,
            samples_per_state=samples_per_state,
            rng=rng,
        )
        action_data.append(actions)
        next_state_data.append(next_states)

    action_out = out_dir / f"{prefix}_action_distributions.png"
    next_state_out = out_dir / f"{prefix}_next_state_distributions.png"

    display_states = [np.array([s[0], s[2]], dtype=np.float32) for s in states]
    action_data = [a[:, :2] for a in action_data]

    _plot_distribution_grid(
        data_per_state=action_data,
        states=display_states,
        grid_x=grid_x,
        grid_y=grid_y,
        bins=bins,
        value_range=action_range,
        title=(
            f"{env_type.upper()} {action_title} "
            f"({samples_per_state} samples/state, grid={grid_x}x{grid_y})"
        ),
        x_label=action_labels[0],
        y_label=action_labels[1],
        cmap="plasma",
        out_path=action_out,
    )

    next_state_data = [ns[:, [0, 2]] for ns in next_state_data]
    _plot_distribution_grid(
        data_per_state=next_state_data,
        states=display_states,
        grid_x=grid_x,
        grid_y=grid_y,
        bins=bins,
        value_range=next_state_range,
        title=(
            f"{env_type.upper()} Next-State Distribution Per State "
            f"({samples_per_state} samples/state, grid={grid_x}x{grid_y})"
        ),
        x_label=next_state_labels[0],
        y_label=next_state_labels[1],
        cmap="viridis",
        out_path=next_state_out,
        show_state_marker=True,
    )


def plot_training_and_uncertainty(
    metrics_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    ma_window: int,
    mc_samples: int,
    env_type: str = "uav_3d",
    policy: VariationalPolicy | None = None,
) -> None:
    with metrics_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)

    episodes = np.array(metrics["episode"], dtype=np.int32)
    ep_returns = np.array(metrics["episode_return"], dtype=np.float32)
    eval_returns = np.array(metrics["eval_return"], dtype=np.float32)
    goal_rate = np.array(metrics.get("goal_rate", [float("nan")] * len(episodes)), dtype=np.float32)
    eval_goal = np.array(metrics.get("eval_goal_rate", [float("nan")] * len(episodes)), dtype=np.float32)

    valid_eval_mask = ~np.isnan(eval_returns)
    valid_eval_goal_mask = ~np.isnan(eval_goal)

    if policy is None:
        policy, _ = load_policy(checkpoint_path)
    xs, ys, epistemic, aleatoric = policy_uncertainty_grid(policy, env_type, num_samples=mc_samples)
    extent = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.8, 1])

    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(episodes, ep_returns, color="#1F4E79", alpha=0.3, label="Train batch return (mean)")
    ax_curve.plot(
        episodes,
        moving_average(ep_returns, ma_window),
        color="#D1495B",
        linewidth=2.5,
        label=f"Moving average (window={ma_window})",
    )
    if np.any(valid_eval_mask):
        ax_curve.plot(
            episodes[valid_eval_mask],
            eval_returns[valid_eval_mask],
            marker="o",
            markersize=4,
            linestyle="--",
            color="#2A9D8F",
            label="Periodic evaluation return",
        )
    ax_curve.set_title(f"{env_type.upper()} VI-BNN PPO Training")
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Return")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best")

    ax_goal = fig.add_subplot(gs[1, :])
    ax_goal.plot(episodes, goal_rate, color="#1F4E79", alpha=0.3, label="Train batch goal rate")
    ax_goal.plot(
        episodes,
        moving_average(goal_rate, ma_window),
        color="#D1495B",
        linewidth=2.5,
        label=f"Moving average (window={ma_window})",
    )
    if np.any(valid_eval_goal_mask):
        ax_goal.plot(
            episodes[valid_eval_goal_mask],
            eval_goal[valid_eval_goal_mask],
            marker="o",
            markersize=4,
            linestyle="--",
            color="#2A9D8F",
            label="Eval goal rate",
        )
    ax_goal.set_ylim(-0.05, 1.05)
    ax_goal.set_xlabel("Episode")
    ax_goal.set_ylabel("Goal rate")
    ax_goal.grid(alpha=0.25)
    ax_goal.legend(loc="best")

    ax_epi = fig.add_subplot(gs[2, 0])
    im1 = ax_epi.imshow(
        epistemic,
        origin="lower",
        cmap="magma",
        extent=extent,
        aspect="auto",
    )
    ax_epi.set_title("Epistemic Uncertainty (Action Mean Variance)")
    ax_epi.set_xlabel("x-position")
    ax_epi.set_ylabel("y-position")
    fig.colorbar(im1, ax=ax_epi, fraction=0.046, pad=0.04)

    ax_ale = fig.add_subplot(gs[2, 1])
    im2 = ax_ale.imshow(
        aleatoric,
        origin="lower",
        cmap="viridis",
        extent=extent,
        aspect="auto",
    )
    ax_ale.set_title("Aleatoric Uncertainty (Policy Noise Variance)")
    ax_ale.set_xlabel("x-position")
    ax_ale.set_ylabel("y-position")
    fig.colorbar(im2, ax=ax_ale, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    print(f"Saved visualization to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VI-BNN training metrics and uncertainty.")
    parser.add_argument("--env-type", type=str, default="uav_3d", choices=["uav_3d"])
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=80)
    parser.add_argument(
        "--policy-source",
        type=str,
        default="vi_bnn",
        choices=["vi_bnn", "random_uniform", "zero_action", "forward_action"],
    )
    parser.add_argument("--grid-x", type=int, default=5)
    parser.add_argument("--grid-y", type=int, default=5)
    parser.add_argument("--samples-per-state", type=int, default=1200)
    parser.add_argument("--grid-bins", type=int, default=26)
    parser.add_argument("--grid-seed", type=int, default=123)
    parser.add_argument("--distribution-prefix", type=str, default="vi_bnn_state_grid")
    return parser.parse_args()


def build_action_sampler(policy_source: str, policy: VariationalPolicy | None, env_type: str) -> ActionSampler:
    if policy_source == "vi_bnn":
        if policy is None:
            raise ValueError("A VI-BNN policy is required when policy_source='vi_bnn'.")

        def vi_bnn_action(state: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
            state_t = torch.tensor(normalize_state(state, env_type), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _raw, _log_prob = policy.sample(state_t)
            return action.squeeze(0).cpu().numpy().astype(np.float32)

        return vi_bnn_action

    if policy_source == "random_uniform":
        def random_action(_state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return rng.uniform(-4.0, 4.0, size=(3,)).astype(np.float32)
        return random_action

    if policy_source == "zero_action":
        def zero_action(_state: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return zero_action

    if policy_source == "forward_action":
        def forward_action(_state: np.ndarray, _rng: np.random.Generator) -> np.ndarray:
            return np.array([4.0, 0.0, 0.0], dtype=np.float32)
        return forward_action

    raise ValueError(f"Unsupported policy_source: {policy_source}")


def main() -> None:
    args = parse_args()

    if args.metrics_path is None:
        args.metrics_path = f"experiments/{args.env_type}/vi_bnn_training_metrics.json"
    if args.checkpoint_path is None:
        args.checkpoint_path = f"experiments/{args.env_type}/vi_bnn_policy.pt"
    if args.out_path is None:
        args.out_path = f"experiments/{args.env_type}/vi_bnn_training_visualization.png"

    policy: VariationalPolicy | None = None
    checkpoint_env_type = args.env_type

    if args.policy_source == "vi_bnn":
        policy, checkpoint_env_type = load_policy(Path(args.checkpoint_path))
        if checkpoint_env_type != args.env_type:
            print(f"Warning: Checkpoint was trained on {checkpoint_env_type} but visualizing for {args.env_type}")
        plot_training_and_uncertainty(
            metrics_path=Path(args.metrics_path),
            checkpoint_path=Path(args.checkpoint_path),
            out_path=Path(args.out_path),
            ma_window=args.ma_window,
            mc_samples=args.mc_samples,
            env_type=checkpoint_env_type,
            policy=policy,
        )
    else:
        print(
            "Skipping training/uncertainty curves because --policy-source is not 'vi_bnn'. "
            "Generating state-grid distribution visualizations only."
        )

    action_sampler = build_action_sampler(args.policy_source, policy, checkpoint_env_type)
    plot_state_grid_distributions(
        action_sampler=action_sampler,
        out_dir=Path(args.out_path).parent,
        prefix=args.distribution_prefix,
        grid_x=max(1, int(args.grid_x)),
        grid_y=max(1, int(args.grid_y)),
        samples_per_state=max(10, int(args.samples_per_state)),
        bins=max(6, int(args.grid_bins)),
        seed=args.grid_seed,
        env_type=checkpoint_env_type,
    )


if __name__ == "__main__":
    main()
