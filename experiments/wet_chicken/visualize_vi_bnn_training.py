import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamics import LENGTH, WIDTH
from train_vi_bnn_rl import VariationalPolicy, normalize_state


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_policy(checkpoint_path: Path) -> VariationalPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        hidden_dim = int(checkpoint.get("hidden_dim", 64))
        prior_std = float(checkpoint.get("prior_std", 1.0))
        model_state_dict = checkpoint["model_state_dict"]
        if "state_dependent_std" in checkpoint:
            state_dependent_std = bool(checkpoint["state_dependent_std"])
        else:
            state_dependent_std = "out_log_std.weight_mu" in model_state_dict
    else:
        hidden_dim = 64
        prior_std = 1.0
        state_dependent_std = "out_log_std.weight_mu" in checkpoint
        model_state_dict = checkpoint

    policy = VariationalPolicy(
        hidden_dim=hidden_dim,
        prior_std=prior_std,
        state_dependent_std=state_dependent_std,
    )
    policy.load_state_dict(model_state_dict)
    policy.eval()
    return policy


def policy_uncertainty_grid(policy: VariationalPolicy, num_samples: int = 80):
    xs = np.linspace(0.0, LENGTH - 1, LENGTH)
    ys = np.linspace(0.0, WIDTH - 1, WIDTH)

    epistemic = np.zeros((WIDTH, LENGTH), dtype=np.float32)
    aleatoric = np.zeros((WIDTH, LENGTH), dtype=np.float32)

    with torch.no_grad():
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                state = np.array([x, y], dtype=np.float32)
                state_t = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)

                means = []
                variances = []
                for _ in range(num_samples):
                    mean, std = policy(state_t)
                    means.append(torch.tanh(mean).squeeze(0).cpu().numpy())
                    variances.append((std.squeeze(0) ** 2).cpu().numpy())

                means_arr = np.stack(means, axis=0)
                vars_arr = np.stack(variances, axis=0)

                # Epistemic uncertainty: variation in action means from posterior weight sampling.
                epistemic[yi, xi] = float(np.mean(np.var(means_arr, axis=0)))
                # Aleatoric uncertainty: expected policy noise variance.
                aleatoric[yi, xi] = float(np.mean(np.mean(vars_arr, axis=0)))

    return xs, ys, epistemic, aleatoric


def plot_training_and_uncertainty(
    metrics_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    ma_window: int,
    mc_samples: int,
) -> None:
    with metrics_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)

    episodes = np.array(metrics["episode"], dtype=np.int32)
    ep_returns = np.array(metrics["episode_return"], dtype=np.float32)
    eval_returns = np.array(metrics["eval_return"], dtype=np.float32)

    valid_eval_mask = ~np.isnan(eval_returns)

    policy = load_policy(checkpoint_path)
    xs, ys, epistemic, aleatoric = policy_uncertainty_grid(policy, num_samples=mc_samples)
    extent = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(episodes, ep_returns, color="#1F4E79", alpha=0.3, label="Episode return")
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
    ax_curve.set_title("Wet Chicken VI-BNN Training")
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Return")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best")

    ax_epi = fig.add_subplot(gs[1, 0])
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

    ax_ale = fig.add_subplot(gs[1, 1])
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
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="experiments/wet_chicken/vi_bnn_training_metrics.json",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="experiments/wet_chicken/vi_bnn_policy.pt",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/wet_chicken/vi_bnn_training_visualization.png",
    )
    parser.add_argument("--ma-window", type=int, default=20)
    parser.add_argument("--mc-samples", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_training_and_uncertainty(
        metrics_path=Path(args.metrics_path),
        checkpoint_path=Path(args.checkpoint_path),
        out_path=Path(args.out_path),
        ma_window=args.ma_window,
        mc_samples=args.mc_samples,
    )


if __name__ == "__main__":
    main()
