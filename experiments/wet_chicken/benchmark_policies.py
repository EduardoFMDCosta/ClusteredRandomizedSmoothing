import argparse
from pathlib import Path

import numpy as np
import torch

from dynamics import LENGTH, MAX_TURBULENCE, MAX_VELOCITY, WIDTH, WetChicken
from train_vi_bnn_rl import VariationalPolicy, deterministic_action, normalize_state


def evaluate_returns(action_fn, episodes: int, horizon: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    env = WetChicken(LENGTH, WIDTH, MAX_TURBULENCE, MAX_VELOCITY)
    returns = []

    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(horizon):
            action = action_fn(state)
            _, reward, state = env.step(action)
            total += float(reward)
        returns.append(total)

    return np.array(returns, dtype=np.float64)


def summarize(name: str, returns: np.ndarray) -> str:
    return (
        f"{name:20s} "
        f"mean={returns.mean():7.3f} "
        f"std={returns.std():7.3f} "
        f"p10={np.percentile(returns, 10):7.3f} "
        f"p50={np.percentile(returns, 50):7.3f} "
        f"p90={np.percentile(returns, 90):7.3f}"
    )


def load_policy(checkpoint_path: Path) -> VariationalPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        prior_std = float(checkpoint.get("prior_std", 0.8))
        state_dependent_std = bool(checkpoint.get("state_dependent_std", True))
        state_dict = checkpoint["model_state_dict"]
    else:
        hidden_dim = 128
        prior_std = 0.8
        state_dependent_std = "out_log_std.weight_mu" in checkpoint
        state_dict = checkpoint

    policy = VariationalPolicy(
        hidden_dim=hidden_dim,
        prior_std=prior_std,
        state_dependent_std=state_dependent_std,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trained Wet Chicken policy against baselines.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--checkpoint-path", type=str, default="experiments/wet_chicken/vi_bnn_policy.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def random_action(_state: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    def zero_action(_state: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=np.float32)

    def forward_action(_state: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy = load_policy(checkpoint)

    def trained_action(state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)
        return deterministic_action(policy, state_t, mc_samples=args.mc_samples)

    print(f"Benchmarking with episodes={args.episodes}, horizon={args.horizon}, seed={args.seed}")
    print(summarize("trained_vi_bnn", evaluate_returns(trained_action, args.episodes, args.horizon, args.seed)))
    print(summarize("random_uniform", evaluate_returns(random_action, args.episodes, args.horizon, args.seed)))
    print(summarize("zero_action", evaluate_returns(zero_action, args.episodes, args.horizon, args.seed)))
    print(summarize("forward_action", evaluate_returns(forward_action, args.episodes, args.horizon, args.seed)))


if __name__ == "__main__":
    main()
