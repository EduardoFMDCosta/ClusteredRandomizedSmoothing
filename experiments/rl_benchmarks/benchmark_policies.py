import argparse
from pathlib import Path

import numpy as np
import torch
from dynamics import UAV3D
from train_rl import VariationalPolicy, deterministic_action, normalize_state


def evaluate_returns(action_fn, episodes: int, horizon: int, seed: int, env) -> np.ndarray:
    np.random.seed(seed)
    returns = []

    for _ in range(episodes):
        state = env.reset(curriculum_frac=1.0)
        total = 0.0
        for _ in range(horizon):
            action = action_fn(state)
            _, reward, state = env.step(action)
            total += float(reward)
            if hasattr(env, 'is_done') and env.is_done():
                break
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


def load_policy(checkpoint_path: Path) -> tuple[VariationalPolicy, str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with model_state_dict (legacy checkpoints are not supported).")

    required_fields = ["state_dim", "action_dim", "hidden_dim", "prior_std", "env_type", "action_scale"]
    missing_fields = [field for field in required_fields if field not in checkpoint]
    if missing_fields:
        raise ValueError(f"Checkpoint missing required fields: {missing_fields}")

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    prior_std = float(checkpoint["prior_std"])
    num_components = int(checkpoint.get("num_components", 1))
    env_type = str(checkpoint["env_type"])
    action_scale = float(checkpoint["action_scale"])
    state_dict = checkpoint["model_state_dict"]

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        prior_std=prior_std,
        action_scale=action_scale,
        num_components=num_components,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, env_type, action_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trained policy against baselines.")
    parser.add_argument("--env-type", type=str, default="uav_3d", choices=["uav_3d"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = f"experiments/{args.env_type}/policy.pt"

    if args.env_type == "uav_3d":
        env = UAV3D(gaussian_noise=True, noise_std=0.05)
        action_dim = 3
        action_min, action_max = float(env.u_min), float(env.u_max)
    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")

    def random_action(_state: np.ndarray) -> np.ndarray:
        return np.random.uniform(action_min, action_max, size=(action_dim,)).astype(np.float32)

    def zero_action(_state: np.ndarray) -> np.ndarray:
        return np.array([0.0] * action_dim, dtype=np.float32)

    def forward_action(_state: np.ndarray) -> np.ndarray:
        action = np.array([0.0] * action_dim, dtype=np.float32)
        action[0] = action_max
        return action

    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    policy, checkpoint_env_type, action_scale = load_policy(checkpoint)

    if checkpoint_env_type != args.env_type:
        print(f"Warning: Checkpoint was trained on {checkpoint_env_type} but evaluating on {args.env_type}")

    def trained_action(state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(normalize_state(state, args.env_type), dtype=torch.float32).unsqueeze(0)
        return deterministic_action(
            policy,
            state_t,
            mc_samples=args.mc_samples,
            env_type=args.env_type,
            action_scale=action_scale,
        )

    print(f"Benchmarking {args.env_type} with episodes={args.episodes}, horizon={args.horizon}, seed={args.seed}")
    print(summarize("trained", evaluate_returns(trained_action, args.episodes, args.horizon, args.seed, env)))
    print(summarize("random_uniform", evaluate_returns(random_action, args.episodes, args.horizon, args.seed, env)))
    print(summarize("zero_action", evaluate_returns(zero_action, args.episodes, args.horizon, args.seed, env)))
    print(summarize("forward_action", evaluate_returns(forward_action, args.episodes, args.horizon, args.seed, env)))


if __name__ == "__main__":
    main()
