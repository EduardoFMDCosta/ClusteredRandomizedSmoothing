import argparse
import json
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from dynamics import LENGTH, MAX_TURBULENCE, MAX_VELOCITY, WIDTH, WetChicken


EPS = 1e-6


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    def _sample(self, mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = softplus(rho)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return torch.nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        weight_sigma = softplus(self.weight_rho)
        bias_sigma = softplus(self.bias_rho)

        prior_var = self.prior_std ** 2
        weight_var = weight_sigma ** 2
        bias_var = bias_sigma ** 2

        weight_kl = 0.5 * (
            (weight_var + self.weight_mu ** 2) / prior_var - 1.0 + 2.0 * torch.log(self.prior_std / weight_sigma)
        ).sum()
        bias_kl = 0.5 * (
            (bias_var + self.bias_mu ** 2) / prior_var - 1.0 + 2.0 * torch.log(self.prior_std / bias_sigma)
        ).sum()
        return weight_kl + bias_kl


class VariationalPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 64, prior_std: float = 1.0, state_dependent_std: bool = True):
        super().__init__()
        self.fc1 = BayesianLinear(2, hidden_dim, prior_std=prior_std)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std)
        self.state_dependent_std = state_dependent_std
        self.out_mean = BayesianLinear(hidden_dim, 2, prior_std=prior_std)

        if state_dependent_std:
            self.out_log_std = BayesianLinear(hidden_dim, 2, prior_std=prior_std)
        else:
            self.log_std = nn.Parameter(torch.full((2,), -0.9))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.out_mean(x)

        if self.state_dependent_std:
            log_std = self.out_log_std(x).clamp(min=-2.5, max=-0.1)
            std = torch.exp(log_std)
        else:
            std = torch.exp(self.log_std).clamp(min=0.05, max=0.8)

        return mean, std

    def kl_divergence(self) -> torch.Tensor:
        kl = self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out_mean.kl_divergence()
        if self.state_dependent_std:
            kl = kl + self.out_log_std.kl_divergence()
        return kl


@dataclass
class TrainConfig:
    episodes: int = 1200
    horizon: int = 60
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-3
    hidden_dim: int = 128
    prior_std: float = 0.8
    kl_weight: float = 3e-6
    kl_warmup_episodes: int = 300
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    batch_episodes: int = 16
    state_dependent_std: bool = True
    seed: int = 1
    eval_interval: int = 50
    eval_episodes: int = 8
    metrics_path: str = "experiments/wet_chicken/vi_bnn_training_metrics.json"


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def normalize_state(state: np.ndarray) -> np.ndarray:
    return np.array([state[0] / max(LENGTH - 1, 1), state[1] / max(WIDTH - 1, 1)], dtype=np.float32)


def run_episode(
    env: WetChicken,
    policy: VariationalPolicy,
    value_net: ValueNetwork,
    horizon: int,
    gamma: float,
    gae_lambda: float,
):
    state = env.reset()
    states = []
    log_probs = []
    entropies = []
    values = []
    rewards = []

    for _ in range(horizon):
        state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)
        mean, std = policy(state_tensor)
        value = value_net(state_tensor)

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # Tanh squash correction for log-probability.
        log_prob = (dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + EPS)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        _, reward, next_state = env.step(action.squeeze(0).detach().cpu().numpy())

        states.append(state_tensor.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        entropies.append(entropy.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(float(reward))
        state = next_state

    next_state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_value = value_net(next_state_tensor).squeeze(0)

    values_tensor = torch.stack(values)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    advantages = torch.zeros_like(rewards_tensor)

    gae = torch.tensor(0.0)
    for t in reversed(range(horizon)):
        if t == horizon - 1:
            value_next = next_value
        else:
            value_next = values_tensor[t + 1]
        delta = rewards_tensor[t] + gamma * value_next - values_tensor[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values_tensor.detach()

    return {
        "states": torch.stack(states),
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "values": values_tensor,
        "rewards": rewards,
        "advantages": advantages,
        "returns": returns,
    }


def deterministic_action(policy: VariationalPolicy, state_tensor: torch.Tensor, mc_samples: int = 16) -> np.ndarray:
    means = []
    for _ in range(mc_samples):
        mean, _ = policy(state_tensor)
        means.append(torch.tanh(mean).squeeze(0).detach().cpu().numpy())
    return np.mean(np.stack(means, axis=0), axis=0)


def evaluate_policy(env: WetChicken, policy: VariationalPolicy, horizon: int, episodes: int = 10) -> float:
    policy.eval()
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        for _ in range(horizon):
            state_tensor = torch.tensor(normalize_state(state), dtype=torch.float32).unsqueeze(0)
            action = deterministic_action(policy, state_tensor)
            _, reward, state = env.step(action)
            total += float(reward)
        returns.append(total)
    policy.train()
    return float(np.mean(returns))


def train(config: TrainConfig) -> tuple[VariationalPolicy, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = WetChicken(LENGTH, WIDTH, MAX_TURBULENCE, MAX_VELOCITY)
    policy = VariationalPolicy(
        hidden_dim=config.hidden_dim,
        prior_std=config.prior_std,
        state_dependent_std=config.state_dependent_std,
    )
    value_net = ValueNetwork(hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=config.lr,
    )
    metrics = {
        "episode": [],
        "episode_return": [],
        "episode_avg_step_reward": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl": [],
        "eval_return": [],
    }

    episode = 0
    while episode < config.episodes:
        batch_log_probs = []
        batch_values = []
        batch_returns = []
        batch_advantages = []
        batch_entropies = []

        for _ in range(config.batch_episodes):
            if episode >= config.episodes:
                break
            episode += 1

            rollout = run_episode(
                env=env,
                policy=policy,
                value_net=value_net,
                horizon=config.horizon,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            batch_log_probs.append(rollout["log_probs"])
            batch_values.append(rollout["values"])
            batch_returns.append(rollout["returns"])
            batch_advantages.append(rollout["advantages"])
            batch_entropies.append(rollout["entropies"])

            episode_return = float(np.sum(rollout["rewards"]))
            avg_train_reward = float(np.mean(rollout["rewards"]))
            eval_return = float("nan")

            if episode % config.eval_interval == 0 or episode == 1:
                eval_return = evaluate_policy(env, policy, config.horizon, episodes=config.eval_episodes)

            metrics["episode"].append(float(episode))
            metrics["episode_return"].append(episode_return)
            metrics["episode_avg_step_reward"].append(avg_train_reward)
            metrics["eval_return"].append(eval_return)

        log_probs_tensor = torch.cat(batch_log_probs)
        values_tensor = torch.cat(batch_values)
        returns_tensor = torch.cat(batch_returns).detach()
        advantages_tensor = torch.cat(batch_advantages)
        entropies_tensor = torch.cat(batch_entropies)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
        value_loss = torch.nn.functional.mse_loss(values_tensor, returns_tensor)
        entropy = entropies_tensor.mean()

        kl = policy.kl_divergence()
        kl_scale = min(1.0, episode / max(config.kl_warmup_episodes, 1))
        loss = (
            policy_loss
            + config.value_coef * value_loss
            + config.kl_weight * kl_scale * kl
            - config.entropy_coef * entropy
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), max_norm=5.0)
        optimizer.step()

        eval_to_print = metrics["eval_return"][-1]
        print(
            f"Episode {episode:4d} | "
            f"batch_avg_return={float(np.mean(metrics['episode_return'][-config.batch_episodes:])):7.3f} | "
            f"last_eval={eval_to_print:7.3f} | "
            f"policy_loss={policy_loss.item():7.4f} | "
            f"value_loss={value_loss.item():7.4f} | "
            f"entropy={entropy.item():6.3f} | "
            f"kl={kl.item():8.2f} | "
            f"kl_scale={kl_scale:5.3f}"
        )

        metrics["policy_loss"].append(float(policy_loss.item()))
        metrics["value_loss"].append(float(value_loss.item()))
        metrics["entropy"].append(float(entropy.item()))
        metrics["kl"].append(float(kl.item()))

    while len(metrics["policy_loss"]) < len(metrics["episode"]):
        metrics["policy_loss"].append(float("nan"))
        metrics["value_loss"].append(float("nan"))
        metrics["entropy"].append(float("nan"))
        metrics["kl"].append(float("nan"))

    return policy, metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a simple VI-BNN policy on Wet Chicken.")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--prior-std", type=float, default=0.8)
    parser.add_argument("--kl-weight", type=float, default=3e-6)
    parser.add_argument("--kl-warmup-episodes", type=int, default=300)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument(
        "--disable-state-dependent-std",
        action="store_true",
        help="Use a global action variance instead of state-dependent variance.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--metrics-path", type=str, default="experiments/wet_chicken/vi_bnn_training_metrics.json")

    args = parser.parse_args()
    return TrainConfig(
        episodes=args.episodes,
        horizon=args.horizon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        prior_std=args.prior_std,
        kl_weight=args.kl_weight,
        kl_warmup_episodes=args.kl_warmup_episodes,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        batch_episodes=args.batch_episodes,
        state_dependent_std=not args.disable_state_dependent_std,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        metrics_path=args.metrics_path,
    )


def main() -> None:
    config = parse_args()
    policy, metrics = train(config)

    out_path = "experiments/wet_chicken/vi_bnn_policy.pt"
    checkpoint: dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "hidden_dim": config.hidden_dim,
        "prior_std": config.prior_std,
        "state_dependent_std": config.state_dependent_std,
        "length": LENGTH,
        "width": WIDTH,
    }
    if not config.state_dependent_std:
        checkpoint["log_std"] = policy.log_std.detach().cpu().tolist()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)

    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp)

    print(f"Saved trained policy to {out_path}")
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
