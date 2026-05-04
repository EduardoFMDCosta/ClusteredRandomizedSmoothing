import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from dynamics import UAV3D
from torch import nn
from torch.distributions import Categorical, Normal

EPS = 1e-8
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class VariationalPolicy(nn.Module):
    """
    Tanh-squashed Gaussian-mixture policy.

    At state s the policy is
        z ~ Categorical(softmax(gate(s)))
        raw ~ N(mean_z(s), exp(log_std_z))
        action = tanh(raw) * action_scale

    State-dependent means and gating allow multi-modal behaviour that varies
    with the state (e.g. near an obstacle the gate can place mass on two
    components corresponding to going left vs right). ``log_std`` is a global
    per-component parameter (shape [K, A]) - simpler and more stable than a
    state-dependent std network.

    With ``num_components = 1`` this reduces to a single-Gaussian policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_scale: float = 1.0,
        num_components: int = 1,
        init_log_std: float = -1.2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_scale = float(action_scale)
        self.num_components = int(num_components)
        assert self.num_components >= 1

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, self.num_components * action_dim)
        if self.num_components > 1:
            self.gate_head = nn.Linear(hidden_dim, self.num_components)

        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        # Small last-layer init: near-zero initial component means, but
        # orthogonal rows make the K components start with different random
        # biases so they do not collapse into a single mode.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        if self.num_components > 1:
            nn.init.orthogonal_(self.gate_head.weight, gain=0.01)
            nn.init.zeros_(self.gate_head.bias)

        # Global per-component log_std, shape [K, A].
        self.log_std = nn.Parameter(
            torch.full((self.num_components, action_dim), float(init_log_std))
        )

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        return x

    def distribution_params(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (means, log_stds, log_mix_weights).

        means: [B, K, A]
        log_stds: [B, K, A] (broadcast from the global parameter)
        log_mix_weights: [B, K] (log-softmax of gate logits; all zeros for K=1)
        """
        h = self._features(state)
        B = state.shape[0]
        K, A = self.num_components, self.action_dim
        means = self.mean_head(h).view(B, K, A)
        log_stds_clamped = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        log_stds = log_stds_clamped.unsqueeze(0).expand(B, K, A)
        if K == 1:
            log_mix_weights = torch.zeros(B, 1, device=state.device, dtype=means.dtype)
        else:
            log_mix_weights = F.log_softmax(self.gate_head(h), dim=-1)
        return means, log_stds, log_mix_weights

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-compat: return (mean, log_std) of the most-likely component."""
        means, log_stds, log_mix_w = self.distribution_params(state)
        if self.num_components == 1:
            return means.squeeze(1), log_stds.squeeze(1)
        idx = log_mix_w.argmax(dim=-1)
        batch_idx = torch.arange(state.shape[0], device=state.device)
        return means[batch_idx, idx], log_stds[batch_idx, idx]

    def _tanh_correction(self, raw: torch.Tensor) -> torch.Tensor:
        """log|d action / d raw| summed over the action dimension.

        Same for all mixture components because they share ``raw``.
        """
        log_det = 2.0 * (math.log(2.0) - raw - F.softplus(-2.0 * raw))
        log_scale = math.log(self.action_scale) if self.action_scale > 0 else 0.0
        return log_det.sum(dim=-1) + log_scale * self.action_dim

    def _mixture_log_prob(
        self,
        raw: torch.Tensor,         # [B, A]
        means: torch.Tensor,       # [B, K, A]
        stds: torch.Tensor,        # [B, K, A]
        log_mix_weights: torch.Tensor,  # [B, K]
    ) -> torch.Tensor:
        raw_expanded = raw.unsqueeze(1)  # [B, 1, A]
        comp_log_prob = Normal(means, stds + EPS).log_prob(raw_expanded).sum(dim=-1)
        log_prob_raw = torch.logsumexp(log_mix_weights + comp_log_prob, dim=-1)
        return log_prob_raw - self._tanh_correction(raw)

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a tanh-squashed action. Returns (action, raw, mixture_log_prob)."""
        means, log_stds, log_mix_w = self.distribution_params(state)
        stds = log_stds.exp()
        B = state.shape[0]
        if self.num_components == 1:
            mean_sel = means.squeeze(1)
            std_sel = stds.squeeze(1)
        else:
            z = Categorical(logits=log_mix_w).sample()  # [B]
            batch_idx = torch.arange(B, device=state.device)
            mean_sel = means[batch_idx, z]
            std_sel = stds[batch_idx, z]
        dist = Normal(mean_sel, std_sel + EPS)
        raw = dist.rsample()
        action = torch.tanh(raw) * self.action_scale
        log_prob = self._mixture_log_prob(raw, means, stds, log_mix_w)
        return action, raw, log_prob

    def log_prob_and_entropy(
        self,
        state: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, log_stds, log_mix_w = self.distribution_params(state)
        stds = log_stds.exp()
        log_prob = self._mixture_log_prob(raw_action, means, stds, log_mix_w)
        # Lower bound on mixture entropy: H(cat) + sum_k pi_k H(N_k).
        # Using Gaussian entropy in raw-space (no tanh correction) is what the
        # existing single-Gaussian code does and is standard for PPO entropy
        # regularisation.
        comp_entropy = Normal(means, stds + EPS).entropy().sum(dim=-1)  # [B, K]
        if self.num_components == 1:
            entropy = comp_entropy.squeeze(-1)
        else:
            mix_w = log_mix_w.exp()
            cat_entropy = -(mix_w * log_mix_w).sum(dim=-1)
            entropy = cat_entropy + (mix_w * comp_entropy).sum(dim=-1)
        return log_prob, entropy

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Most-likely component's mean action (tanh-squashed)."""
        means, _, log_mix_w = self.distribution_params(state)
        if self.num_components == 1:
            mean = means.squeeze(1)
        else:
            idx = log_mix_w.argmax(dim=-1)
            batch_idx = torch.arange(state.shape[0], device=state.device)
            mean = means[batch_idx, idx]
        return torch.tanh(mean) * self.action_scale


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.zeros_(layer.bias)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.orthogonal_(last.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    episodes: int = 200000
    horizon: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    hidden_dim: int = 128
    num_components: int = 2
    init_log_std: float = -1.2
    action_scale: float = 4.0

    batch_episodes: int = 32
    ppo_epochs: int = 10
    minibatch_size: int = 512
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5

    seed: int = 42
    eval_interval: int = 200  # in updates, not episodes
    eval_episodes: int = 16
    uav3d_noise_std: float = 0.05
    uav3d_max_steps: int = 64
    uav3d_curriculum_episodes: int = 40000
    metrics_path: str = ""


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize UAV3D state: [x_pos, x_vel, y_pos, y_vel, z_pos, z_vel]."""
    return np.array(
        [
            state[0] / 15.0,
            state[1] / 7.0,
            state[2] / 9.0,
            state[3] / 7.0,
            state[4] / 7.0,
            state[5] / 7.0,
        ],
        dtype=np.float32,
    )


def curriculum_fraction(episode: int, curriculum_episodes: int) -> float:
    if curriculum_episodes <= 0:
        return 1.0
    return float(min(1.0, max(0.0, episode / curriculum_episodes)))


def collect_rollout_batch(
    env: UAV3D,
    policy: VariationalPolicy,
    critic: ValueNetwork,
    num_episodes: int,
    horizon: int,
    gamma: float,
    gae_lambda: float,
    curr_frac_start: float,
    curr_frac_end: float,
) -> dict[str, Any]:
    """Collect ``num_episodes`` rollouts with current policy and compute GAE."""
    ep_states, ep_raws, ep_logps = [], [], []
    ep_advantages, ep_returns, ep_values = [], [], []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    reached_goal: list[int] = []

    policy.eval()
    critic.eval()
    for ep_idx in range(num_episodes):
        # Linearly spread the curriculum across the batch so each update
        # observes a mix of difficulties.
        if num_episodes == 1:
            frac = 0.5 * (curr_frac_start + curr_frac_end)
        else:
            t = ep_idx / (num_episodes - 1)
            frac = (1.0 - t) * curr_frac_start + t * curr_frac_end
        state = env.reset(curriculum_frac=frac)

        states, raws, lps, rs, vs = [], [], [], [], []
        for _ in range(horizon):
            s_t = torch.from_numpy(normalize_state(state)).unsqueeze(0)
            with torch.no_grad():
                action, raw, lp = policy.sample(s_t)
                v = critic(s_t)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            _, reward, next_state = env.step(a_np)

            states.append(s_t.squeeze(0))
            raws.append(raw.squeeze(0))
            lps.append(lp.squeeze(0))
            rs.append(float(reward))
            vs.append(float(v.item()))
            state = next_state
            if env.is_done():
                break

        T = len(rs)
        if T == 0:
            continue

        if env.is_done():
            last_val = 0.0
            got_goal = 1 if rs[-1] > 9.0 else 0
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(normalize_state(state)).unsqueeze(0)
                last_val = float(critic(s_t).item())
            got_goal = 0

        rs_arr = np.asarray(rs, dtype=np.float32)
        vs_ext = np.asarray(vs + [last_val], dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rs_arr[t] + gamma * vs_ext[t + 1] - vs_ext[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
        returns = advantages + vs_ext[:T]

        ep_states.append(torch.stack(states))
        ep_raws.append(torch.stack(raws))
        ep_logps.append(torch.stack(lps))
        ep_advantages.append(torch.from_numpy(advantages))
        ep_returns.append(torch.from_numpy(returns))
        ep_values.append(torch.tensor(vs, dtype=torch.float32))

        episode_returns.append(float(rs_arr.sum()))
        episode_lengths.append(T)
        reached_goal.append(got_goal)

    return {
        "states": torch.cat(ep_states, dim=0),
        "raw_actions": torch.cat(ep_raws, dim=0),
        "log_probs": torch.cat(ep_logps, dim=0),
        "advantages": torch.cat(ep_advantages, dim=0),
        "returns": torch.cat(ep_returns, dim=0),
        "values": torch.cat(ep_values, dim=0),
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "reached_goal": reached_goal,
    }


def ppo_update(
    policy: VariationalPolicy,
    critic: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    config: TrainConfig,
) -> dict[str, float]:
    states = batch["states"]
    raws = batch["raw_actions"]
    old_log_probs = batch["log_probs"].detach()
    advantages = batch["advantages"]
    returns = batch["returns"]

    # Normalize advantages
    adv_std = advantages.std()
    if torch.isfinite(adv_std) and adv_std > 1e-6:
        advantages = (advantages - advantages.mean()) / (adv_std + EPS)
    else:
        advantages = advantages - advantages.mean()

    N = states.shape[0]
    minibatch_size = max(32, min(config.minibatch_size, N))

    policy.train()
    critic.train()

    policy_losses, value_losses, entropies, kls, clipfracs = [], [], [], [], []
    for _ in range(config.ppo_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]
            mb_states = states[idx]
            mb_raws = raws[idx]
            mb_old_lp = old_log_probs[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]

            new_lp, entropy = policy.log_prob_and_entropy(mb_states, mb_raws)
            values = critic(mb_states)

            log_ratio = new_lp - mb_old_lp
            ratio = torch.exp(log_ratio)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_ret)
            ent_loss = -entropy.mean()

            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(critic.parameters()),
                max_norm=config.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = (-log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > config.clip_range).float().mean().item()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.mean().item()))
            kls.append(float(approx_kl))
            clipfracs.append(float(clip_frac))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "kl": float(np.mean(kls)),
        "clip_frac": float(np.mean(clipfracs)),
    }

def evaluate_policy(
    env: UAV3D,
    policy: VariationalPolicy,
    horizon: int,
    episodes: int,
    curriculum_frac: float,
) -> dict[str, float]:
    policy.eval()
    returns: list[float] = []
    reached: list[int] = []
    for _ in range(episodes):
        state = env.reset(curriculum_frac=curriculum_frac)
        total = 0.0
        got_goal = 0
        for _ in range(horizon):
            s_t = torch.from_numpy(normalize_state(state)).unsqueeze(0)
            action, _, _ = policy.sample(s_t)
            a = action.squeeze(0).cpu().numpy().astype(np.float32)
            _, reward, state = env.step(a)
            total += float(reward)
            if env.is_done():
                if reward > 9.0:
                    got_goal = 1
                break
        returns.append(total)
        reached.append(got_goal)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "goal_rate": float(np.mean(reached)),
    }


def train(config: TrainConfig) -> tuple[VariationalPolicy, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env = UAV3D(
        gaussian_noise=True,
        noise_std=config.uav3d_noise_std,
        max_steps=max(int(config.uav3d_max_steps), int(config.horizon)),
    )
    eval_env = UAV3D(
        gaussian_noise=True,
        noise_std=config.uav3d_noise_std,
        max_steps=max(int(config.uav3d_max_steps), int(config.horizon)),
    )
    state_dim, action_dim = 6, 3

    policy = VariationalPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        action_scale=config.action_scale,
        num_components=config.num_components,
        init_log_std=config.init_log_std,
    )
    critic = ValueNetwork(state_dim=state_dim, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
        lr=config.lr,
    )

    metrics: dict[str, list[float]] = {
        "episode": [],
        "update": [],
        "episode_return": [],
        "episode_avg_step_reward": [],
        "episode_length": [],
        "goal_rate": [],
        "curriculum_frac": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl": [],
        "clip_frac": [],
        "eval_return": [],
        "eval_goal_rate": [],
    }

    episode = 0
    update = 0
    while episode < config.episodes:
        curr_frac_start = curriculum_fraction(episode, config.uav3d_curriculum_episodes)
        curr_frac_end = curriculum_fraction(
            episode + config.batch_episodes, config.uav3d_curriculum_episodes
        )

        batch = collect_rollout_batch(
            env=env,
            policy=policy,
            critic=critic,
            num_episodes=config.batch_episodes,
            horizon=config.horizon,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            curr_frac_start=curr_frac_start,
            curr_frac_end=curr_frac_end,
        )
        stats = ppo_update(policy, critic, optimizer, batch, config)

        episode += config.batch_episodes
        update += 1

        ep_returns = batch["episode_returns"]
        ep_lens = batch["episode_lengths"]
        goal_rate = float(np.mean(batch["reached_goal"])) if batch["reached_goal"] else 0.0
        mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")
        mean_len = float(np.mean(ep_lens)) if ep_lens else float("nan")
        mean_step_reward = mean_return / mean_len if mean_len > 0 else float("nan")

        # Eval every ``eval_interval`` updates.
        eval_return = float("nan")
        eval_goal = float("nan")
        if update % config.eval_interval == 0 or update == 1:
            eval_stats = evaluate_policy(
                eval_env,
                policy,
                config.horizon,
                config.eval_episodes,
                curriculum_frac=1.0,  # Always evaluate on the hard task.
            )
            eval_return = eval_stats["mean_return"]
            eval_goal = eval_stats["goal_rate"]
            print(
                f"Update {update:5d} | Ep {episode:7d} | curr={curr_frac_end:.2f} | "
                f"train_ret={mean_return:7.3f} (len={mean_len:5.1f}, goal={goal_rate:.2f}) | "
                f"eval_ret={eval_return:7.3f} (goal={eval_goal:.2f}) | "
                f"pi_loss={stats['policy_loss']:+7.4f} | v_loss={stats['value_loss']:6.3f} | "
                f"ent={stats['entropy']:+5.3f} | kl={stats['kl']:+.4f}"
            )

        metrics["episode"].append(float(episode))
        metrics["update"].append(float(update))
        metrics["episode_return"].append(mean_return)
        metrics["episode_avg_step_reward"].append(mean_step_reward)
        metrics["episode_length"].append(mean_len)
        metrics["goal_rate"].append(goal_rate)
        metrics["curriculum_frac"].append(curr_frac_end)
        metrics["policy_loss"].append(stats["policy_loss"])
        metrics["value_loss"].append(stats["value_loss"])
        metrics["entropy"].append(stats["entropy"])
        metrics["kl"].append(stats["kl"])
        metrics["clip_frac"].append(stats["clip_frac"])
        metrics["eval_return"].append(eval_return)
        metrics["eval_goal_rate"].append(eval_goal)

    return policy, metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="PPO training for policy on UAV3D benchmark.")
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-components", type=int, default=2,
                        help="Number of Gaussian components in the policy mixture.")
    parser.add_argument("--init-log-std", type=float, default=-1.2)
    parser.add_argument("--action-scale", type=float, default=4.0)

    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Evaluate every N updates (not episodes).")
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--uav3d-noise-std", type=float, default=0.05)
    parser.add_argument("--uav3d-max-steps", type=int, default=64)
    parser.add_argument("--uav3d-curriculum-episodes", type=int, default=40000)
    parser.add_argument("--metrics-path", type=str, default=None)

    args = parser.parse_args()
    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = "experiments/uav_3d/training_metrics.json"

    return TrainConfig(
        episodes=args.episodes,
        horizon=args.horizon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_components=args.num_components,
        init_log_std=args.init_log_std,
        action_scale=args.action_scale,
        batch_episodes=args.batch_episodes,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        uav3d_noise_std=args.uav3d_noise_std,
        uav3d_max_steps=args.uav3d_max_steps,
        uav3d_curriculum_episodes=args.uav3d_curriculum_episodes,
        metrics_path=metrics_path,
    )


def main() -> None:
    config = parse_args()
    policy, metrics = train(config)

    out_path = "experiments/uav_3d/policy.pt"
    checkpoint: dict[str, Any] = {
        "model_state_dict": policy.state_dict(),
        "state_dim": policy.state_dim,
        "action_dim": policy.action_dim,
        "hidden_dim": config.hidden_dim,
        "num_components": config.num_components,
        "action_scale": config.action_scale,
    }
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
