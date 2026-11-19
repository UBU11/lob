# Single-File Double + Dueling Scalar DQN Baseline for Limit Order Book Execution.



from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lob_rainbow_execution.env_lob import LOBExecutionEnv


class ReplayBuffer:

    def __init__(self, capacity: int = 100_000, state_dim: int = 6) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states_t = torch.as_tensor(self.states[indices], dtype=torch.float32)
        actions_t = torch.as_tensor(self.actions[indices], dtype=torch.int64)
        rewards_t = torch.as_tensor(self.rewards[indices], dtype=torch.float32)
        next_states_t = torch.as_tensor(self.next_states[indices], dtype=torch.float32)
        dones_t = torch.as_tensor(self.dones[indices], dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t


class DuelingDQNNetwork(nn.Module):
    # Q(s, a) = V(s) + (A(s, a) - 1/|A| sum_a' A(s, a'))


    def __init__(self, state_dim: int = 6, action_dim: int = 5) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared Torso matching C51 Rainbow architecture for fair benchmark isolation
        self.torso = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # State-value stream V(s) -> R^1
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream A(s, a) -> R^|A|
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.torso(state)
        value = self.value_stream(features)  # [B, 1]
        advantage = self.advantage_stream(features)  # [B, action_dim]

        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.forward(state)
            return int(torch.argmax(q_values, dim=-1).item())


def compute_double_dqn_loss(
    online_net: DuelingDQNNetwork,
    target_net: DuelingDQNNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
) -> torch.Tensor:

    batch_size = states.size(0)

    current_q_all = online_net(states)
    current_q = current_q_all[torch.arange(batch_size, device=states.device), actions]

    with torch.no_grad():
        next_q_online = online_net(next_states)
        next_actions = torch.argmax(next_q_online, dim=-1)

        next_q_target = target_net(next_states)
        next_q_val = next_q_target[torch.arange(batch_size, device=states.device), next_actions]

        target_q = rewards + (1.0 - dones) * gamma * next_q_val

    loss = F.smooth_l1_loss(current_q, target_q)
    return loss


class DoubleDQNAgent:


    def __init__(
        self,
        env: LOBExecutionEnv,
        learning_rate: float = 1e-4,
        gamma: float = 0.995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.tau = float(tau)

        self.online_net = DuelingDQNNetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        ).to(self.device)

        self.target_net = DuelingDQNNetwork(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=env.observation_space.shape[0],
        )

    def soft_update_target(self) -> None:
        for param, target_param in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        loss = compute_double_dqn_loss(
            online_net=self.online_net,
            target_net=self.target_net,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            gamma=self.gamma,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.soft_update_target()
        return float(loss.item())

    def train(
        self,
        total_episodes: int = 1_000,
        warmup_episodes: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 600,
    ) -> list[float]:
        episode_rewards = []
        for ep in range(total_episodes):
            epsilon = max(
                epsilon_end,
                epsilon_start - (ep / max(1, epsilon_decay_episodes)) * (epsilon_start - epsilon_end),
            )
            obs, _ = self.env.reset(seed=ep)
            done = False
            total_r = 0.0

            while not done:
                if ep < warmup_episodes or np.random.rand() < epsilon:
                    action = int(self.env.action_space.sample())
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    action = self.online_net.get_action(obs_t)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                total_r += reward

                if ep >= warmup_episodes:
                    self.train_step()

            episode_rewards.append(total_r)

        return episode_rewards

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.online_net.state_dict(), path)
        print(f"Saved Double DQN weights to {path}")

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        print(f"Loaded Double DQN weights from {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Double DQN Baseline on LOB execution.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup random exploration episodes")
    parser.add_argument("--save_path", type=str, default="checkpoints/dqn_baseline.pt", help="Path to save weights")
    args = parser.parse_args()

    env = LOBExecutionEnv()
    agent = DoubleDQNAgent(env)
    print(f"Training Double DQN baseline for {args.episodes} episodes...")
    agent.train(
        total_episodes=args.episodes,
        warmup_episodes=args.warmup,
        epsilon_decay_episodes=int(args.episodes * 0.75),
    )
    agent.save(args.save_path)

