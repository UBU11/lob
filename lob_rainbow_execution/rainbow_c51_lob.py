from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lob_rainbow_execution.env_lob import LOBExecutionEnv


class SumTree:

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float) -> int:
        data_idx = self.data_pointer
        tree_idx = data_idx + self.capacity - 1
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        change = float(priority) - self.tree[tree_idx]
        self.tree[tree_idx] = float(priority)
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> tuple[int, float, int]:
        parent_idx = 0
        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), data_idx


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 6,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 50_000,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_frames = int(beta_frames)
        self.frame = 0

        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self.tree.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        priority = (self.max_priority + 1e-5) ** self.alpha
        data_idx = self.tree.add(priority)

        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = done

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:

        self.frame += 1
        beta = min(
            self.beta_end,
            self.beta_start + self.frame * (self.beta_end - self.beta_start) / max(1, self.beta_frames),
        )

        segment = self.tree.total() / batch_size
        tree_indices = []
        data_indices = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            val = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(val)
            tree_indices.append(tree_idx)
            priorities.append(priority)
            data_indices.append(data_idx)

        total_p = self.tree.total()
        probs = np.array(priorities, dtype=np.float64) / (total_p + 1e-8)
        n = len(self)

        weights = (n * probs) ** (-beta)
        weights /= weights.max() + 1e-8
        weights_t = torch.as_tensor(weights, dtype=torch.float32)

        states_t = torch.as_tensor(self.states[data_indices], dtype=torch.float32)
        actions_t = torch.as_tensor(self.actions[data_indices], dtype=torch.int64)
        rewards_t = torch.as_tensor(self.rewards[data_indices], dtype=torch.float32)
        next_states_t = torch.as_tensor(self.next_states[data_indices], dtype=torch.float32)
        dones_t = torch.as_tensor(self.dones[data_indices], dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t, weights_t, tree_indices

    def update_priorities(self, tree_indices: list[int], td_errors: np.ndarray | torch.Tensor) -> None:
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = float(np.abs(td_error) + 1e-5) ** self.alpha
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, float(np.abs(td_error)))


class DuelingC51Network(nn.Module):
    """
    Categorical Dueling Q-Network estimating discrete probability distributions
    over 51 return atoms for each action. Separates scalar value logits V(s)
    and advantage logits A(s, a) to produce normalised categorical distributions.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 5,
        n_atoms: int = 51,
        v_min: float = -25.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.delta_z = (self.v_max - self.v_min) / float(self.n_atoms - 1)

        support = torch.linspace(self.v_min, self.v_max, self.n_atoms, dtype=torch.float32)
        self.register_buffer("support", support)

        # Torso: Linear(6, 128) -> LayerNorm(128) -> ReLU -> Linear(128, 128) -> ReLU
        self.torso = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream: Linear(128, 64) -> ReLU -> Linear(64, n_atoms)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_atoms),
        )

        # Advantage stream: Linear(128, 64) -> ReLU -> Linear(64, action_dim * n_atoms)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim * self.n_atoms),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        features = self.torso(state)
        batch_size = state.size(0)

        value_logits = self.value_stream(features).view(batch_size, 1, self.n_atoms)
        adv_logits = self.advantage_stream(features).view(batch_size, self.action_dim, self.n_atoms)

        q_logits = value_logits + (adv_logits - adv_logits.mean(dim=1, keepdim=True))

        probs = F.softmax(q_logits, dim=-1)
        return probs

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:

        probs = self.forward(state)
        q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=-1)
        return q_values

    def get_action(
        self,
        state: torch.Tensor,
        mode: str = "expected",
        alpha: float = 0.25,
    ) -> int:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if mode == "expected":
                q_vals = self.get_q_values(state)
                return int(torch.argmax(q_vals, dim=-1).item())
            elif mode == "cvar":
                probs = self.forward(state)  # [1, action_dim, n_atoms]
                cdf = torch.cumsum(probs, dim=-1)

                tail_mask = (cdf <= alpha).float()
                tail_mask[..., 0] = 1.0

                tail_probs = probs * tail_mask
                tail_probs_norm = tail_probs / (tail_probs.sum(dim=-1, keepdim=True) + 1e-8)
                cvar_values = torch.sum(tail_probs_norm * self.support.view(1, 1, -1), dim=-1)
                return int(torch.argmax(cvar_values, dim=-1).item())
            else:
                raise ValueError(f"Unknown mode: {mode}. Must be 'expected' or 'cvar'.")


def project_bellman_distribution(
    online_net: DuelingC51Network,
    target_net: DuelingC51Network,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.995,
) -> torch.Tensor:

    batch_size = next_states.size(0)
    n_atoms = online_net.n_atoms
    v_min = online_net.v_min
    v_max = online_net.v_max
    delta_z = online_net.delta_z
    support = online_net.support

    with torch.no_grad():
        next_q_values = online_net.get_q_values(next_states)
        next_actions = torch.argmax(next_q_values, dim=-1)

        next_probs = target_net(next_states)
        next_best_probs = next_probs[torch.arange(batch_size, device=next_states.device), next_actions]

        # Clamp shifted Bellman return support to [v_min, v_max] interval
        t_z = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
        t_z = torch.clamp(t_z, min=v_min, max=v_max)

        b = (t_z - v_min) / delta_z
        l = torch.floor(b).long().clamp(0, n_atoms - 1)
        u = torch.ceil(b).long().clamp(0, n_atoms - 1)

        eq_mask = (l == u).float()
        l_weight = (u.float() - b) + eq_mask
        u_weight = (b - l.float())

        proj_dist = torch.zeros((batch_size, n_atoms), dtype=torch.float32, device=next_states.device)
        offset = (
            torch.linspace(0, (batch_size - 1) * n_atoms, batch_size, device=next_states.device)
            .long()
            .unsqueeze(1)
        )

        proj_dist.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_best_probs * l_weight).view(-1),
        )
        proj_dist.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_best_probs * u_weight).view(-1),
        )

    return proj_dist


def compute_c51_loss(
    online_net: DuelingC51Network,
    target_net: DuelingC51Network,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 0.995,
) -> tuple[torch.Tensor, torch.Tensor]:

    batch_size = states.size(0)
    proj_dist = project_bellman_distribution(
        online_net=online_net,
        target_net=target_net,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
    )

    curr_probs = online_net(states)
    curr_action_probs = curr_probs[torch.arange(batch_size, device=states.device), actions]

    # Sample-wise cross-entropy loss L_k = - sum_i m_i * log(p_i + eps)
    sample_losses = -torch.sum(proj_dist * torch.log(curr_action_probs.clamp(min=1e-8)), dim=-1)
    weighted_loss = torch.mean(weights * sample_losses)

    return weighted_loss, sample_losses


class RainbowC51Agent:

    def __init__(
        self,
        env: LOBExecutionEnv,
        learning_rate: float = 1e-4,
        gamma: float = 0.995,
        n_atoms: int = 51,
        v_min: float = -25.0,
        v_max: float = 10.0,
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

        self.online_net = DuelingC51Network(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
        ).to(self.device)

        self.target_net = DuelingC51Network(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            state_dim=env.observation_space.shape[0],
        )

    def soft_update_target(self) -> None:
        for param, target_param in zip(self.online_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, tree_indices = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        loss, sample_losses = compute_c51_loss(
            online_net=self.online_net,
            target_net=self.target_net,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            weights=weights,
            gamma=self.gamma,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(tree_indices, sample_losses)
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
                    action = self.online_net.get_action(obs_t, mode="expected")

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
        print(f"Saved Rainbow C51 weights to {path}")

    def load(self, path: str) -> None:
        self.online_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        print(f"Loaded Rainbow C51 weights from {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Distributional Rainbow C51 Agent on LOB execution.")
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup random exploration episodes")
    parser.add_argument("--save_path", type=str, default="checkpoints/rainbow_c51.pt", help="Path to save weights")
    args = parser.parse_args()

    env = LOBExecutionEnv()
    agent = RainbowC51Agent(env)
    print(f"Training Rainbow C51 agent for {args.episodes} episodes...")
    agent.train(
        total_episodes=args.episodes,
        warmup_episodes=args.warmup,
        epsilon_decay_episodes=int(args.episodes * 0.75),
    )
    agent.save(args.save_path)

