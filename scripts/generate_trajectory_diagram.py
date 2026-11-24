"""Generate liquidation trajectory comparisons across paired episodes."""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from lob_rainbow_execution.env_lob import LOBExecutionEnv
from lob_rainbow_execution.dqn_baseline import DuelingDQNNetwork
from lob_rainbow_execution.rainbow_c51_lob import DuelingC51Network


def generate_trajectories(output_path: str = "assets/execution_trajectories.png", seed: int = 10005) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = LOBExecutionEnv()

    # Load agents
    dqn_net = DuelingDQNNetwork()
    dqn_net.load_state_dict(torch.load("checkpoints/dqn_baseline.pt", map_location="cpu", weights_only=True))
    dqn_net.eval()

    c51_net = DuelingC51Network()
    c51_net.load_state_dict(torch.load("checkpoints/rainbow_c51.pt", map_location="cpu", weights_only=True))
    c51_net.eval()

    policies = {
        "TWAP Baseline": lambda obs: 2,
        "Double DQN": lambda obs: int(torch.argmax(dqn_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), dim=-1).item()),
        "C51 Rainbow (E[Z])": lambda obs: c51_net.select_action(torch.as_tensor(obs, dtype=torch.float32), mode="expected"),
        "C51 Rainbow (CVaR_0.25)": lambda obs: c51_net.select_action(torch.as_tensor(obs, dtype=torch.float32), mode="cvar", alpha=0.25),
    }

    colors = {
        "TWAP Baseline": "#7f7f7f",
        "Double DQN": "#1f77b4",
        "C51 Rainbow (E[Z])": "#ff7f0e",
        "C51 Rainbow (CVaR_0.25)": "#2ca02c",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for name, act_fn in policies.items():
        obs, _ = env.reset(seed=seed)
        done = False
        inv_history = [1.0]
        price_history = [env.mid_price]
        time_steps = [0]

        while not done:
            action = act_fn(obs)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            inv_history.append(env.q_t / env.Q0)
            price_history.append(env.mid_price)
            time_steps.append(env.t)

        ax1.plot(time_steps, inv_history, label=name, color=colors[name], linewidth=2.0)
        ax2.plot(time_steps, price_history, label=name, color=colors[name], linewidth=1.5, alpha=0.85)

    ax1.set_title(f"Remaining Inventory Trajectory q_t / Q0 (Seed {seed})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Decision Step t", fontsize=10)
    ax1.set_ylabel("Normalized Inventory", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.set_title(f"LOB Mid-Price Realization (Seed {seed})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decision Step t", fontsize=10)
    ax2.set_ylabel("Mid-Price ($)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved trajectory comparison to: {output_path}")


if __name__ == "__main__":
    generate_trajectories()
