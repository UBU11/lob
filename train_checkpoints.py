"""
Script to train full-horizon converged checkpoints for Double DQN and Rainbow C51.
Saves weights to checkpoints/dqn_baseline.pt and checkpoints/rainbow_c51.pt.
"""

import os
import time
from lob_rainbow_execution.env_lob import LOBExecutionEnv
from lob_rainbow_execution.dqn_baseline import DoubleDQNAgent
from lob_rainbow_execution.rainbow_c51_lob import RainbowC51Agent


def train_all(
    episodes: int = 2000,
    warmup: int = 100,
    checkpoints_dir: str = "checkpoints",
) -> None:
    os.makedirs(checkpoints_dir, exist_ok=True)
    env = LOBExecutionEnv()

    # 1. Train Double DQN Baseline
    print(f"\n{'='*70}")
    print(f"[1/2] Training Double DQN Baseline ({episodes} episodes)...")
    print(f"{'='*70}")
    t0 = time.time()
    dqn_agent = DoubleDQNAgent(env)
    dqn_agent.train(
        total_episodes=episodes,
        warmup_episodes=warmup,
        epsilon_decay_episodes=int(episodes * 0.75),
    )
    dqn_path = os.path.join(checkpoints_dir, "dqn_baseline.pt")
    dqn_agent.save(dqn_path)
    print(f"Double DQN training completed in {time.time() - t0:.1f}s -> {dqn_path}")

    # 2. Train Distributional Rainbow C51 Agent
    print(f"\n{'='*70}")
    print(f"[2/2] Training Distributional Rainbow C51 Agent ({episodes} episodes)...")
    print(f"{'='*70}")
    t0 = time.time()
    c51_agent = RainbowC51Agent(env)
    c51_agent.train(
        total_episodes=episodes,
        warmup_episodes=warmup,
        epsilon_decay_episodes=int(episodes * 0.75),
    )
    c51_path = os.path.join(checkpoints_dir, "rainbow_c51.pt")
    c51_agent.save(c51_path)
    print(f"Rainbow C51 training completed in {time.time() - t0:.1f}s -> {c51_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train full-horizon LOB execution models.")
    parser.add_argument("--episodes", type=int, default=2000, help="Episodes per agent (default: 2000)")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup episodes (default: 100)")
    parser.add_argument("--dir", type=str, default="checkpoints", help="Checkpoints directory")
    args = parser.parse_args()

    train_all(episodes=args.episodes, warmup=args.warmup, checkpoints_dir=args.dir)
