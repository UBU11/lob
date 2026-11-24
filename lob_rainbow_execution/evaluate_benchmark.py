"""
Paired-Seed Out-of-Sample (OOS) Benchmark Evaluation and Visualization.
Strictly adheres to Section 7 of AGENTS.md and Section 5 of Project.md:
- Evaluates 4 policies on paired out-of-sample seeds (seed = 10_000 .. 10_999):
    1. TWAP Baseline
    2. Standard Double Dueling DQN
    3. C51 Rainbow DQN (E[Z] policy)
    4. C51 Rainbow DQN (CVaR_0.25 policy)
- Computes Mean IS (bps), IS Std Dev (bps), Execution Sharpe, and 95% CVaR (tail risk bps).
- Generates markdown report table and saves 'benchmark_distributions.png'.
"""

from typing import Any, Callable
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from lob_rainbow_execution.env_lob import LOBExecutionEnv
from lob_rainbow_execution.dqn_baseline import DuelingDQNNetwork, DoubleDQNAgent
from lob_rainbow_execution.rainbow_c51_lob import DuelingC51Network, RainbowC51Agent


def calculate_execution_metrics(is_arr: np.ndarray, alpha_cvar: float = 0.05) -> dict[str, float]:
    """Compute summary execution metrics from array of implementation shortfall bps."""
    n = len(is_arr)
    mean_val = float(np.mean(is_arr))
    std_val = float(np.std(is_arr, ddof=1)) if n > 1 else 0.0
    sharpe_val = float(mean_val / (std_val + 1e-8))
    k = max(1, int(np.ceil(alpha_cvar * n)))
    sorted_vals = np.sort(is_arr)
    cvar_val = float(np.mean(sorted_vals[:k]))
    return {
        "mean_is_bps": mean_val,
        "std_is_bps": std_val,
        "sharpe": sharpe_val,
        "cvar_95_bps": cvar_val,
    }

def evaluate_policy_on_seeds(
    env: LOBExecutionEnv,
    action_selector: Callable[[np.ndarray], int],
    seeds: list[int],
) -> dict[str, Any]:

    is_bps_list = []
    levels_walked_list = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        total_cash = 0.0
        total_levels_walked = 0

        while not done:
            action = action_selector(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            total_cash += step_info["cash_earned"]
            total_levels_walked += step_info["levels_walked"]

        is_bps = ((total_cash - env.Q0 * env.arrival_price) / (env.Q0 * env.arrival_price)) * 1e4
        is_bps_list.append(is_bps)
        levels_walked_list.append(total_levels_walked)

    is_arr = np.array(is_bps_list, dtype=np.float64)
    n_episodes = len(is_arr)

    mean_is = float(np.mean(is_arr))
    std_is = float(np.std(is_arr, ddof=1)) if n_episodes > 1 else 0.0
    sharpe = float(mean_is / (std_is + 1e-8))

    k_worst = max(1, int(np.ceil(0.05 * n_episodes)))
    sorted_is = np.sort(is_arr)
    cvar_95 = float(np.mean(sorted_is[:k_worst]))

    return {
        "mean_is_bps": mean_is,
        "std_is_bps": std_is,
        "sharpe": sharpe,
        "cvar_95_bps": cvar_95,
        "raw_is_bps": is_arr,
        "levels_walked": np.array(levels_walked_list, dtype=np.int32),
    }


def format_markdown_table(results: dict[str, dict[str, Any]]) -> str:
    header = (
        "| Policy Architecture | Mean IS (bps) | IS Std Dev (bps) | Execution Sharpe | 95% CVaR (Tail Risk bps) |\n"
        "| :-------------------------- | -------------: | ----------------: | ---------------: | -----------------------: |"
    )
    rows = []
    for name, m in results.items():
        row = (
            f"| **{name}** "
            f"| {m['mean_is_bps']:.2f} "
            f"| {m['std_is_bps']:.2f} "
            f"| {m['sharpe']:.2f} "
            f"| {m['cvar_95_bps']:.2f} |"
        )
        rows.append(row)

    table = header + "\n" + "\n".join(rows)
    return table


def plot_benchmark_distributions(
    results: dict[str, dict[str, Any]],
    c51_net: DuelingC51Network | None = None,
    output_path: str = "benchmark_distributions.png",
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax1 = axes[0]
    colors = {
        "TWAP Baseline": "#7f7f7f",
        "Standard Double DQN": "#1f77b4",
        "C51 Rainbow (E[Z])": "#ff7f0e",
        "C51 Rainbow (CVaR_0.25)": "#2ca02c",
    }

    for name, m in results.items():
        is_data = m["raw_is_bps"]
        color = colors.get(name, None)
        # Plot KDE line
        kde = stats.gaussian_kde(is_data)
        x_grid = np.linspace(min(is_data) - 2.0, max(is_data) + 2.0, 300)
        ax1.plot(x_grid, kde(x_grid), label=f"{name} (Mean: {m['mean_is_bps']:.2f})", color=color, linewidth=2)
        ax1.fill_between(x_grid, kde(x_grid), alpha=0.15, color=color)

    ax1.set_title("Out-of-Sample Implementation Shortfall Distributions (1,000 Seeds)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Implementation Shortfall (bps relative to P0)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.axvline(0, color="black", linestyle="--", alpha=0.5, label="Arrival Price P0")

    ax2 = axes[1]
    if c51_net is None:
        c51_net = DuelingC51Network()

    c51_net.eval()
    support = c51_net.support.cpu().numpy()

    with torch.no_grad():
        s_high_obi = torch.tensor([[0.5, 0.5, 2.0, 0.8, 1.0, 0.0]], dtype=torch.float32)
        probs_high = c51_net(s_high_obi)[0, 2].cpu().numpy()

        s_low_obi = torch.tensor([[0.5, 0.5, 2.0, -0.8, 1.0, 0.0]], dtype=torch.float32)
        probs_low = c51_net(s_low_obi)[0, 2].cpu().numpy()

    ax2.bar(support - 0.15, probs_high, width=0.3, alpha=0.7, color="#2ca02c", label="High OBI (+0.8, Buy Pressure)")
    ax2.bar(support + 0.15, probs_low, width=0.3, alpha=0.7, color="#d62728", label="Low OBI (-0.8, Sell Pressure)")

    ax2.set_title("C51 51-Atom Return Distribution Z(s, a=TWAP) vs. OBI", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Support Atoms z_i (bps)", fontsize=11)
    ax2.set_ylabel("Categorical Probability Mass", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Benchmark figure saved to: {output_path}")


def run_paired_benchmark(
    num_seeds: int = 1_000,
    start_seed: int = 10_000,
    dqn_model_path: str | None = "checkpoints/dqn_baseline.pt",
    c51_model_path: str | None = "checkpoints/rainbow_c51.pt",
    train_if_missing: bool = True,
    output_plot_path: str = "benchmark_distributions.png",
) -> tuple[dict[str, dict[str, Any]], str]:
    """
    Executes paired-seed out-of-sample evaluation across all 4 policies.
    """
    seeds = list(range(start_seed, start_seed + num_seeds))
    env = LOBExecutionEnv()

    # 1. TWAP Policy Baseline (Action Index 2)
    twap_selector = lambda obs: 2
    twap_results = evaluate_policy_on_seeds(env, twap_selector, seeds)

    # 2. Standard Double Dueling DQN Policy
    dqn_agent = DoubleDQNAgent(env)
    if dqn_model_path and os.path.exists(dqn_model_path):
        print(f"Loading Double DQN weights from {dqn_model_path}...")
        dqn_agent.load(dqn_model_path)
    elif train_if_missing:
        print("Training Standard Double DQN baseline (fast demo run)...")
        dqn_agent.train(total_episodes=300, warmup_episodes=30, epsilon_decay_episodes=200)
        if dqn_model_path:
            dqn_agent.save(dqn_model_path)

    dqn_agent.online_net.eval()
    dqn_selector = lambda obs: dqn_agent.online_net.get_action(torch.as_tensor(obs, dtype=torch.float32))
    dqn_results = evaluate_policy_on_seeds(env, dqn_selector, seeds)

    # 3 & 4. C51 Rainbow Policies (Expected Value & CVaR_0.25)
    c51_agent = RainbowC51Agent(env)
    if c51_model_path and os.path.exists(c51_model_path):
        print(f"Loading Rainbow C51 weights from {c51_model_path}...")
        c51_agent.load(c51_model_path)
    elif train_if_missing:
        print("Training Distributional Rainbow C51 agent (fast demo run)...")
        c51_agent.train(total_episodes=300, warmup_episodes=30, epsilon_decay_episodes=200)
        if c51_model_path:
            c51_agent.save(c51_model_path)

    c51_agent.online_net.eval()
    c51_expected_selector = lambda obs: c51_agent.online_net.get_action(
        torch.as_tensor(obs, dtype=torch.float32), mode="expected"
    )
    c51_cvar_selector = lambda obs: c51_agent.online_net.get_action(
        torch.as_tensor(obs, dtype=torch.float32), mode="cvar", alpha=0.25
    )

    c51_expected_results = evaluate_policy_on_seeds(env, c51_expected_selector, seeds)
    c51_cvar_results = evaluate_policy_on_seeds(env, c51_cvar_selector, seeds)

    results = {
        "TWAP Baseline": twap_results,
        "Standard Double DQN": dqn_results,
        "C51 Rainbow (E[Z])": c51_expected_results,
        "C51 Rainbow (CVaR_0.25)": c51_cvar_results,
    }

    markdown_table = format_markdown_table(results)
    plot_benchmark_distributions(results, c51_agent.online_net, output_plot_path)

    return results, markdown_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LOB execution policies on paired seeds.")
    parser.add_argument("--num_seeds", type=int, default=1000, help="Number of paired OOS seeds (default: 1000)")
    parser.add_argument("--start_seed", type=int, default=10000, help="Initial seed (default: 10000)")
    parser.add_argument("--dqn_path", type=str, default="checkpoints/dqn_baseline.pt", help="Path to trained Double DQN checkpoint")
    parser.add_argument("--c51_path", type=str, default="checkpoints/rainbow_c51.pt", help="Path to trained C51 checkpoint")
    parser.add_argument("--plot_path", type=str, default="benchmark_distributions.png", help="Path for output plot")
    args = parser.parse_args()

    results, md_table = run_paired_benchmark(
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        dqn_model_path=args.dqn_path,
        c51_model_path=args.c51_path,
        output_plot_path=args.plot_path,
    )
    print("\nBenchmark Evaluation Results (Paired Seeds):\n")
    print(md_table)
