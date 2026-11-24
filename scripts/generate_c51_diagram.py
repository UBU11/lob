"""Generate schematic diagram of C51 Bellman projection operator."""
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_c51_projection_figure(output_path: str = "assets/c51_projection.png") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True)

    support = np.linspace(-25.0, 10.0, 51)
    target_probs = np.zeros(51)
    target_probs[25:35] = np.exp(-0.5 * ((np.arange(10) - 4.5) / 1.5) ** 2)
    target_probs /= target_probs.sum()

    # Step 1: Target distribution
    ax1.bar(support, target_probs, width=0.5, color="#3498db", alpha=0.8, edgecolor="#2980b9")
    ax1.set_title("1. Target Network Distribution Z(s', a*)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Step 2: Shifted and Projected distribution
    reward = -1.2
    gamma = 0.995
    shifted_support = np.clip(reward + gamma * support, -25.0, 10.0)

    # Simple projection illustration
    proj_probs = np.zeros(51)
    for p, s in zip(target_probs, shifted_support):
        b = (s - (-25.0)) / ((10.0 - (-25.0)) / 50.0)
        l = int(np.floor(b))
        u = int(np.ceil(b))
        if l == u:
            proj_probs[l] += p
        else:
            proj_probs[l] += p * (u - b)
            proj_probs[u] += p * (b - l)

    ax2.bar(support, proj_probs, width=0.5, color="#2ecc71", alpha=0.8, edgecolor="#27ae60")
    ax2.set_title(f"2. Bellman Shift (r={reward:.1f} bps, γ={gamma}) and Projected Distribution ΦTZ", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Return Support Atoms z_i (basis points relative to P0)", fontsize=10)
    ax2.set_ylabel("Projected Probability", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved C51 projection figure to: {output_path}")

if __name__ == "__main__":
    generate_c51_projection_figure()
