"""Generate architectural flow diagram for LOB RL execution framework."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_figure(output_path: str = "assets/system_architecture.png") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Colors
    bg_box = "#f8f9fa"
    border_color = "#2c3e50"
    state_color = "#e8f4f8"
    c51_color = "#e8f8f5"
    action_color = "#fef9e7"
    lob_color = "#f4ecf7"

    # Environment Box
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 4.0, 7.0, boxstyle="round,pad=0.2", facecolor=lob_color, edgecolor=border_color, linewidth=1.5))
    ax.text(2.5, 7.1, "LOB Execution Environment", fontsize=12, fontweight="bold", ha="center")
    ax.text(2.5, 6.4, "5-Level Limit Order Book\nTransient Depth Replenishment (κ=0.60)\nPermanent Mid-Price Impact (η)\nSequential Book Walking\nInventory Conservation: Σ x_t = Q0", fontsize=9.5, ha="center", va="top")

    # State Vector Box
    ax.add_patch(patches.FancyBboxPatch((5.0, 4.3), 3.8, 3.2, boxstyle="round,pad=0.2", facecolor=state_color, edgecolor=border_color, linewidth=1.5))
    ax.text(6.9, 7.1, "State Vector s_t ∈ R^6", fontsize=12, fontweight="bold", ha="center")
    state_desc = (
        "1. Time Remaining: (T - t) / T\n"
        "2. Inventory Remaining: q_t / Q0\n"
        "3. Bid-Ask Spread (ticks)\n"
        "4. Order Book Imbalance (OBI)\n"
        "5. Top-of-Book Depth Ratio\n"
        "6. Mid-Price Drift (bps)"
    )
    ax.text(6.9, 6.4, state_desc, fontsize=9.5, ha="center", va="top")

    # Agent Network Box
    ax.add_patch(patches.FancyBboxPatch((9.3, 3.8), 4.2, 3.7, boxstyle="round,pad=0.2", facecolor=c51_color, edgecolor=border_color, linewidth=1.5))
    ax.text(11.4, 7.1, "Dueling C51 Rainbow Agent", fontsize=12, fontweight="bold", ha="center")
    c51_desc = (
        "Shared Torso (128-dim + LayerNorm)\n"
        "Value Stream: V(s) ∈ R^51\n"
        "Advantage Stream: A(s, a) ∈ R^(5 x 51)\n"
        "Support: 51 Atoms [-25.0, +10.0] bps\n"
        "Distributional Projection Operator\n"
        "Prioritized Experience Replay (SumTree)"
    )
    ax.text(11.4, 6.4, c51_desc, fontsize=9.5, ha="center", va="top")

    # Policy Action Selection Box
    ax.add_patch(patches.FancyBboxPatch((9.3, 0.5), 4.2, 2.7, boxstyle="round,pad=0.2", facecolor=action_color, edgecolor=border_color, linewidth=1.5))
    ax.text(11.4, 2.8, "Action & Risk Selection", fontsize=12, fontweight="bold", ha="center")
    act_desc = (
        "Discrete Multipliers: [0.0, 0.5, 1.0, 1.5, 2.0]\n"
        "1. Risk-Neutral: a* = argmax E[Z(s, a)]\n"
        "2. Risk-Averse: a* = argmax CVaR_0.25(Z)\n"
        "Mandatory Terminal Liquidation at t=T"
    )
    ax.text(11.4, 2.2, act_desc, fontsize=9.5, ha="center", va="top")

    # Arrows
    arrow_kw = dict(arrowstyle="->", lw=2, color="#34495e")
    ax.annotate("", xy=(5.0, 5.9), xytext=(4.5, 5.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.3, 5.9), xytext=(8.8, 5.9), arrowprops=arrow_kw)
    ax.annotate("", xy=(11.4, 3.2), xytext=(11.4, 3.8), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.5, 1.8), xytext=(9.3, 1.8), arrowprops=arrow_kw)

    ax.text(6.9, 2.1, "Execute Order Slice x_t", fontsize=10, fontweight="bold", ha="center", color="#2c3e50")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved architecture diagram to: {output_path}")

if __name__ == "__main__":
    generate_architecture_figure()
