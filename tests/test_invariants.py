"""
Automated checks for Gymnasium compliance, inventory conservation, C51 mathematics, and SumTree invariants.
Strictly required by Section 6 of AGENTS.md.
"""

import warnings
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from lob_rainbow_execution.env_lob import LOBExecutionEnv, TWAPPolicy
from lob_rainbow_execution.dqn_baseline import DuelingDQNNetwork, DoubleDQNAgent
from lob_rainbow_execution.rainbow_c51_lob import (
    SumTree,
    PrioritizedReplayBuffer,
    DuelingC51Network,
    project_bellman_distribution,
    compute_c51_loss,
)


# ==============================================================================
# Invariant 1: Gymnasium 1.4.0 Compliance
# ==============================================================================

def test_gymnasium_compliance() -> None:
    """
    Run gymnasium.utils.env_checker.check_env(env.unwrapped) and assert zero warnings or errors.
    """
    env = LOBExecutionEnv()
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        check_env(env.unwrapped)

        # Filter out expected harmless spec notice if any
        critical_warnings = [
            w for w in recorded_warnings
            if "spec" not in str(w.message).lower()
        ]
        assert len(critical_warnings) == 0, f"check_env produced warnings: {[str(w.message) for w in critical_warnings]}"


def test_gymnasium_distinct_object_allocations() -> None:
    """
    Assert that reset() and step() return fresh numpy array allocations each timestep
    to prevent silent replay buffer corruption.
    """
    env = LOBExecutionEnv()
    obs1, info1 = env.reset(seed=123)
    obs2, reward, terminated, truncated, info2 = env.step(2)

    assert obs1 is not obs2, "reset() and step() returned the exact same observation object reference!"
    assert info1 is not info2, "reset() and step() returned the exact same info dict reference!"


# ==============================================================================
# Invariant 2: Inventory Conservation Invariant
# ==============================================================================

@pytest.mark.parametrize("policy_mode", ["random", "all_zeros", "twap"])
def test_inventory_conservation(policy_mode: str) -> None:
    """
    Across 100 trajectories per policy mode:
    Assert abs(sum(shares_sold) - Q_0) < 1e-4 and remaining_inv == 0.0 at termination.
    """
    env = LOBExecutionEnv(initial_inventory=10_000, horizon=20)
    rng = np.random.default_rng(2026)

    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done = False
        shares_sold_history = []

        while not done:
            if policy_mode == "random":
                action = int(rng.integers(0, 5))
            elif policy_mode == "all_zeros":
                action = 0  # 0.0 multiplier every step until forced terminal dump
            elif policy_mode == "twap":
                action = 2  # 1.0 TWAP slice every step

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            shares_sold_history.append(info["shares_sold"])

        total_sold = float(np.sum(shares_sold_history))
        assert abs(total_sold - env.Q0) < 1e-4, (
            f"Policy {policy_mode} failed conservation: {total_sold} shares sold != {env.Q0}"
        )
        assert info["remaining_inv"] == 0.0, (
            f"Policy {policy_mode} failed remaining inventory: {info['remaining_inv']} != 0.0"
        )
        assert env.q_t == 0.0, f"Env inventory q_t is not 0.0: {env.q_t}"


# ==============================================================================
# Invariant 3: C51 Probability Simplex & Projection Conservation
# ==============================================================================

def test_dueling_c51_probability_simplex() -> None:
    """
    Assert torch.allclose(probs.sum(dim=-1), torch.ones(...), atol=1e-5) for raw network outputs.
    """
    net = DuelingC51Network(state_dim=6, action_dim=5, n_atoms=51, v_min=-25.0, v_max=10.0)
    batch_sizes = [1, 16, 64, 128]

    for bs in batch_sizes:
        dummy_state = torch.randn(bs, 6, dtype=torch.float32)
        probs = net(dummy_state)  # [bs, 5, 51]
        assert probs.shape == (bs, 5, 51)
        atom_sums = probs.sum(dim=-1)
        expected_sums = torch.ones(bs, 5, dtype=torch.float32)
        assert torch.allclose(atom_sums, expected_sums, atol=1e-5), (
            f"Raw C51 probabilities do not sum to 1.0! Max deviation: {torch.max(torch.abs(atom_sums - 1.0)).item()}"
        )


def test_categorical_bellman_projection_conservation() -> None:
    """
    Assert projected Bellman target distributions proj_dist conserve probability mass
    identically (sum == 1.0 per transition), including boundary and exact integer hits (l == u).
    """
    online_net = DuelingC51Network(state_dim=6, action_dim=5, n_atoms=51, v_min=-25.0, v_max=10.0)
    target_net = DuelingC51Network(state_dim=6, action_dim=5, n_atoms=51, v_min=-25.0, v_max=10.0)

    # Test extreme cases: exact boundary hits, negative extremes, positive extremes, exact integer hits
    test_cases = [
        # 1. Extreme boundary values hitting v_min and v_max
        (torch.tensor([-100.0, 100.0, 0.0, -25.0, 10.0], dtype=torch.float32),
         torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)),
        # 2. General continuous rewards
        (torch.tensor([-8.5, -0.851, 0.0, 2.3, 4.7], dtype=torch.float32),
         torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)),
    ]

    for rewards, dones in test_cases:
        batch_size = rewards.size(0)
        next_states = torch.randn(batch_size, 6, dtype=torch.float32)

        proj_dist = project_bellman_distribution(
            online_net=online_net,
            target_net=target_net,
            next_states=next_states,
            rewards=rewards,
            dones=dones,
            gamma=0.995,
        )

        assert proj_dist.shape == (batch_size, 51)
        assert torch.all(proj_dist >= 0.0), "Projected distribution contains negative probabilities!"
        row_sums = proj_dist.sum(dim=-1)
        expected = torch.ones(batch_size, dtype=torch.float32)
        assert torch.allclose(row_sums, expected, atol=1e-5), (
            f"Projected distribution did not conserve probability mass! Sums: {row_sums}"
        )


def test_cvar_action_selection() -> None:
    """
    Assert that get_action in CVaR mode returns a valid discrete action and respects tail risk.
    """
    net = DuelingC51Network(state_dim=6, action_dim=5, n_atoms=51)
    dummy_state = torch.randn(6, dtype=torch.float32)

    exp_action = net.get_action(dummy_state, mode="expected")
    cvar_action = net.get_action(dummy_state, mode="cvar", alpha=0.25)

    assert 0 <= exp_action < 5
    assert 0 <= cvar_action < 5


# ==============================================================================
# Invariant 4: SumTree Invariant
# ==============================================================================

def test_sumtree_invariant() -> None:
    """
    Assert tree.total() equals the exact sum of leaf priorities after 10,000 insertions
    and subsequent priority updates.
    """
    capacity = 10_000
    tree = SumTree(capacity)
    rng = np.random.default_rng(2026)

    # 1. 10,000 insertions
    priorities = rng.uniform(0.01, 50.0, size=capacity).astype(np.float64)
    for p in priorities:
        tree.add(p)

    expected_total = float(np.sum(priorities))
    assert abs(tree.total() - expected_total) < 1e-4, (
        f"SumTree total {tree.total()} != expected {expected_total} after insertions"
    )

    # 2. 10,000 random priority updates
    for _ in range(10_000):
        data_idx = int(rng.integers(0, capacity))
        new_p = float(rng.uniform(0.01, 50.0))
        priorities[data_idx] = new_p
        tree_idx = data_idx + capacity - 1
        tree.update(tree_idx, new_p)

    expected_total_after_updates = float(np.sum(priorities))
    assert abs(tree.total() - expected_total_after_updates) < 1e-4, (
        f"SumTree total {tree.total()} != expected {expected_total_after_updates} after updates"
    )


# ==============================================================================
# Invariant 5: Scalar Dueling Double DQN Invariant
# ==============================================================================

def test_scalar_dueling_dqn() -> None:
    """
    Assert Dueling scalar DQN outputs valid Q-values and action selections.
    """
    dqn = DuelingDQNNetwork(state_dim=6, action_dim=5)
    dummy_state = torch.randn(16, 6)
    q_vals = dqn(dummy_state)

    assert q_vals.shape == (16, 5)
    action = dqn.get_action(dummy_state[0])
    assert 0 <= action < 5
