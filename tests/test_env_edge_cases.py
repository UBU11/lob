import numpy as np
import pytest
from lob_rainbow_execution.env_lob import LOBExecutionEnv


def test_terminal_liquidation_all_zero_actions() -> None:
    env = LOBExecutionEnv()
    obs, info = env.reset(seed=42)
    done = False
    while not done:
        obs, reward, term, trunc, step_info = env.step(0)  # Always zero action
        done = term or trunc
    assert np.isclose(env.total_shares_sold, env.Q0, atol=1e-3)
    assert np.isclose(env.q_t, 0.0, atol=1e-3)


def test_terminal_liquidation_max_actions() -> None:
    env = LOBExecutionEnv()
    obs, info = env.reset(seed=99)
    done = False
    while not done:
        obs, reward, term, trunc, step_info = env.step(4)  # 2.0x TWAP action
        done = term or trunc
    assert np.isclose(env.total_shares_sold, env.Q0, atol=1e-3)
    assert np.isclose(env.q_t, 0.0, atol=1e-3)


def test_order_book_imbalance_bounds() -> None:
    env = LOBExecutionEnv()
    env.reset(seed=123)
    # Test extreme bid volume dominance
    env.bid_vols = np.array([1000.0] * env.num_levels, dtype=np.float32)
    env.ask_vols = np.array([0.0] * env.num_levels, dtype=np.float32)
    obi = env._compute_obi()
    assert 0.99 <= obi <= 1.0

    # Test extreme ask volume dominance
    env.bid_vols = np.array([0.0] * env.num_levels, dtype=np.float32)
    env.ask_vols = np.array([1000.0] * env.num_levels, dtype=np.float32)
    obi = env._compute_obi()
    assert -1.0 <= obi <= -0.99
