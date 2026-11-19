from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class LOBExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_inventory: float = 10_000.0,
        horizon: int = 20,
        num_levels: int = 5,
        tick_size: float = 0.01,
        arrival_price: float = 100.00,
        lambda_walk: float = 0.02,
        lambda_term: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.Q0 = float(initial_inventory)
        self.T = int(horizon)
        self.num_levels = int(num_levels)
        self.tick_size = float(tick_size)
        self.arrival_price = float(arrival_price)
        self.lambda_walk = float(lambda_walk)
        self.lambda_term = float(lambda_term)
        self.twap_slice = self.Q0 / float(self.T)

        self.action_multipliers = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_multipliers))
        low_obs = np.array([0.0, 0.0, 0.0, -1.0, 0.0, -10_000.0], dtype=np.float32)
        high_obs = np.array([1.0, 1.0, 1000.0, 1.0, 1000.0, 10_000.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(6,),
            dtype=np.float32,
        )

        self.t: int = 0
        self.q_t: float = self.Q0
        self.mid_price: float = self.arrival_price
        self.prev_mid_price: float = self.arrival_price
        self.bid_prices: np.ndarray = np.zeros(self.num_levels, dtype=np.float32)
        self.ask_prices: np.ndarray = np.zeros(self.num_levels, dtype=np.float32)
        self.bid_vols: np.ndarray = np.zeros(self.num_levels, dtype=np.float32)
        self.ask_vols: np.ndarray = np.zeros(self.num_levels, dtype=np.float32)
        self.total_shares_sold: float = 0.0
        self.total_cash_earned: float = 0.0

        if seed is not None:
            self.reset(seed=seed)

    def _compute_obi(self) -> float:
        bid_sum = float(np.sum(self.bid_vols))
        ask_sum = float(np.sum(self.ask_vols))
        denom = bid_sum + ask_sum + 1e-8
        return float((bid_sum - ask_sum) / denom)

    def _generate_lob_step(self, temporary_impact: float = 0.0) -> None:
        obi_signal = self._compute_obi()
        noise = float(self.np_random.normal(0.0, 0.02))
        self.mid_price += (0.015 * obi_signal) - temporary_impact + noise
        self.mid_price = max(10.0, float(np.round(self.mid_price, 2)))

        spread_ticks = int(self.np_random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05]))
        half_spread = (spread_ticks * self.tick_size) / 2.0

        best_bid = float(np.round(self.mid_price - half_spread, 2))
        best_ask = float(np.round(best_bid + spread_ticks * self.tick_size, 2))

        self.bid_prices = np.array(
            [best_bid - i * self.tick_size for i in range(self.num_levels)],
            dtype=np.float32,
        )
        self.ask_prices = np.array(
            [best_ask + i * self.tick_size for i in range(self.num_levels)],
            dtype=np.float32,
        )

        target_bid_vols = self.np_random.gamma(shape=2.0, scale=250.0, size=self.num_levels).astype(np.float32)
        target_ask_vols = self.np_random.gamma(shape=2.0, scale=250.0, size=self.num_levels).astype(np.float32)

        self.bid_vols = np.maximum(50.0, 0.4 * self.bid_vols + 0.6 * target_bid_vols).astype(np.float32)
        self.ask_vols = np.maximum(50.0, 0.4 * self.ask_vols + 0.6 * target_ask_vols).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        time_rem = float(max(0.0, (self.T - self.t) / float(self.T)))
        inv_rem = float(max(0.0, min(1.0, self.q_t / self.Q0)))
        spread_ticks = float(max(0.0, (self.ask_prices[0] - self.bid_prices[0]) / self.tick_size))
        obi = float(np.clip(self._compute_obi(), -1.0, 1.0))
        l1_depth_ratio = float(self.bid_vols[0] / (self.twap_slice + 1e-8))
        mid_drift_bps = float(((self.mid_price - self.prev_mid_price) / self.arrival_price) * 1e4)

        return np.array(
            [time_rem, inv_rem, spread_ticks, obi, l1_depth_ratio, mid_drift_bps],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.t = 0
        self.q_t = float(self.Q0)
        self.mid_price = self.arrival_price
        self.prev_mid_price = self.arrival_price
        self.total_shares_sold = 0.0
        self.total_cash_earned = 0.0

        self.bid_vols = self.np_random.gamma(shape=2.0, scale=250.0, size=self.num_levels).astype(np.float32)
        self.ask_vols = self.np_random.gamma(shape=2.0, scale=250.0, size=self.num_levels).astype(np.float32)

        self._generate_lob_step(temporary_impact=0.0)

        info = {
            "arrival_price": float(self.arrival_price),
            "initial_inventory": float(self.Q0),
            "horizon": int(self.T),
        }
        return self._get_obs(), info

    def _execute_sell_order(self, shares_to_sell: float) -> tuple[float, int]:
        if shares_to_sell <= 1e-6:
            return 0.0, 0

        rem = float(shares_to_sell)
        total_cash = 0.0
        levels_walked = 0

        for lvl in range(self.num_levels):
            avail = float(self.bid_vols[lvl])
            filled = min(rem, avail)
            total_cash += filled * float(self.bid_prices[lvl])
            self.bid_vols[lvl] -= filled
            rem -= filled
            if filled > 0 and lvl > 0:
                levels_walked += 1
            if rem <= 1e-5:
                break

        if rem > 1e-5:
            deep_price = float(self.bid_prices[-1] - 3.0 * self.tick_size)
            total_cash += rem * deep_price
            levels_walked += 2

        return total_cash, levels_walked

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.prev_mid_price = self.mid_price
        self.t += 1
        is_terminal = (self.t >= self.T)

        action_int = int(action)
        if is_terminal:
            shares_to_sell = float(self.q_t)
        else:
            desired_shares = float(self.action_multipliers[action_int] * self.twap_slice)
            shares_to_sell = float(min(self.q_t, desired_shares))

        cash_earned, levels_walked = self._execute_sell_order(shares_to_sell)
        self.q_t = float(max(0.0, self.q_t - shares_to_sell))
        if is_terminal:
            self.q_t = 0.0 

        self.total_shares_sold += shares_to_sell
        self.total_cash_earned += cash_earned

        if shares_to_sell > 1e-6:
            avg_exec_price = cash_earned / shares_to_sell
            shortfall_bps = (shares_to_sell / self.Q0) * ((avg_exec_price - self.arrival_price) / self.arrival_price) * 1e4
        else:
            shortfall_bps = 0.0

        walk_penalty = self.lambda_walk * float(levels_walked**2)
        terminal_dump_penalty = (
            self.lambda_term * float((shares_to_sell / self.twap_slice) ** 2)
            if (is_terminal and shares_to_sell > 1.5 * self.twap_slice)
            else 0.0
        )
        reward = float(shortfall_bps - walk_penalty - terminal_dump_penalty)

        impact = 0.03 * (shares_to_sell / (self.twap_slice + 1e-8))
        self._generate_lob_step(temporary_impact=impact)

        terminated = bool(is_terminal or (self.q_t <= 1e-5))
        truncated = False

        info = {
            "shares_sold": float(shares_to_sell),
            "cash_earned": float(cash_earned),
            "shortfall_bps": float(shortfall_bps),
            "remaining_inv": float(self.q_t),
            "levels_walked": int(levels_walked),
            "total_shares_sold": float(self.total_shares_sold),
            "total_cash_earned": float(self.total_cash_earned),
            "step": int(self.t),
        }
        return self._get_obs(), reward, terminated, truncated, info


class TWAPPolicy:

    def __init__(self, action_idx: int = 2) -> None:
        self.action_idx = action_idx

    def select_action(self, obs: np.ndarray) -> int:
        return self.action_idx

    def evaluate(
        self,
        env: LOBExecutionEnv,
        num_episodes: int = 500,
        start_seed: int = 0,
    ) -> dict[str, Any]:
        shortfalls = []
        for ep in range(num_episodes):
            seed = start_seed + ep
            obs, info = env.reset(seed=seed)
            done = False
            ep_cash = 0.0
            while not done:
                action = self.select_action(obs)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                ep_cash += step_info["cash_earned"]

            is_bps = ((ep_cash - env.Q0 * env.arrival_price) / (env.Q0 * env.arrival_price)) * 1e4
            shortfalls.append(is_bps)

        shortfalls_arr = np.array(shortfalls, dtype=np.float32)
        return {
            "mean_is_bps": float(np.mean(shortfalls_arr)),
            "std_is_bps": float(np.std(shortfalls_arr)),
            "min_is_bps": float(np.min(shortfalls_arr)),
            "max_is_bps": float(np.max(shortfalls_arr)),
            "q01_is_bps": float(np.percentile(shortfalls_arr, 1)),
            "q99_is_bps": float(np.percentile(shortfalls_arr, 99)),
            "shortfalls": shortfalls_arr,
        }
