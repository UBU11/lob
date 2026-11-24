# API Reference

## Environment: `lob_rainbow_execution.env_lob.LOBExecutionEnv`
Gymnasium-compatible limit order book liquidation environment.

### Initialization Parameters
- `initial_inventory` (float, default: 10000.0): Total parent order shares $Q_0$.
- `horizon` (int, default: 20): Total discrete liquidation steps $T$.
- `num_levels` (int, default: 5): Visible order book levels per side $D$.
- `tick_size` (float, default: 0.01): Discrete price increment in dollars.
- `arrival_price` (float, default: 100.00): Mid-price at order arrival $P_0$.
- `lambda_walk` (float, default: 0.02): Penalty coefficient for walking book depth.
- `lambda_term` (float, default: 0.5): Penalty coefficient for terminal liquidation spikes.

### Methods
- `reset(seed=None, options=None)` -> `tuple[np.ndarray, dict]`: Reset environment state.
- `step(action: int)` -> `tuple[np.ndarray, float, bool, bool, dict]`: Execute order slice.

## Policy: `lob_rainbow_execution.rainbow_c51_lob.RainbowC51Agent`
Single-file PyTorch implementation of Categorical Distributional DQN with PER and Dueling streams.
