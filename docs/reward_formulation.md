# Implementation Shortfall Reward Formulation

The single-step reward $r_t$ is denominated in basis points (bps) relative to the initial arrival mid-price $P_0$:

$$r_t = \frac{x_t}{Q_0} \left( \frac{\bar{P}_{\text{exec},t} - P_0}{P_0} \right) \times 10^4 - \lambda_{\text{walk}} (\text{levels\_walked})^2 - \lambda_{\text{term}} \mathbb{I}(t = T \land x_T > 1.5 x_{\text{TWAP}}) \left(\frac{x_T}{x_{\text{TWAP}}}\right)^2$$

## Terms
1. **Execution Shortfall Component:**
   Measures price realization relative to arrival price $P_0$, weighted by fraction of parent order filled at step $t$.
2. **Book Walking Slippage Penalty:**
   Quadratic penalty scaling with order book levels traversed, penalizing aggressive market orders that consume deep liquidity.
3. **Terminal Inventory Liquidation Penalty:**
   Applied strictly at $t = T$ when residual volume exceeds $1.5 \times x_{\text{TWAP}}$, incentivizing smooth schedule completion without terminal fire-sale dumping.

The cumulative undiscounted return $\sum_{t=1}^T r_t$ corresponds directly to total episodic implementation shortfall minus execution friction penalties.
