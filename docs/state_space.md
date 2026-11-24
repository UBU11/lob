# State Space Specification

The observation vector $s_t \in \mathbb{R}^6$ represents a dimensionless, stationary encoding of the limit order book state at decision timestep $t$:

1. Remaining Time Fraction ($\tau_t$):
   $$\tau_t = \frac{T - t}{T} \in [0, 1]$$
   Measures time remaining until mandatory terminal liquidation at $t = T$.

2. Remaining Inventory Fraction ($I_t$):
   $$I_t = \frac{q_t}{Q_0} \in [0, 1]$$
   Tracks remaining inventory normalized by parent order size $Q_0$.

3. Bid-Ask Spread in Ticks ($S_t$):
   $$S_t = \frac{P_{\text{ask},1}(t) - P_{\text{bid},1}(t)}{\text{tick\_size}}$$
   Measures instantaneous execution cost and liquidity friction at the top of the book.

4. Multi-Level Order Book Imbalance ($\text{OBI}_t$):
   $$\text{OBI}_t = \frac{\sum_{k=1}^D V_{\text{bid},k}(t) - \sum_{k=1}^D V_{\text{ask},k}(t)}{\sum_{k=1}^D V_{\text{bid},k}(t) + \sum_{k=1}^D V_{\text{ask},k}(t) + \epsilon} \in [-1, 1]$$
   Computed across $D = 5$ visible price levels. Positive values signify excess bid depth (buying pressure); negative values signify excess ask depth (selling pressure).

5. Top-of-Book Depth Ratio ($D_t$):
   $$D_t = \frac{V_{\text{bid},1}(t)}{Q_0 / T}$$
   Normalized ratio between available volume at the best bid and standard single-step TWAP allocation.

6. Mid-Price Drift ($\Delta P_t$):
   $$\Delta P_t = \frac{P_{\text{mid}}(t) - P_{\text{mid}}(t-1)}{P_0} \times 10^4$$
   Recent mid-price movement measured in basis points relative to initial arrival price $P_0$.
