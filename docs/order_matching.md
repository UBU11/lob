# Order Matching Engine and Queue Dynamics

Execution mechanics model sequential matching against resting limit orders across $D$ discrete price levels:

## Book Walking Mechanics
When a market sell order of size $x_t$ is submitted:
1. The matching engine matches against the best bid $P_{\text{bid},1}$ up to available volume $V_{\text{bid},1}$.
2. Residual order volume cascades sequentially to deeper levels $k = 2, \dots, D$.
3. If $x_t > \sum_{k=1}^D V_{\text{bid},k}$, overflow volume executes at a synthetic penalty price $P_{\text{bid},D} - 3 \cdot \text{tick\_size}$.
4. The volume-weighted execution price is computed as:
   $$\bar{P}_{\text{exec},t} = \frac{\sum_k x_{t,k} P_{\text{bid},k}}{x_t}$$
5. The number of levels traversed is logged as $\text{levels\_walked}$, penalizing aggressive orders that deplete book depth.
