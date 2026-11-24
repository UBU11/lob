# Prioritized Experience Replay via SumTree

Prioritized Experience Replay (PER) samples transitions proportionally to temporal-difference error magnitudes.

## Data Structure
A binary SumTree stores priorities in leaf nodes with internal nodes maintaining branch sums:
- Priority value: $p_i = (\mathcal{L}_{\text{KL},i} + \epsilon)^\alpha$
- Sampling probability: $P(i) = \frac{p_i}{\sum_k p_k}$
- Importance sampling correction weight:
  $$w_i = \left( N_{\text{buffer}} \cdot P(i) \right)^{-\beta} / \max_k w_k$$

The exponent $\beta$ anneals from $\beta_0 = 0.4$ to $1.0$ over training to compensate for non-uniform sampling bias.
