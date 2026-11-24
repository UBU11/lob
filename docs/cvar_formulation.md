# Conditional Value-at-Risk (CVaR) Objective

Institutional trade execution requires managing tail risk in adverse market regimes. Distributional RL provides access to the complete return probability density $Z(s, a)$, enabling risk-averse execution policies.

## Definition
For tail confidence level $\alpha \in (0, 1]$ (e.g., $\alpha = 0.25$):

$$\text{VaR}_\alpha(Z) = \inf \{ z \in \mathbb{R} : F_Z(z) \ge \alpha \}$$

$$\text{CVaR}_\alpha(Z) = \mathbb{E}[Z \mid Z \le \text{VaR}_\alpha(Z)] = \frac{1}{\alpha} \int_0^\alpha F_Z^{-1}(u) du$$

## Discrete Atom Computation
Given discrete atom probabilities $p_i$ and locations $z_i$:
1. Accumulate cumulative distribution $C_k = \sum_{i=0}^k p_i$.
2. Integrate atom values up to cumulative mass $\alpha$.
3. Normalize by $\alpha$ to obtain expected return in the worst $\alpha$-quantile tail.

Actions are selected to maximize tail return:
$$a^* = \arg\max_a \text{CVaR}_\alpha(Z(s, a))$$
