# Empirical Out-of-Sample Benchmark Report

## Experimental Protocol
- **Evaluation Seeds:** 1,000 paired out-of-sample seeds (`seed = 10000 .. 10999`).
- **Initial Inventory:** $Q_0 = 10,000$ shares.
- **Liquidation Horizon:** $T = 20$ timesteps.
- **Initial Arrival Price:** $P_0 = \$100.00$.

## Empirical Performance Summary

| Policy Architecture | Mean IS (bps) | IS Std Dev (bps) | Execution Sharpe | 95% CVaR (Tail Risk bps) |
| :-------------------------- | -------------: | ----------------: | ---------------: | -----------------------: |
| **TWAP Baseline** | -31.79 | 5.28 | -6.02 | -42.09 |
| **Standard Double DQN** | -30.87 | 4.82 | -6.40 | -40.98 |
| **C51 Rainbow (E[Z])** | -11.76 | 8.08 | -1.46 | -27.64 |
| **C51 Rainbow (CVaR_0.25)** | -9.26 | 8.15 | -1.14 | -26.30 |

## Key Empirical Findings
1. **Distributional Superiority:** C51 Rainbow agents reduce mean Implementation Shortfall from -31.79 bps (TWAP) to -9.26 bps, achieving a 22.53 bps improvement in realized execution price.
2. **Tail Risk Mitigation:** Under the risk-averse $\text{CVaR}_{0.25}$ policy, 95% CVaR improves by 15.79 bps over TWAP (-26.30 bps vs. -42.09 bps), effectively mitigating liquidity evaporation and adverse selection penalties.
3. **Adaptive Slicing:** The C51 agent exploits high Order Book Imbalance (favorable buying pressure) to execute larger slices without incurring severe book walking penalties.
