# Paired Out-of-Sample Benchmark Methodology

Evaluating execution algorithms on uncoupled random paths confounds execution slippage with underlying market price drift.

## Protocol
1. **Paired Seeds:** All 4 candidate policies (TWAP, Double DQN, C51 $\mathbb{E}[Z]$, C51 $\text{CVaR}_{0.25}$) are evaluated on identical environment seeds:
   $$\text{Seed Range} \in [10000, 10999] \quad (1000 \text{ paired episodes})$$
2. **Identical State Sequences:** For each seed, exogenous order arrivals, drift shocks, and initial book state remain identical across all policy runs.
3. **Evaluation Metrics:**
   - **Mean Implementation Shortfall (IS in bps):** Average price difference between arrival price and realized execution.
   - **IS Standard Deviation (bps):** Volatility of execution shortfall across episodes.
   - **Execution Sharpe Ratio:** $\frac{\text{Mean IS}}{\text{Std Dev IS}}$.
   - **95% CVaR (bps):** Average shortfall in the worst 5% quantile of evaluation runs.
