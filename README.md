# Optimal Trade Execution in a Limit Order Book (Distributional Rainbow DQN)

[![Python 3.14](https://img.shields.io/badge/python-3.14%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.15](https://img.shields.io/badge/pytorch-2.15.0-orange.svg)](https://pytorch.org/)
[![Gymnasium 1.4](https://img.shields.io/badge/gymnasium-1.4.0-green.svg)](https://gymnasium.farama.org/)
[![CleanRL Design](https://img.shields.io/badge/architecture-CleanRL%20Single--File-brightgreen.svg)]()

Research-grade institutional trade execution system built from scratch in pure PyTorch and Gymnasium following modern quantitative reinforcement learning and market microstructure standards.

---

## 1. Core Objective

Liquidate a large institutional parent order ($Q_0 = 10,000$ shares) over a fixed discrete horizon ($T = 20$ timesteps) inside a simulated Limit Order Book (LOB) while minimizing **Implementation Shortfall (IS in bps)** and tail risk (**95% CVaR**).

### CleanRL Single-File Architectural Philosophy
Every algorithm file is completely self-contained, transparent, and readable top-to-bottom without deep object-oriented inheritance hierarchies, abstract factory wrappers, or external framework bloat.

### 4 Benchmarked Policies (Paired Out-of-Sample Seeds)
1. **TWAP Baseline:** Deterministic execution slicing $Q_0 / T$ shares every timestep.
2. **Standard Double Dueling DQN:** Risk-neutral expected-value scalar Q-learning baseline.
3. **C51 Rainbow DQN ($\mathbb{E}[Z]$ Policy):** Distributional return modeling optimizing expected return across 51 categorical atoms.
4. **C51 Rainbow DQN ($\text{CVaR}_{0.25}$ Policy):** Risk-averse tail-sensitive distributional policy maximizing returns in the worst 25% quantile tail.

---

## 2. Repository Structure

```text
lob_rainbow_execution/
├── AGENT.md                  # Detailed 2026 architectural & policy specification
├── env_lob.py                # Custom Gymnasium LOB execution environment + TWAP policy
├── dqn_baseline.py           # Single-file Double + Dueling scalar DQN baseline
├── rainbow_c51_lob.py        # Single-file Pure PyTorch C51 + Dueling + Double + PER agent
├── evaluate_benchmark.py     # Paired-seed OOS evaluation (IS bps, Sharpe, 95% CVaR) & plots
├── evaluate_risk.py          # Compatibility alias for evaluate_benchmark.py
└── tests/
    └── test_invariants.py    # Automated checks for Gymnasium API, conservation & C51 math
```

---

## 3. Mathematical & Domain Formulation

### State Space ($s_t \in \mathbb{R}^6$, dimensionless and stationary)
1. `time_rem`: $\frac{T - t}{T} \in [0, 1]$
2. `inv_rem`: $\frac{q_t}{Q_0} \in [0, 1]$
3. `spread_ticks`: $\frac{P_{\text{ask},1}(t) - P_{\text{bid},1}(t)}{\text{tick\_size}}$
4. `obi`: Order Book Imbalance across $D$ visible bid/ask levels $\in [-1, 1]$
5. `l1_depth_ratio`: $\frac{V_{\text{bid},1}(t)}{Q_0 / T}$
6. `mid_drift_bps`: $\frac{P_{\text{mid}}(t) - P_{\text{mid}}(t-1)}{P_0} \times 10^4$

### Action Space (`Discrete(5)`)
Multipliers relative to TWAP slice $x_{\text{TWAP}} = Q_0 / T$:
$$M = [0.0, 0.5, 1.0, 1.5, 2.0]$$
- For $t < T$: $x_t = \min(q_t, M[a_t] \cdot x_{\text{TWAP}})$.
- **Mandatory Terminal Liquidation ($t = T$):** Overridden to $x_T = q_T$ such that $\sum_{t=1}^T x_t \equiv Q_0$ identically on every episode.

### Step Reward (Implementation Shortfall in Basis Points)
$$r_t = \frac{x_t}{Q_0} \left( \frac{\bar{P}_{\text{exec},t} - P_0}{P_0} \right) \times 10^4 - \lambda_{\text{walk}} (\text{levels\_walked})^2 - \lambda_{\text{term}} \mathbb{I}(t = T \land x_T > 1.5 x_{\text{TWAP}}) \left(\frac{x_T}{x_{\text{TWAP}}}\right)^2$$

---

## 4. Invariants & Verification Protocol

The implementation strictly satisfies all four verification invariants defined in `AGENT.md`:
- **Gymnasium 1.4.0 Compliance:** Zero warnings or errors in `gymnasium.utils.env_checker.check_env(env.unwrapped)`.
- **Inventory Conservation:** Identity $\sum_{t=1}^T x_t \equiv Q_0$ holds with absolute tolerance $< 10^{-4}$ across all policies (including random and zero actions).
- **C51 Probability Simplex:** All output distributions and projected Bellman distributions satisfy $\sum_i p_i = 1.0$ (handling integer hits $l == u$ without mass leakage).
- **PER SumTree Invariant:** `tree.total()` strictly equals the sum of leaf priorities after 10,000 insertions and priority updates.

### Running Automated Tests
```bash
pytest -v
```

---

## 5. Running the Benchmark

To run the paired out-of-sample benchmark across all 4 policies on 1,000 paired seeds:
```bash
python -m lob_rainbow_execution.evaluate_benchmark --num_seeds 1000 --start_seed 10000
```
This prints the Markdown summary table and generates `benchmark_distributions.png` comparing the empirical shortfall distributions and C51 atom distributions under high vs. low Order Book Imbalance.
