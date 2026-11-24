**Project:** Optimal Trade Execution in a Limit Order Book (Distributional Rainbow DQN)  
**Target Standard:** November 2026 Quantitative RL & Market Microstructure Engineering  

---

## 1. Mission & Architectural Philosophy
You are building a research-grade institutional trade execution system from scratch in pure PyTorch and Gymnasium.
- **Core Objective:** Liquidate a large institutional parent order ($Q_0$) over a fixed discrete horizon ($T$) inside a simulated/historical Limit Order Book (LOB) while minimizing **Implementation Shortfall (IS)** and tail risk (**95% CVaR**).
- **CleanRL Single-File Design:** Follow CleanRL's architectural philosophy. Do **not** build deep OOP inheritance hierarchies, abstract factory classes, or external RL framework wrappers (e.g., Stable-Baselines3, RLlib). Every algorithm file must be self-contained, transparent, and readable top-to-bottom.
- **Deliverable:** A reproducible benchmark comparing four policies on **paired out-of-sample RNG seeds**:
  1. **TWAP** (Time-Weighted Average Price deterministic baseline)
  2. **Standard Double Dueling DQN** (expected-value scalar Q-learning baseline)
  3. **C51 Rainbow DQN ($\mathbb{E}[Z]$ Policy)** (risk-neutral distributional policy)
  4. **C51 Rainbow DQN ($\text{CVaR}_{0.25}$ Policy)** (tail-risk-averse distributional policy)

---

## 2. Mandatory 2026 Tech Stack & Banned Deprecated APIs
You MUST target current 2026 stable APIs. Never generate legacy Gym or deprecated PyTorch syntax.

| Dependency | Minimum Version (2026) | Mandatory API Rules |
| :--- | :--- | :--- |
| **Python** | `>= 3.13` (3.13/3.14 supported) | Use standard built-in type hints (`list[float]`, `tuple[np.ndarray, dict]`). |
| **PyTorch** | `>= 2.15.0` | Use `torch.amp.autocast("cuda")` (if CUDA used), `torch.use_deterministic_algorithms(True)`, and `nn.Module`. |
| **Gymnasium** | `>= 1.4.0` | `step()` MUST return 5-tuple `(obs, reward, terminated, truncated, info)`. `reset()` MUST return `(obs, info)`. |
| **NumPy** | `>= 2.5.0` | Use `np.random.default_rng(seed)` exclusively. |

### Strict Ban List (NEVER Use These)
- **BANNED:** `import gym` (OpenAI Gym is abandoned; always `import gymnasium as gym`).
- **BANNED:** `torch.jit.script` / `torch.jit.trace` (TorchScript is formally deprecated in PyTorch 2.10; use eager PyTorch or `torch.compile()`).
- **BANNED:** Legacy NumPy RNG `np.random.seed()` or `np.random.randn()` inside environment logic (breaks paired-path evaluation).
- **BANNED:** Legacy NumPy aliases (`np.bool8`, `np.float_`, `np.int_`). Use `np.bool_`, `np.float32`, `np.int64`.
- **BANNED:** Reusing/mutating the same NumPy array or `info` dict reference across `step()` calls. Gymnasium `1.4.0` `check_env` strictly verifies that `reset()` and `step()` return distinct object allocations each timestep to prevent silent replay buffer corruption.
- **BANNED:** Reward clipping (`np.clip(reward, -1, 1)`). In C51, reward clipping destroys basis-point return calibration and breaks `[v_min, v_max]` support mapping.

---

## 3. Repository Structure & File Contracts
Keep the codebase strictly organized into five flat files:

```text
lob_rainbow_execution/
├── AGENT.md                  # This architectural & policy specification
├── env_lob.py                # Custom Gymnasium LOB environment + TWAP baseline policy
├── dqn_baseline.py           # Single-file Double + Dueling scalar DQN baseline
├── rainbow_c51_lob.py        # Single-file Pure PyTorch C51 + Dueling + Double + PER agent
├── evaluate_benchmark.py     # Paired-seed OOS evaluation (IS bps, Sharpe, 95% CVaR) & plots
└── tests/
    └── test_invariants.py    # Automated checks for Gymnasium API, inventory conservation & C51 math

```

---

## 4. Domain Specification: Limit Order Book (LOB) MDP

Grounded in 2026 market microstructure research on multi-level queue dynamics, implementation shortfall decomposition, and exact fractional/residual inventory execution:

### 4.1 State Space ($s_t \in \mathbb{R}^6$, `dtype=np.float32`)

All observations MUST be dimensionless, normalized, and stationary across price levels:

1. `time_rem`: $\frac{T - t}{T} \in [0, 1]$ (fraction of horizon remaining).
2. `inv_rem`: $\frac{q_t}{Q_0} \in [0, 1]$ (fraction of parent order remaining).
3. `spread_ticks`: $\frac{P_{\text{ask},1}(t) - P_{\text{bid},1}(t)}{\delta_{\text{tick}}}$ (current bid-ask spread in ticks).
4. `obi`: Multi-level Order Book Imbalance across $D$ visible levels:

$$\text{OBI}_t = \frac{\sum_{k=1}^D V_{\text{bid},k}(t) - \sum_{k=1}^D V_{\text{ask},k}(t)}{\sum_{k=1}^D V_{\text{bid},k}(t) + \sum_{k=1}^D V_{\text{ask},k}(t) + \epsilon} \in [-1, 1]$$


5. `l1_depth_ratio`: $\frac{V_{\text{bid},1}(t)}{Q_0 / T}$ (ratio of top-of-book liquidity to standard TWAP slice).
6. `mid_drift_bps`: $\frac{P_{\text{mid}}(t) - P_{\text{mid}}(t-1)}{P_0} \times 10^4$ (short-term mid-price momentum in basis points).

### 4.2 Action Space (`gymnasium.spaces.Discrete(5)`)

Following recent 2026 execution stabilization findings, parameterize actions relative to the baseline TWAP slice $x_{\text{TWAP}} = Q_0 / T$ rather than arbitrary raw share counts:

* Action multipliers: `M = [0.0, 0.5, 1.0, 1.5, 2.0]`.
* At step $t < T$, desired order size is $x_t = \min(q_t, M[a_t] \cdot x_{\text{TWAP}})$.
* **Mandatory Terminal Constraint ($t = T$):** At the final step $t = T$, the environment MUST override the action and liquidate all remaining inventory ($x_T = q_T$) so that $\sum_{t=1}^T x_t \equiv Q_0$ holds identically on every episode.

### 4.3 Order Matching, Book Walking & Transient Impact

* **Walking the Book:** A market sell order $x_t$ matches sequentially against Bid levels $k = 1 \dots D$ ($P_{\text{bid},1} > P_{\text{bid},2} > \dots > P_{\text{bid},D}$), depleting available queue volume $V_{\text{bid},k}(t)$ at each level. If $x_t > \sum_{k=1}^D V_{\text{bid},k}(t)$, fill the overflow at $P_{\text{bid},D} - 3 \cdot \delta_{\text{tick}}$ and record additional levels walked.
* **Transient Resilience & Permanent Impact:** Static LOB replay without replenishment allows RL agents to exploit unrealistic book states. After each step, apply permanent mid-price impact proportional to executed volume and exponential depth replenishment ($\approx 60\%$ recovery per step toward stochastic Gamma-distributed queue depth).

### 4.4 Reward Function (Implementation Shortfall in Basis Points)

Define step reward $r_t$ directly in **basis points (bps)** relative to the arrival mid-price $P_0$:


$$r_t = \frac{x_t}{Q_0} \left( \frac{\bar{P}_{\text{exec},t} - P_0}{P_0} \right) \times 10^4 - \lambda_{\text{walk}} (\text{levels walked})^2 - \lambda_{\text{term}} \mathbb{I}(t = T \land x_T > 1.5 x_{\text{TWAP}}) \left(\frac{x_T}{x_{\text{TWAP}}}\right)^2$$

* Note: $\sum_{t=1}^T r_t$ equals the total episodic **Implementation Shortfall (bps)** minus quadratic microstructure slippage penalties.

---

## 5. Algorithmic Specification: Distributional Rainbow DQN (`rainbow_c51_lob.py`)

### 5.1 Dueling Categorical Architecture (`DuelingC51Network`)

* **Support Atoms:** $N = 51$ linearly spaced atoms $z_i \in [V_{\min}, V_{\max}]$ with $\Delta z = \frac{V_{\max} - V_{\min}}{50}$.
* **Support Calibration Rule:** Before training C51, run the deterministic TWAP baseline over 500 episodes to measure empirical episodic IS bounds. Set $V_{\min}$ and $V_{\max}$ tightly around those empirical tails (e.g., `v_min = -25.0`, `v_max = 10.0` bps). Never use arbitrary wide bounds like `[-1000, 1000]`.
* **Dueling Fusion in Logit Space:**
* Shared torso: `Linear(6, 128) -> LayerNorm(128) -> ReLU -> Linear(128, 128) -> ReLU`.
* Value stream outputs logits $V_{\text{logits}}(s) \in \mathbb{R}^{B \times 1 \times 51}$.
* Advantage stream outputs logits $A_{\text{logits}}(s, a) \in \mathbb{R}^{B \times |A| \times 51}$.
* Combine **before** softmax:

$$Q_{\text{logits}}(s, a, z_i) = V_{\text{logits}}(s, z_i) + \left( A_{\text{logits}}(s, a, z_i) - \frac{1}{|A|} \sum_{a'} A_{\text{logits}}(s, a', z_i) \right)$$


* Apply `probs = F.softmax(q_logits, dim=-1).clamp(min=1e-6)` across the atom dimension.



### 5.2 Double DQN + Categorical Bellman Projection ($\Phi \hat{\mathcal{T}} Z$)

1. **Double DQN Action Selection:** Select next greedy action $a^*$ using the **online network's** expected Q-values:

$$a^* = \arg\max_{a'} \sum_{i=0}^{50} z_i p_{\theta}(s', a', z_i)$$


2. **Target Projection:** Evaluate $p_{\theta^-}(s', a^*)$ from the **target network** (updated via Polyak soft updates $\tau = 0.005$ or hard sync every 200 steps). Compute shifted support $\hat{\mathcal{T}} z_j = \text{clip}(r + (1 - d)\gamma z_j, V_{\min}, V_{\max})$ with $\gamma = 0.995$.
3. **Linear Interpolation:** Project probability mass $p_{\theta^-}(s', a^*, z_j)$ onto lower/upper neighbor atoms $l = \lfloor b_j \rfloor$ and $u = \lceil b_j \rceil$ where $b_j = (\hat{\mathcal{T}} z_j - V_{\min}) / \Delta z$. Handle exact integer hits ($l == u$) explicitly so probability mass never vanishes.
4. **Loss Function:** Compute sample-wise cross-entropy loss:

$$\mathcal{L}_k = -\sum_{i=0}^{50} m_i^{(k)} \log p_{\theta}(s_k, a_k, z_i)$$



### 5.3 Risk-Sensitive Inference ($\text{CVaR}_\alpha$ Policy)

At evaluation time, support two action-selection modes on the trained C51 network:

* **Risk-Neutral ($\mathbb{E}[Z]$):** $\arg\max_a \sum_{i=0}^{50} z_i p_i(s, a)$.
* **Risk-Averse ($\text{CVaR}_\alpha$, default $\alpha = 0.25$):** Compute the cumulative distribution $F_a(z_i) = \sum_{j \le i} p_j(s, a)$, mask atoms in the lower $\alpha$-quantile ($F_a(z_i) \le \alpha$, always retaining atom $0$), renormalize masked probabilities, and pick the action maximizing expected return in that worst-case tail.

### 5.4 Prioritized Experience Replay (PER)

* Implement an $O(\log N)$ array-backed `SumTree` replay buffer (`capacity = 100_000`, `alpha = 0.6`, `beta` linearly annealed from `0.4` to `1.0` over training).
* Update transition priority directly using the C51 sample cross-entropy loss: $p_k = (\mathcal{L}_k + 10^{-5})^\alpha$.
* Multiply batch losses by normalized Importance Sampling weights $w_k = (N \cdot P(k))^{-\beta} / \max_j w_j$ before backpropagation.

---

## 6. Mandatory Verification & Testing Protocol

Before declaring any milestone complete, you MUST write and pass `pytest tests/test_invariants.py`:

1. **Gymnasium 1.4.0 Compliance:** Run `gymnasium.utils.env_checker.check_env(env.unwrapped)` and assert zero warnings or errors.
2. **Inventory Conservation Invariant:** Across 100 random trajectories (including random actions and all-zero `action=0` policies), assert `abs(sum(shares_sold) - Q_0) < 1e-4` and `remaining_inv == 0.0` at termination.
3. **C51 Probability Simplex & Projection Conservation:** Assert `torch.allclose(probs.sum(dim=-1), torch.ones(...), atol=1e-5)` for both raw network outputs and projected Bellman target distributions `proj_dist` (including boundary hits where $l == u$).
4. **SumTree Invariant:** Assert `tree.total()` equals the exact sum of leaf priorities after 10,000 insertions and priority updates.

---

## 7. Paired-Seed Benchmarking Standard (`evaluate_benchmark.py`)

To isolate execution skill from background market drift, evaluate all four policies across the **exact same 1,000 out-of-sample test seeds** (`seed = 10_000 .. 10_999`):

* Output a Markdown summary table reporting:
* **Mean Implementation Shortfall (bps)** (higher / less negative is better)
* **IS Standard Deviation (bps)**
* **Execution Sharpe Ratio** ($\text{Mean IS} / \text{Std IS}$)
* **95% CVaR (Expected Shortfall in bps)** — mean IS of the worst 5% (50 worst) episodes.


* Save a comparative matplotlib figure (`benchmark_distributions.png`) showing overlaid kernel density / histogram plots of episodic Implementation Shortfall for TWAP vs. Standard DQN vs. C51 ($\text{CVaR}_{0.25}$), plus an example 51-atom probability mass function $Z(s, a)$ under high vs. low Order Book Imbalance.

```

---

### How to Use This File Immediately
1. Create a folder named `lob_rainbow_execution` and save the block above as `AGENT.md`.
2. Open your coding agent in that workspace and give it a sequential, phase-gated prompt:
   > *"Read `AGENT.md` completely. Implement Phase 1 (`env_lob.py` and `tests/test_invariants.py`) strictly following every API rule and invariant in `AGENT.md`, then run `pytest` to verify."*
3. Once `pytest` passes on `env_lob.py`, instruct the agent to implement `dqn_baseline.py`, `rainbow_c51_lob.py`, and `evaluate_benchmark.py` one file at a time so each component is verified against the invariants before moving forward.

```