## 1. Current Research Landscape & Architectural Shifts

Recent quantitative finance and market microstructure literature has exposed three critical failure modes in naive RL execution setups that you must account for in your design:

* **Residual Action Formulations & Paired Path Evaluation (Tosi, Barbierato, & Gatti, Oct 2026):** Unconstrained RL policies trained on order books frequently learn erratic, all-or-nothing liquidation schedules and overfit to historical drift. Recent benchmarks demonstrate that evaluating an RL agent requires **paired out-of-sample price paths** to isolate execution friction from background market drift. Furthermore, formulating the action space around or alongside a baseline schedule (or bounding trade-through depth penalties) significantly stabilizes training across random seeds.
* **Joint Market/Limit Order Allocation & Queue Mechanics (Cheridito & Weiss, Jan–Mar 2026):** Modern LOB execution research has moved beyond market-order-only (Almgren-Chriss style) assumptions. If your agent posts passive limit orders or slices aggressive market orders, you must account for both **direct impact** (consuming visible depth across ticks) and **indirect impact** (order flow toxicity and tactical traders reacting to your order book imbalance).
* **Hardware-Accelerated Microstructure Replay (JAX-LOB / ABIDES-Gym / Hawkes Simulators, 2025–2026):** Because static historical LOB data (like LOBSTER) does not naturally react after your agent depletes a price level, current engineering standards either inject synthetic transient resilience (exponential recovery of depleted depth) or use stochastic intensity models (Hawkes processes) so the agent cannot exploit static replay artifacts.

---

## 2. System Architecture (CleanRL Single-File Philosophy)

Following CleanRL's design philosophy, avoid deep object-oriented inheritance trees. Keep your code modularized into three flat, self-contained files so every tensor shape and mathematical transformation is transparent:

```
lob_rainbow_execution/
├── env_lob.py           # Custom Gymnasium LOB Execution Environment + TWAP Baseline
├── rainbow_c51_lob.py   # Single-file PyTorch C51 + Dueling + Double + PER implementation
└── evaluate_risk.py     # Paired-path evaluation: IS (bps), Sharpe, and 95% CVaR metrics

```

---

## 3. Worked Numerical Example: A Single Step in the LOB

To make the math concrete before writing code, trace a single decision step at $t = 2$ of a $T = 5$ horizon where the agent must liquidate an initial inventory $Q_0 = 1,000$ shares.

### Step A: Observing the State Vector $s_t$

Suppose at $t=2$, the agent still holds $q_t = 600$ shares. The arrival mid-price at $t=0$ was $P_0 = 100.00\text{ USD}$. The current top-3 Bid levels (since we are **selling** into the bid book) and Ask levels are:

* **Bid Book (Prices / Volumes):** Level $L_1$: (99.98 USD, 200 shares), Level $L_2$: (99.96 USD, 300 shares), Level $L_3$: (99.93 USD, 500 shares)
* **Ask Book (Prices / Volumes):** Level $L_1$: (100.02 USD, 100 shares), Level $L_2$: (100.04 USD, 150 shares), Level $L_3$: (100.06 USD, 250 shares)

We compute the normalized state vector $s_t \in \mathbb{R}^6$:

1. **Remaining Time Fraction ($\tau_t$):** $\frac{T - t}{T} = \frac{5 - 2}{5} = 0.60$
2. **Remaining Inventory Fraction ($I_t$):** $\frac{q_t}{Q_0} = \frac{600}{1000} = 0.60$
3. **Spread in Ticks ($S_t$):** $\frac{P_{\text{ask},1} - P_{\text{bid},1}}{\delta_{\text{tick}}} = \frac{100.02 - 99.98}{0.01} = 4.0\text{ ticks}$
4. **Order Book Imbalance ($OBI_t$):**

$$OBI_t = \frac{\sum_{k=1}^3 V_{\text{bid},k} - \sum_{k=1}^3 V_{\text{ask},k}}{\sum_{k=1}^3 V_{\text{bid},k} + \sum_{k=1}^3 V_{\text{ask},k}} = \frac{1000 - 500}{1000 + 500} = +0.333$$



*(A positive imbalance indicates strong buying pressure supporting the bid—favorable for selling).*
5. **Book Depth Ratio ($D_t$):** Ratio of top-level bid volume to average slice size $\frac{V_{\text{bid},1}}{Q_0 / T} = \frac{200}{200} = 1.0$.
6. **Short-Term Mid-Price Drift ($\Delta P_t$):** $\frac{P_{\text{mid},t} - P_{\text{mid},t-1}}{\delta_{\text{tick}}} = \frac{100.00 - 100.01}{0.01} = -1.0$.

Thus, the input tensor fed into the network is `s_t = [0.60, 0.60, 4.0, 0.333, 1.0, -1.0]`.

### Step B: Action Execution & Walking the Book

Suppose our discrete action space $A = \{0, 1, 2, 3, 4\}$ maps to selling $\{0\%, 50\%, 100\%, 150\%, 200\%\}$ of the standard TWAP slice ($\frac{Q_0}{T} = 200$ shares).

* The agent picks action $a_t = 3$, meaning **sell $1.5 \times 200 = 300$ shares immediately via Market Order**.
* **Order Matching:**
* First 200 shares match against Bid level $L_1$ at 99.98 USD (depleting $L_1$ completely).
* Remaining 100 shares walk the book to Bid level $L_2$ and match at 99.96 USD.


* **Average Execution Price ($\bar{P}_{\text{exec}}$):** $\frac{200(99.98) + 100(99.96)}{300} = \$99.9733$.
* **Step Implementation Shortfall Reward ($r_t$):** Measured in basis points (bps) relative to the arrival mid-price $P_0 = 100.00\text{ USD}$, minus a quadratic trade-through penalty $\alpha$ for walking past level $L_1$:

$$r_t = \frac{x_t}{Q_0} \cdot \left(\frac{\bar{P}_{\text{exec}} - P_0}{P_0}\right) \times 10^4 - \alpha \cdot (\text{levels walked})^2$$


$$r_t = \frac{300}{1000} \cdot \left(\frac{99.9733 - 100.00}{100.00}\right) \times 10^4 - 0.05(1)^2 = 0.3 \times (-2.67\text{ bps}) - 0.05 = -0.851\text{ bps}$$



### Step C: C51 Categorical Projection ($\Phi \hat{\mathcal{T}} Z(s', a^*)$)

Instead of outputting a single scalar $Q(s_t, a_t)$, the C51 network outputs a probability distribution over $N=51$ fixed support atoms $z_i \in [V_{\min}, V_{\max}]$, say $[-20.0\text{ bps}, +5.0\text{ bps}]$ with spacing $\Delta z = \frac{25}{50} = 0.5\text{ bps}$.

1. **Double DQN Action Selection:** Use the **online network** to select the greedy next action $a^*$ at $s_{t+1}$ (either via expected value $\sum_i z_i p_i(s_{t+1}, a)$ or via $\text{CVaR}_\alpha$).
2. **Distributional Shift & Scale:** Using the **target network's** probabilities $p_j(s_{t+1}, a^*)$ for each atom $z_j$, shift the support by the observed reward $r_t = -0.851$ and discount $\gamma = 0.99$:

$$\hat{\mathcal{T}} z_j = \text{clip}(r_t + \gamma z_j, V_{\min}, V_{\max}) = \text{clip}(-0.851 + 0.99 z_j, -20.0, 5.0)$$


3. **Linear Interpolation onto Adjacent Atoms:** For atom $z_j = 0.0$, $\hat{\mathcal{T}} z_j = -0.851$. Since $-0.851$ falls between atom $z_{38} = -1.0$ and atom $z_{39} = -0.5$, we distribute its probability mass $p_j(s_{t+1}, a^*)$ proportionally:
* Index position $b_j = \frac{-0.851 - (-20.0)}{0.5} = 38.298 \implies l = 38, u = 39$.
* Atom $l=38$ ($-1.0\text{ bps}$) receives $p_j \times (39 - 38.298) = 0.702 p_j$.
* Atom $u=39$ ($-0.5\text{ bps}$) receives $p_j \times (38.298 - 38) = 0.298 p_j$.


4. **KL-Divergence Loss & PER Priority:** Compute the cross-entropy loss $D_{\text{KL}}(\Phi \hat{\mathcal{T}} Z(s_{t+1}, a^*) \parallel Z(s_t, a_t))$ and use this exact scalar KL loss as the transition priority $p_i = (\mathcal{L}_{\text{KL}} + \epsilon)^\alpha$ in the SumTree replay buffer.

---

## 4. Step-by-Step Implementation Roadmap

1. **Phase 1: Environment Verification & TWAP Baseline (Days 1–3)**
* Implement `LOBExecutionEnv` and write a deterministic `TWAPAgent` that always executes action index `2` ($1.0 \times \text{TWAP slice}$).
* Verify accounting invariants: total shares executed over $t \in [1, T]$ must strictly equal $Q_0 = 10,000$. Log the distribution of episodic Implementation Shortfall in basis points (bps) across 500 fixed evaluation seeds.


2. **Phase 2: Standard Double + Dueling DQN Baseline (Days 4–6)**
* Build the scalar Dueling Double DQN baseline first.
* **Crucial Hyperparameter Check:** Because horizon $T$ is short ($20$ to $60$ steps) and you care equally about cash flows at $t=1$ and $t=T$, set the discount factor $\gamma \in [0.995, 0.9999]$.


3. **Phase 3: Prioritized Experience Replay (PER) & C51 Integration (Days 7–11)**
* Add a proportional `SumTree` buffer ($\alpha = 0.6$, $\beta$ annealed from $0.4 \to 1.0$). In LOB execution, terminal forced-liquidation crashes and deep book walk penalties are rare but catastrophic; PER ensures the agent replays these high-loss tail events frequently.
* Calibrate $[V_{\min}, V_{\max}]$ using your Phase 1 TWAP logs. If episodic rewards range from $-22\text{ bps}$ (99th percentile crash) to $+6\text{ bps}$ (favorable drift), set $V_{\min} = -25.0$ and $V_{\max} = +10.0$. If $[V_{\min}, V_{\max}]$ is set too wide (e.g., $[-1000, 1000]$), all probability mass collapses into 1 or 2 middle atoms, destroying C51's resolution.


4. **Phase 4: Paired Out-of-Sample Benchmarking (Days 12–14)**
* Following the October 2026 evaluation protocol (Tosi et al.), run **TWAP**, **Standard DQN**, **C51 (Expected Value)**, and **C51 ($\text{CVaR}_{0.25}$ Policy)** on the exact same 1,000 random number generator seeds.



---

## 5. Target Benchmark Deliverable Table

Structure your final evaluation report around these exact institutional metrics across 1,000 paired out-of-sample test episodes:

| Policy Architecture | Mean IS (bps) | IS Std Dev (bps) | Execution Sharpe | 95% CVaR (Tail Risk bps) |
| --- | --- | --- | --- | --- |
| **TWAP Baseline** | -6.42 | 4.15 | -1.55 | -14.80 |
| **Standard Double DQN** | -4.85 | 4.60 | -1.05 | -13.90 |
| **C51 Rainbow ($\mathbb{E}[Z]$ Policy)** | -4.10 | 3.75 | -1.09 | -11.20 |
| **C51 Rainbow ($\text{CVaR}_{0.25}$ Policy)** | -4.35 | 2.80 | -1.55 | -8.95 |

* **Implementation Shortfall (IS in bps):** $\frac{\text{Total Cash Received} - Q_0 P_0}{Q_0 P_0} \times 10^4$. Less negative is better (saved transaction costs).
* **95% CVaR (Expected Shortfall):** The average shortfall of the worst 5% of execution episodes. Notice how the $\text{CVaR}_{0.25}$ C51 policy sacrifices $0.25\text{ bps}$ of mean return compared to expected-value C51 in exchange for cutting tail-risk drawdowns from $-11.20\text{ bps}$ to $-8.95\text{ bps}$—which is the primary objective of Distributional RL in institutional execution.