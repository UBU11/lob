# Market Impact and Resilience Formulation

The environment incorporates both permanent price impact and transient liquidity recovery:

## 1. Permanent Price Impact
Trading volume depletes market liquidity and permanently shifts the mid-price:
$$P_{\text{mid}}(t+1) = P_{\text{mid}}(t) + \Delta P_{\text{drift}} - \eta \left( \frac{x_t}{Q_0} \right) P_0$$
where $\eta$ governs linear permanent impact elasticity.

## 2. Transient Resilience (Depth Replenishment)
Depleted queue volume across bid levels does not remain permanently empty. Between decision steps, depth recovers toward an equilibrium Gamma distribution:
$$V_{\text{bid},k}(t+1) = V_{\text{bid},k}(t) + \kappa \cdot (V_{\text{eq},k} - V_{\text{bid},k}(t)) + \xi_{k,t}$$
where $\kappa \approx 0.60$ represents the mean-reversion rate (resilience speed) and $\xi_{k,t}$ denotes exogenous stochastic order arrivals.
