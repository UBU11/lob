# Institutional Execution Deployment Notes

## Practical Considerations
1. **Latency Constraints:** Inference time for forward distribution evaluation is $< 0.5$ ms on CPU, well within algorithmic execution intervals (1-60 seconds per slice).
2. **Order Routing Integration:** Output action multipliers map to fixed parent-slice allocations dispatched via FIX protocol / Smart Order Router (SOR).
3. **Execution Safety Invariant:** Mandatory terminal liquidation ensures order completion without residual unhedged inventory risk.
