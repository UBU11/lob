# Categorical Distributional Reinforcement Learning (C51)

Rather than estimating scalar expected returns $Q(s, a) = \mathbb{E}[R]$, the C51 algorithm models the full return distribution $Z(s, a)$ over a discrete support:

$$\mathcal{Z} = \{ z_i = V_{\min} + i \cdot \Delta z \}_{i=0}^{N-1}, \quad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}$$

Here $N = 51$, $V_{\min} = -25.0$ bps, and $V_{\max} = 10.0$ bps.

## Categorical Projection Operator
Under Bellman operator $\hat{\mathcal{T}} z_j = \text{clip}(r + \gamma z_j, V_{\min}, V_{\max})$, shifted atoms typically fall between support points. The projection operator $\Phi$ distributes probability mass to adjacent support atoms:

$$b_j = \frac{\hat{\mathcal{T}} z_j - V_{\min}}{\Delta z}, \quad l = \lfloor b_j \rfloor, \quad u = \lceil b_j \rceil$$

$$(\Phi \hat{\mathcal{T}} p)_l \mathrel{+}= p_j(s', a^*) \cdot (u - b_j)$$
$$(\Phi \hat{\mathcal{T}} p)_u \mathrel{+}= p_j(s', a^*) \cdot (b_j - l)$$

When $l = u$, the full mass $p_j(s', a^*)$ is assigned directly to atom $l$.

## Loss Function
Network optimization minimizes Kullback-Leibler divergence between projected target distribution $m$ and predicted distribution $p(s, a)$:

$$\mathcal{L}_{\text{C51}} = D_{\text{KL}}(m \parallel p(s, a)) = - \sum_{i=0}^{N-1} m_i \log p_i(s, a)$$
