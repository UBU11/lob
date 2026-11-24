import torch
import numpy as np
from lob_rainbow_execution.rainbow_c51_lob import DuelingC51Network, project_bellman_distribution


def test_categorical_projection_simplex() -> None:
    online_net = DuelingC51Network()
    target_net = DuelingC51Network()
    batch_size = 32

    next_states = torch.randn(batch_size, 6)
    rewards = torch.randn(batch_size) * 5.0
    dones = torch.zeros(batch_size)
    dones[::4] = 1.0

    proj = project_bellman_distribution(online_net, target_net, next_states, rewards, dones)

    # Check sum of probabilities equals 1.0 for every sample
    sums = proj.sum(dim=-1).cpu().numpy()
    np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)
