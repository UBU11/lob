import torch
from lob_rainbow_execution.dqn_baseline import DuelingDQNNetwork


def test_dueling_dqn_tensor_shapes() -> None:
    batch_size = 16
    state_dim = 6
    action_dim = 5
    net = DuelingDQNNetwork(state_dim=state_dim, action_dim=action_dim)
    dummy_input = torch.randn(batch_size, state_dim)
    q_values = net(dummy_input)

    assert q_values.shape == (batch_size, action_dim)
    assert not torch.isnan(q_values).any()
