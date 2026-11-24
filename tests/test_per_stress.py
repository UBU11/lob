import numpy as np
from lob_rainbow_execution.rainbow_c51_lob import SumTree


def test_sumtree_randomized_updates() -> None:
    capacity = 256
    tree = SumTree(capacity)
    rng = np.random.default_rng(2026)

    # Populate tree
    priorities = rng.uniform(0.1, 10.0, size=capacity)
    for p in priorities:
        tree.add(float(p))

    # Verify total
    leaf_values = tree.tree[capacity - 1 : 2 * capacity - 1]
    assert np.isclose(tree.total(), np.sum(leaf_values), atol=1e-5)

    # Perform updates
    for _ in range(100):
        idx = int(rng.integers(0, capacity))
        new_p = float(rng.uniform(0.1, 15.0))
        tree_idx = idx + capacity - 1
        tree.update(tree_idx, new_p)

    updated_leaves = tree.tree[capacity - 1 : 2 * capacity - 1]
    assert np.isclose(tree.total(), np.sum(updated_leaves), atol=1e-5)
