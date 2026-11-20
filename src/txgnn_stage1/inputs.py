"""Frozen Xavier node inputs for the independently owned pipeline.

The NumPy generator is the Step 1 default. It matches the Xavier-uniform
bound sqrt(6 / (N + d)) and is reproducible for a fixed NumPy version.
It does not reproduce PyTorch CPU bit patterns. The Torch helper remains
optional and is not required to build a bundle.
"""
import math
import numpy as np

INPUT_RNG_ALGORITHM = "numpy.random.Generator(PCG64)"


def xavier_uniform_bound(n_rows, n_cols):
    return math.sqrt(6.0 / (n_rows + n_cols))


def make_numpy_inputs(node_counts, node_type_order, input_dim, input_seed):
    """One float32 Xavier-uniform matrix per type, in the given type order."""
    rng = np.random.Generator(np.random.PCG64(int(input_seed)))
    matrices = {}
    for node_type in node_type_order:
        n_rows = int(node_counts[node_type])
        bound = xavier_uniform_bound(n_rows, input_dim)
        drawn = rng.uniform(-bound, bound, size=(n_rows, input_dim))
        matrices[node_type] = np.ascontiguousarray(drawn, dtype=np.float32)
    return matrices


def make_owned_inputs(node_counts, node_type_order, input_dim, input_seed):
    import torch
    caller_state = torch.get_rng_state().clone()
    try:
        torch.manual_seed(input_seed)
        before = torch.get_rng_state().clone()
        matrices = {}
        for node_type in node_type_order:
            values = torch.empty((node_counts[node_type], input_dim), dtype=torch.float32, device='cpu')
            torch.nn.init.xavier_uniform_(values)
            values.requires_grad_(False)
            matrices[node_type] = values
        after = torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(caller_state)
    return matrices, before, after
