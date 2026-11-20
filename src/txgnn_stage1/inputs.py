"""Frozen Xavier node inputs for the independently owned pipeline."""
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
