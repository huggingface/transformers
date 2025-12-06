import torch


def bmm_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: callable,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)

    device = hidden_states.device
    num_tokens = hidden_states.size(0)
    tok_idx = torch.arange(num_tokens, device=device).unsqueeze(-1).expand(*top_k_index.shape)

    # Flatten token x top_k pairs
    expert_ids = top_k_index.reshape(-1)  # (S,)
    weights = top_k_weights.reshape(-1)  # (S,)
    tok_idx = tok_idx.reshape(-1)  # (S,)

    # Gather inputs for all selected (token, expert) pairs
    current_states = hidden_states[tok_idx]  # (S, hidden_dim)

    # Compute MoE forward pass for all selected pairs
    # Up projection: gate_up_proj (num_experts, 2*intermediate_dim, hidden_dim) -> (S, 2*intermediate_dim, hidden_dim)
    gate_up = gate_up_proj[expert_ids]  # (S, 2*intermediate_dim, hidden_dim)
    gate_up_out = torch.bmm(gate_up, current_states.unsqueeze(-1)).squeeze(-1)  # (S, 2*intermediate_dim)
    gate, up = gate_up_out.chunk(2, dim=-1)  # each (S, intermediate_dim)

    # Apply activation to gate and combine with up projection
    current_hidden = act_fn(gate) * up  # (S, intermediate_dim)

    # Down projection: down_proj (num_experts, hidden_dim, intermediate_dim) -> (S, hidden_dim, intermediate_dim)
    down = down_proj[expert_ids]  # (S, hidden_dim, intermediate_dim)
    out = torch.bmm(down, current_hidden.unsqueeze(-1)).squeeze(-1)  # (S, hidden_dim)

    # Apply routing weight
    out = out * weights.unsqueeze(-1).to(out.dtype)  # (S, hidden_dim)

    # Accumulate per original token index
    final_hidden_states.index_add_(0, tok_idx, out.to(final_hidden_states.dtype))

    return final_hidden_states
