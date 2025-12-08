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
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = gate_up_proj.size(0)

    # Flatten indices for gathering
    expert_ids = top_k_index.reshape(-1)  # (S,)
    tok_idx = torch.arange(num_tokens, device=device).unsqueeze(-1).expand(-1, num_top_k).reshape(-1)  # (S,)
    # top_k_weights can be either:
    #  - per-top-k: shape (num_tokens, num_top_k) -> each column is a top-k position
    #  - per-expert: shape (num_tokens, num_experts) -> each column is an expert
    if top_k_weights.shape == (num_tokens, num_top_k):
        weights = top_k_weights.reshape(-1)
    elif top_k_weights.shape == (num_tokens, num_experts):
        weights = top_k_weights[tok_idx, expert_ids]
    else:
        weights = top_k_weights.reshape(-1)

    # Gather inputs for all selected (token, expert) pairs
    current_states = hidden_states[tok_idx]  # (S, hidden_dim)

    # Compute MoE forward pass for all selected pairs
    # Up projection: gate_up_proj (num_experts, 2*intermediate_dim, hidden_dim) -> (S, 2*intermediate_dim, hidden_dim)
    gate_up = gate_up_proj[expert_ids]  # (S, 2*intermediate_dim, hidden_dim)
    gate, up = (
        torch.bmm(gate_up, current_states.unsqueeze(-1)).squeeze(-1).chunk(2, dim=-1)
    )  # gate: (S, intermediate_dim), up: (S, intermediate_dim)

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
