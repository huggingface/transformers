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

    # Flatten selected (token, top-k position) pairs into a single list of samples S
    # expert_ids: (S,)  - expert id for each selected sample
    # token_idx:  (S,)  - original token index for each selected sample
    expert_ids = top_k_index.reshape(-1)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    # Resolve routing weights per selected sample:
    # allow top_k_weights to be either (num_tokens, num_top_k) or (num_tokens, num_experts)
    if top_k_weights.shape == (num_tokens, num_top_k):
        sample_weights = top_k_weights.reshape(-1)  # (S,)
    elif top_k_weights.shape == (num_tokens, num_experts):
        sample_weights = top_k_weights[token_idx, expert_ids]  # (S,)
    else:
        raise ValueError(
            f"top_k_weights has an invalid shape. Should be either ({num_tokens}, {num_top_k}) "
            f"or ({num_tokens}, {num_experts}), but got {top_k_weights.shape}."
        )

    # Gather the hidden states for each selected sample: current_state ~ (S, hidden_dim)
    current_states = hidden_states[token_idx]

    # --- Up projection per expert (batched) ---
    # gate_up_proj has shape (num_experts, 2*intermediate_dim, hidden_dim)
    # select the expert-specific up projections for each sample: (S, 2*intermediate_dim, hidden_dim)
    selected_gate_up = gate_up_proj[expert_ids]

    # Perform the linear: (S, 2*intermediate_dim) = bmm((S, 2*intermediate_dim, hidden_dim), (S, hidden_dim, 1))
    gate_up_out = torch.bmm(selected_gate_up, current_states.unsqueeze(-1)).squeeze(-1)

    # Split into gate and up components to match eager implementation
    gate, up = gate_up_out.chunk(2, dim=-1)  # both have shape (S, intermediate_dim)

    # Apply activation to gate and combine with up projection
    hidden_after_activation = act_fn(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (batched) ---
    # down_proj has shape (num_experts, hidden_dim, intermediate_dim)
    selected_down = down_proj[expert_ids]  # (S, hidden_dim, intermediate_dim)
    out_per_sample = torch.bmm(selected_down, hidden_after_activation.unsqueeze(-1)).squeeze(-1)  # (S, hidden_dim)

    # Apply routing weights and cast to output dtype
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1).to(out_per_sample.dtype)  # (S, hidden_dim)

    # Accumulate results back to the final_hidden_states using original token indices
    final_hidden_states.index_add_(0, token_idx, out_per_sample.to(final_hidden_states.dtype))

    return final_hidden_states
