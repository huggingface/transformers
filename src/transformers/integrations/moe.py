from collections.abc import Callable

import torch


def eager_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    num_experts = gate_up_proj.size(0)
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def batched_mm_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: Callable[..., torch.Tensor],
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


def _pad_dim_end(t: torch.Tensor, dim: int, pad_elems: int):
    if pad_elems == 0:
        return t
    new_shape = list(t.shape)
    new_shape[dim] += pad_elems
    padded = t.new_zeros(*new_shape)
    idx = [slice(None)] * t.dim()
    idx[dim] = slice(0, t.shape[dim])
    padded[tuple(idx)] = t
    return padded


def _make_stride_multiple_of(t: torch.Tensor, dim: int, multiple: int):
    stride = t.stride(dim)
    if stride % multiple == 0:
        return t
    elem_size = t.element_size()
    align_elems = max(1, multiple // elem_size)
    k = t.shape[dim]
    pad = (-k) % align_elems
    return _pad_dim_end(t, dim, pad)


def grouped_mm_moe_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn: Callable[..., torch.Tensor],
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

    # --- Up projection per expert (grouped_mm) ---
    # Group by expert (stable sort to keep deterministic behavior)
    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.argsort(perm, stable=True)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_states[perm].contiguous()

    # tokens per expert -> offsets for grouped_mm (int32)
    num_tokens_per_expert = torch.bincount(expert_ids_g, minlength=num_experts)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Important: torch._grouped_mm requires mat_a.dtype == out_dtype when out_dtype is provided.
    # Important: torch._grouped_mm requires mat_a and mat_b to have strides that are multiples of 16
    # still can't find a reference for this constraint but I had models failing if not respected
    mat_a_up = current_states_g
    mat_b_up = gate_up_proj.transpose(-2, -1)

    if mat_a_up.stride(1) % 16 != 0:
        mat_a_up = _make_stride_multiple_of(mat_a_up, 1, 16)
    if mat_b_up.stride(1) % 16 != 0:
        mat_b_up = _make_stride_multiple_of(mat_b_up, 1, 16)

    gate_up_out = torch._grouped_mm(mat_a_up, mat_b_up, offs=offsets).to(current_states_g.dtype)

    gate, up = gate_up_out.chunk(2, dim=-1)
    hidden_after_activation = act_fn(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (grouped_mm) ---
    mat_a_down = hidden_after_activation
    mat_b_down = down_proj.transpose(-2, -1)

    if mat_a_down.stride(1) % 16 != 0:
        mat_a_down = _make_stride_multiple_of(mat_a_down, 1, 16)
    if mat_b_down.stride(1) % 16 != 0:
        mat_b_down = _make_stride_multiple_of(mat_b_down, 1, 16)

    out_per_sample_g = torch._grouped_mm(mat_a_down, mat_b_down, offs=offsets).to(current_states_g.dtype)

    # apply weights and restore order
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1).to(out_per_sample_g.dtype)
    out_per_sample = out_per_sample_g[inv_perm]

    final_hidden_states.index_add_(0, token_idx, out_per_sample.to(final_hidden_states.dtype))
    return final_hidden_states
