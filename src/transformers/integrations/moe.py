# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from ..utils.generic import GeneralInterface


# Can be removed, only here to show what the eager implementation in modeling files looks like
def eager_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.gate_up_proj.size(0)
    final_hidden_states = torch.zeros_like(hidden_states)

    # Flatten top_k_index to get expert_ids per selected sample
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

    # Get current hidden states for selected samples
    current_hidden_states = hidden_states[token_idx]  # (S, hidden_dim)

    # Select projection matrices for selected experts
    selected_gate_up = self.gate_up_proj[expert_ids]  # (S, hidden_dim, 2 * intermediate_dim)
    selected_down = self.down_proj[expert_ids]  # (S, hidden_dim, intermediate_dim)

    # --- Up projection per expert (batched) ---
    gate_up_out = torch.bmm(selected_gate_up, current_hidden_states.unsqueeze(-1)).squeeze(-1)

    # Split into gate and up components
    gate, up = gate_up_out.chunk(2, dim=-1)  # both have shape (S, intermediate_dim)

    # Apply activation
    hidden_after_activation = self.act_fn(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (batched) ---
    out_per_sample = torch.bmm(selected_down, hidden_after_activation.unsqueeze(-1)).squeeze(-1)

    # Apply routing weights
    out_per_sample = out_per_sample * sample_weights.unsqueeze(-1)  # (S, hidden_dim)

    # Accumulate results back to the final_hidden_states using original token indices
    final_hidden_states.index_add_(0, token_idx, out_per_sample.to(final_hidden_states.dtype))

    return final_hidden_states


def grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    # TODO: we might wanna add more checks here, e.g. check the inputs and weights strides and raise a meaningful error
    if not hasattr(torch, "_grouped_mm"):
        raise ImportError(
            "torch._grouped_mm is not available. Please make sure you are using a PyTorch version that includes it (2.9+)."
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    num_experts = self.gate_up_proj.size(0)
    final_hidden_states = torch.zeros_like(hidden_states)

    # Flatten top_k_index to get expert_ids per selected sample
    expert_ids = top_k_index.reshape(-1)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    # Get permutation to group by expert
    perm = torch.argsort(expert_ids, stable=True)
    inv_perm = torch.argsort(perm, stable=True)

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

    # Get current hidden states for selected samples
    current_hidden_states = hidden_states[token_idx]  # (S, hidden_dim)

    # Group by expert for grouped_mm
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    current_states_g = current_hidden_states[perm]

    # Compute offsets for grouped_mm
    num_tokens_per_expert = torch.bincount(expert_ids_g, minlength=num_experts)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (grouped_mm) ---
    gate_up_out = torch._grouped_mm(current_states_g, self.gate_up_proj.transpose(-2, -1), offs=offsets)

    # Split into gate and up components
    gate, up = gate_up_out.chunk(2, dim=-1)  # both have shape (S, intermediate_dim)

    # Apply activation
    hidden_after_activation = self.act_fn(gate) * up  # (S, intermediate_dim)

    # --- Down projection per expert (grouped_mm) ---
    out_per_sample_g = torch._grouped_mm(hidden_after_activation, self.down_proj.transpose(-2, -1), offs=offsets)

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g.unsqueeze(-1)

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results back to the final_hidden_states using original token indices
    final_hidden_states.index_add_(0, token_idx, out_per_sample.to(final_hidden_states.dtype))

    return final_hidden_states


class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts implementations."""

    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
    }


ALL_EXPERTS_FUNCTIONS = ExpertsInterface()


def use_experts_implementation(experts_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
    original_forward = experts_class.forward

    def forward(self, *args, **kwargs):
        experts_forward = original_forward

        if self.config._experts_implementation != "eager":
            experts_forward = ALL_EXPERTS_FUNCTIONS[self.config._experts_implementation]

        return experts_forward(self, *args, **kwargs)

    experts_class.forward = forward
    return experts_class
