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

from functools import wraps

from ..utils.generic import GeneralInterface
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch

# Examples of experts class with its eager mm implementation
# class Experts(nn.Module):
#     """Collection of expert weights stored as 3D tensors."""

#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.n_routed_experts
#         self.hidden_dim = config.hidden_size
#         self.intermediate_dim = config.moe_intermediate_size
#         self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
#         self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         top_k_index: torch.Tensor,
#         top_k_weights: torch.Tensor,
#     ) -> torch.Tensor:
#         final_hidden_states = torch.zeros_like(hidden_states)
#         with torch.no_grad():
#             expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
#             expert_mask = expert_mask.permute(2, 1, 0)
#             expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

#         for expert_idx in expert_hit:
#             expert_idx = expert_idx[0]
#             if expert_idx == self.num_experts:
#                 continue
#             top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
#             current_state = hidden_states[token_idx]
#             gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
#             current_hidden_states = self.act_fn(gate) * up
#             current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
#             current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
#             final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

#         return final_hidden_states


def _batched_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Batched linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (batch_size, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (batch_size, output_dim, input_dim) if transposed is `False`,
            else of shape (batch_size, input_dim, output_dim).
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (batch_size, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (batch_size, output_dim).
    """
    if is_transposed:
        # (batch_size, 1, input_dim) @ (batch_size, input_dim, output_dim) -> (batch_size, 1, output_dim) -> (batch_size, output_dim)
        out = torch.bmm(input.unsqueeze(1), weight).squeeze(1)
    else:
        # (batch_size, output_dim, input_dim) @ (batch_size, input_dim, 1) -> (batch_size, output_dim, 1) -> (batch_size, output_dim)
        out = torch.bmm(weight, input.unsqueeze(-1)).squeeze(-1)

    if bias is not None:
        out = out + bias

    return out


def batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    num_experts = self.gate_up_proj.size(0)

    # Flatten top_k_index to get expert_ids per selected sample
    expert_ids = top_k_index.reshape(-1)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    # Resolve routing weights per selected sample, allowing top_k_weights to be either:
    # - (num_tokens, num_top_k) Qwen2MoE style
    # - (num_tokens, num_experts) DeepseekV2 style
    if top_k_weights.shape == (num_tokens, num_top_k):
        sample_weights = top_k_weights
    elif top_k_weights.shape == (num_tokens, num_experts):
        # TODO: routers that output full expert distribution
        # should probably be corrected to output only top_k weights
        sample_weights = top_k_weights[token_idx, expert_ids]
    else:
        raise ValueError(
            f"top_k_weights has an invalid/unsupported shape. It should be either (num_tokens, num_top_k)({num_tokens}, {num_top_k}) "
            f"or (num_tokens, num_experts)({num_tokens}, {num_experts}), but got {top_k_weights.shape}."
        )
    sample_weights = sample_weights.reshape(-1, 1)  # (S, 1)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Select expert weights and biases for selected samples
    selected_gate_up = self.gate_up_proj[expert_ids]
    selected_down = self.down_proj[expert_ids]
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids] if self.has_bias else None

    # --- Up projection per expert (batched) ---
    gate_up_out = _batched_linear(
        selected_hidden_states, selected_gate_up, selected_gate_up_bias, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (batched) ---
    out_per_sample = _batched_linear(
        gated_out, selected_down, selected_down_bias, is_transposed=self.is_transposed
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample = out_per_sample * sample_weights  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Grouped linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, output_dim, input_dim) if transposed is `False`,
            else of shape (num_experts, input_dim, output_dim).
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (num_experts, output_dim). Default is `None`.
        offs (`torch.Tensor`, *optional*):
            Offsets tensor indicating the boundaries of each group in the input tensor.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """
    if is_transposed:
        # (S, input_dim) @ grouped (num_experts, input_dim, output_dim) -> (S, output_dim)
        out = torch._grouped_mm(input, weight, offs=offs)
    else:
        # (S, input_dim) @ grouped (num_experts, output_dim, input_dim).T -> (S, output_dim)
        out = torch._grouped_mm(input, weight.transpose(-2, -1), offs=offs)

    if bias is not None:
        # We should be able to pass bias to the grouped_mm call, but it's not yet supported.
        out = out + bias

    return out


def grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(torch, "_grouped_mm"):
        raise ImportError(
            "torch._grouped_mm is not available. Please make sure you are using a PyTorch version that includes it (2.9+)."
        )

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    num_experts = self.gate_up_proj.size(0)

    # Flatten top_k_index to get expert_ids per selected sample
    expert_ids = top_k_index.reshape(-1)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)

    # Resolve routing weights per selected sample, allowing top_k_weights to be either:
    # - (num_tokens, num_top_k) Qwen2MoE style
    # - (num_tokens, num_experts) DeepseekV2 style
    if top_k_weights.shape == (num_tokens, num_top_k):
        sample_weights = top_k_weights
    elif top_k_weights.shape == (num_tokens, num_experts):
        # TODO: routers that output full expert distribution
        # should probably be corrected to output only top_k weights
        sample_weights = top_k_weights[token_idx, expert_ids]
    else:
        raise ValueError(
            f"top_k_weights has an invalid/unsupported shape. It should be either (num_tokens, num_top_k)({num_tokens}, {num_top_k}) "
            f"or (num_tokens, num_experts)({num_tokens}, {num_experts}), but got {top_k_weights.shape}."
        )
    sample_weights = sample_weights.reshape(-1, 1)  # (S, 1)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    inv_perm = torch.argsort(perm)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Select expert weights and biases for selected samples
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    selected_gate_up = self.gate_up_proj
    selected_down = self.down_proj
    selected_gate_up_bias = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    selected_down_bias = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    num_tokens_per_expert = torch.histc(expert_ids_g.float(), bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # --- Up projection per expert (grouped) ---
    gate_up_out = _grouped_linear(
        selected_hidden_states_g, selected_gate_up, selected_gate_up_bias, offsets, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim)

    # Apply gating
    gated_out = self._apply_gate(gate_up_out)  # (S, intermediate_dim)

    # --- Down projection per expert (grouped) ---
    out_per_sample_g = _grouped_linear(
        gated_out, selected_down, selected_down_bias, offsets, is_transposed=self.is_transposed
    )  # (S, hidden_dim)

    # Apply routing weights
    out_per_sample_g = out_per_sample_g * sample_weights_g

    # Restore original order
    out_per_sample = out_per_sample_g[inv_perm]

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # (index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd)
    final_hidden_states = out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts implementations."""

    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
    }


ALL_EXPERTS_FUNCTIONS = ExpertsInterface()


def _default_apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
    """
    Default gating mechanism: splits the gate_up_out into gate and up parts,
    applies the activation function to the gate part, and multiplies it with the up part.
    Args:
        gate_up_out (`torch.Tensor`):
            The output tensor from the gate and up projection of shape (S, 2 * intermediate_dim).
    Returns:
        `torch.Tensor`: The gated output tensor of shape (S, intermediate_dim).
    """
    gate, up = gate_up_out.chunk(2, dim=-1)  # (S, intermediate_dim)
    return self.act_fn(gate) * up  # (S, intermediate_dim)


def use_experts_implementation(
    experts_class: type[torch.nn.Module] | None = None, *, is_transposed: bool = False, has_bias: bool = False
) -> type[torch.nn.Module]:
    """Decorator to modify experts class to support different experts implementations.

    Args:
        experts_class (`type[torch.nn.Module]`, *optional*):
            The experts class to modify. If not provided, returns a decorator that can be applied to the class.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the expert weights are stored in transposed format.
        has_bias (`bool`, *optional*, defaults to `False`):
            Whether the expert layers include bias terms.

    Returns:
        `type[torch.nn.Module]`: The modified experts class.
    """

    def wrapper(experts_class: type[torch.nn.Module]) -> type[torch.nn.Module]:
        original_init = experts_class.__init__
        original_forward = experts_class.forward

        @wraps(original_init)
        def __init__(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            self.config = config
            self.has_bias = has_bias
            self.is_transposed = is_transposed

        @wraps(original_forward)
        def forward(self, *args, **kwargs):
            experts_forward = original_forward

            if self.config._experts_implementation != "eager":
                experts_forward = ALL_EXPERTS_FUNCTIONS[self.config._experts_implementation]

            return experts_forward(self, *args, **kwargs)

        if not hasattr(experts_class, "_apply_gate"):
            experts_class._apply_gate = _default_apply_gate
        experts_class.__init__ = __init__
        experts_class.forward = forward
        return experts_class

    if experts_class is not None:
        return wrapper(experts_class)

    return wrapper
