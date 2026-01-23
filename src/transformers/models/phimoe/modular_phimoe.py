# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phimoe model."""

from collections.abc import Callable

import torch
from torch import nn

from ...modeling_layers import (
    GenericForSequenceClassification,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...utils.generic import OutputRecorder, maybe_autocast
from ..llama.modeling_llama import LlamaAttention
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRotaryEmbedding,
)
from .configuration_phimoe import PhimoeConfig


class PhimoeRotaryEmbedding(MixtralRotaryEmbedding):
    def __init__(self, config: PhimoeConfig, device=None):
        nn.Module.__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        self.rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    def forward(self, x, position_ids=None, layer_type=None):
        if layer_type is not None:
            raise ValueError(
                f"{self.__class__.__name__} does not support layer types, but got `layer_type={layer_type}`"
            )

        mscale = None
        seq_len = torch.max(position_ids) + 1
        if self.config.rope_parameters["rope_type"] != "default" and seq_len:
            mscale = (
                self.long_mscale
                if seq_len > self.config.rope_parameters["original_max_position_embeddings"]
                else self.short_mscale
            )
        inv_freq, attention_scaling = self.rope_init_fn(self.config, x.device, seq_len)
        mscale = attention_scaling if mscale is None else mscale
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * mscale
            sin = emb.sin() * mscale
        return cos.to(x.dtype), sin.to(x.dtype)


class PhimoeAttention(LlamaAttention):
    pass


class PhimoeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        """
        Forward pass for the custom autograd function.

        Args:
            ctx: Context object to save information for backward computation.
            scores (torch.Tensor): Input scores tensor.
            multiplier (torch.Tensor): Multiplier tensor.
            selected_experts (torch.Tensor): Tensor of selected experts.
            masked_gates (torch.Tensor): Masked gates tensor.
            mask_for_one (torch.Tensor): Mask for one tensor.

        Returns:
            torch.Tensor: Result of the forward pass.
        """
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        """
        Backward pass for the custom autograd function.

        Args:
            ctx: Context object with saved tensors from the forward pass.
            grad_at_output (torch.Tensor): Gradient at the output.

        Returns:
            tuple[torch.Tensor, None, None, None, None]: Gradients for the inputs.
        """
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expanded = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expanded.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expanded,
            None,
            None,
            None,
            None,
        )


def sparsemixer(scores, jitter_eps, training, top_k=2):
    """
    Sparse mixer function to select top-k experts and compute multipliers.
    Based on the paper: https://huggingface.co/papers/2409.12136
    We first replace the TopK(·) function as random sampling of discrete variables
    in model training. Then, following Liu et al. (2023a) and Liu et al. (2023b), we apply Heun's
    third order method to approximate the expert routing gradient and construct a modified
    back-propagation to give a mathematically sound gradient estimation for expert routing.

    Args:
        scores (torch.Tensor): Input scores tensor.
        jitter_eps (float): Jitter epsilon for numerical stability.
        training (bool): Flag indicating if the model is in training mode.
        top_k (int): Number of top experts to select.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Multiplier and selected experts tensors.
    """
    with torch.no_grad():
        # Compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # Apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float("-inf"))
    if training:
        selected_experts = (
            (
                masked_gates
                - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )  # Gumbel sampling, more robust than the multinomial method
    else:
        selected_experts = max_ind

    # Compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    if training:
        # Compute midpoint mask
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75,  # Heun's third-order method
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = PhimoeMultiplier.apply(
            scores,
            multiplier_o,
            selected_experts,
            masked_gates,
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # Masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float("-inf"),
    )
    with torch.no_grad():
        # Compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # Apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))
    if training:
        selected_experts_top2 = (
            (
                masked_gates_top2
                - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            .max(dim=-1)[1]
            .unsqueeze(-1)
        )  # Gumbel sampling, more robust than the multinomial method
    else:
        selected_experts_top2 = max_ind
    # Compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    if training:
        # Compute midpoint mask
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75,  # Heun's third-order method
        )
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = PhimoeMultiplier.apply(
            scores,
            multiplier_top2_o,
            selected_experts_top2,
            masked_gates_top2,
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


class PhimoeExperts(MixtralExperts):
    pass


class PhimoeTopKRouter(nn.Linear):
    def __init__(self, config: PhimoeConfig):
        super().__init__(config.hidden_size, config.num_local_experts, bias=False)
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )
        router_logits = super().forward(hidden_states)
        routing_weights, selected_experts = sparsemixer(
            router_logits, jitter_eps=self.router_jitter_noise, training=self.training, top_k=self.top_k
        )
        return routing_weights, selected_experts


class PhimoeSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router = PhimoeTopKRouter(config)
        self.experts = PhimoeExperts(config)
        self.input_jitter_noise = config.input_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise
            )

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        routing_weights, selected_experts = self.router(hidden_states)
        final_hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class PhimoeDecoderLayer(MixtralDecoderLayer):
    pass


class PhimoePreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(PhimoeTopKRouter, layer_name="mlp.router", index=0),
        "hidden_states": PhimoeDecoderLayer,
        "attentions": PhimoeAttention,
    }


class PhimoeModel(MixtralModel):
    def __init__(self, config: PhimoeConfig):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)


class PhimoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)

    # Copied from transformers.models.phi3.modeling_phi3.Phi3ForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- this model may need to switch between short and long rope, invalidating the cache in the
        # process

        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if (
            past_key_values
            and hasattr(self.config, "original_max_position_embeddings")
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1
        ):
            past_length = cache_position[0]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        return model_inputs


class PhimoeForSequenceClassification(GenericForSequenceClassification, PhimoePreTrainedModel): ...


__all__ = [
    "PhimoePreTrainedModel",
    "PhimoeModel",
    "PhimoeForCausalLM",
    "PhimoeForSequenceClassification",
]
