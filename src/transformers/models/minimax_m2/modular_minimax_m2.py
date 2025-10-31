# coding=utf-8
# Copyright 2025 the MiniMax AI Team and HuggingFace Team. All rights reserved.
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

from typing import Callable, Optional, Unpack
import torch
from torch import nn

from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...cache_utils import Cache


from ..mixtral.configuration_mixtral import MixtralConfig
from ..mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralExperts,
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralMLP,
    MixtralModel,
    MixtralPreTrainedModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)



class MiniMaxM2Config(MixtralConfig):
    model_type = "minimax_m2"


class MiniMaxM2MLP(MixtralMLP):
    pass


class MiniMaxM2Experts(MixtralExperts):
    pass


class MiniMaxM2SparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = MiniMaxM2Experts(config)
        self.e_score_correction_bias = nn.Parameter(torch.zeros((config.num_local_experts), dtype=torch.float32))

    def route_tokens_to_experts(self, router_logits):
        routing_weights = torch.nn.functional.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return top_k_index, top_k_weights.to(router_logits.dtype)


class MiniMaxM2RMSNorm(MixtralRMSNorm):
    pass


class MiniMaxM2RotaryEmbedding(Glm4MoeRotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[MiniMaxM2Config] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "rotary_dim", None)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class MiniMaxM2Attention(Glm4MoeAttention):
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_attention_heads, eps=config.rms_norm_eps)
            self.k_norm = MiniMaxM2RMSNorm(self.head_dim * config.num_key_value_heads, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        key_states = key_states.view(hidden_shape)
        query_states = query_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxM2DecoderLayer(MixtralDecoderLayer):
    pass


class MiniMaxM2PreTrainedModel(MixtralPreTrainedModel):
    pass


class MiniMaxM2Model(MixtralModel):
    pass


class MiniMaxM2ForCausalLM(MixtralForCausalLM):
    pass


class MiniMaxM2ForSequenceClassification(MixtralForSequenceClassification):
    pass


class MiniMaxM2ForTokenClassification(MixtralForTokenClassification):
    pass


class MiniMaxM2ForQuestionAnswering(MixtralForQuestionAnswering):
    pass


__all__ = [
    "MiniMaxM2Config",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM2ForQuestionAnswering",
    "MiniMaxM2Model",
    "MiniMaxM2PreTrainedModel",
    "MiniMaxM2ForSequenceClassification",
    "MiniMaxM2ForTokenClassification",
]
