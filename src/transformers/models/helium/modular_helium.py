# Copyright 2024 The Kyutai and HuggingFace Inc. teams. All rights reserved.
#
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
import math
from collections.abc import Callable

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.generic import no_inherit_decorator
from ..gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    eager_attention_forward,
)
from ..granite.modeling_granite import GraniteAttention
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaMLP, LlamaModel, LlamaPreTrainedModel, LlamaRotaryEmbedding
from .configuration_helium import HeliumConfig


logger = logging.get_logger(__name__)


class HeliumRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class HeliumRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class HeliumMLP(LlamaMLP):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


@no_inherit_decorator
class HeliumAttention(GraniteAttention):
    def __init__(self, config: HeliumConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scaling = 1 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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


class HeliumDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: HeliumConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)

        self.mlp = HeliumMLP(config)
        self.input_layernorm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = HeliumAttention(config=config, layer_idx=layer_idx)


class HeliumPreTrainedModel(LlamaPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": HeliumDecoderLayer,
        "attentions": HeliumAttention,
    }


class HeliumModel(HeliumPreTrainedModel, LlamaModel):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HeliumDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HeliumRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class HeliumForCausalLM(GemmaForCausalLM):
    def __init__(self, config: HeliumConfig):
        super().__init__(config)
        self.model = HeliumModel(config)


class HeliumForSequenceClassification(GemmaForSequenceClassification):
    pass


class HeliumForTokenClassification(GemmaForTokenClassification):
    pass


__all__ = [
    "HeliumPreTrainedModel",
    "HeliumModel",
    "HeliumForCausalLM",
    "HeliumForSequenceClassification",
    "HeliumForTokenClassification",
]
