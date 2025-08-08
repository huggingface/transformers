# coding=utf-8
# Copyright 2024 weak-kajuma and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Llama implementations in this library and Microsoft's
# Differential Transformer implementations.

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
from typing import Optional

import torch
from torch import nn

from ...cache_utils import Cache
from ...utils import logging
from ...utils.deprecation import deprecate_kwarg
from ..gemma.modeling_gemma import GemmaForCausalLM
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ..mistral.modeling_mistral import MistralMLP
from .configuration_diffllama import DiffLlamaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "kajuma/DiffLlama-0.3B-handcut"
_CONFIG_FOR_DOC = "DiffLlamaConfig"


class DiffLlamaMLP(MistralMLP):
    pass


def lambda_init_fn(layer_idx):
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class DiffLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DiffLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiffLlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_q2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.groupnorm = nn.RMSNorm(2 * self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    @deprecate_kwarg("position_embeddings", version="4.60.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, target_len, _ = hidden_states.size()
        q_len = target_len

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            logger.warning_once(
                "The attention layers in this model are transitioning to computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens). Suing pre-computed"
                "`position_embeddings` (Tuple of tensors, containing cos and sin) is deprecated and will be "
                "removed in v4.60.0. Make sure to pass `position_ids` instead."
            )
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        value_states = torch.cat(torch.chunk(value_states, 2, dim=1), dim=-1)
        value_states = value_states.repeat(1, 2, 1, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, dtype=torch.float32)).to(
            query_states.dtype
        )
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, dtype=torch.float32)).to(
            query_states.dtype
        )
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output1, attn_output2 = torch.chunk(attn_output, 2, dim=1)

        attn_output = attn_output1 - lambda_full * attn_output2
        attn_output = (1 - self.lambda_init) * self.groupnorm(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffLlamaPreTrainedModel(LlamaPreTrainedModel):
    _supports_flex_attn = False
    _supports_attention_backend = False

    def _init_weights(self, module):
        LlamaPreTrainedModel._init_weights(module)
        if isinstance(module, DiffLlamaAttention):
            module.lambda_q1.data.normal_(0, self.config.lambda_std_dev)
            module.lambda_k1.data.normal_(0, self.config.lambda_std_dev)
            module.lambda_q2.data.normal_(0, self.config.lambda_std_dev)
            module.lambda_k2.data.normal_(0, self.config.lambda_std_dev)


class DiffLlamaForCausalLM(GemmaForCausalLM):
    pass


class DiffLlamaForSequenceClassification(LlamaForSequenceClassification):
    pass


class DiffLlamaForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class DiffLlamaForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "DiffLlamaPreTrainedModel",
    "DiffLlamaModel",  # noqa: F822
    "DiffLlamaForCausalLM",
    "DiffLlamaForSequenceClassification",
    "DiffLlamaForQuestionAnswering",
    "DiffLlamaForTokenClassification",
]
