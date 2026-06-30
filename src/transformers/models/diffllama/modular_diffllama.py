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
from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import logging
from ..gemma.modeling_gemma import GemmaForCausalLM
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
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
    """Multi-headed differential attention (https://huggingface.co/papers/2410.05258).

    Computes ``(softmax(Q1 K1ᵀ) - λ · softmax(Q2 K2ᵀ)) · V`` as two standard attention calls
    sharing Q and K over the two halves of V. The two-call structure is ~30% faster than the
    V-doubling shortcut at production shapes, since asymmetric V (``head_dim_v != head_dim_q``)
    forces SDPA off Flash/cuDNN onto the memory-efficient/math kernel; Flash Attention 2 also
    requires ``head_dim_v == head_dim_q``.
    """

    def __init__(self, config: DiffLlamaConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        # The Differential Transformer paper (https://huggingface.co/papers/2410.05258) does not
        # specify how attention dropout should be applied to the differential combination, and our
        # two-call implementation has no single softmax to share a dropout mask across. Refuse
        # rather than pick semantics the paper doesn't define.
        if config.attention_dropout > 0.0:
            raise ValueError(
                "DiffLlama does not support `attention_dropout > 0`: the differential attention "
                "mechanism has no paper-defined dropout semantics."
            )
        # ``torch.chunk(value_states, 2, dim=1).repeat(1, 2, ...)`` below requires the KV-head axis
        # to split evenly into the two halves of the differential combination.
        if config.num_key_value_heads is None or config.num_key_value_heads % 2 != 0:
            raise ValueError(
                "DiffLlama requires `num_key_value_heads` to be even (and at least 2): the two-call "
                f"differential attention splits the value tensor along KV heads, got {config.num_key_value_heads}."
            )
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k1 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_q2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.lambda_k2 = nn.Parameter(torch.normal(0, config.lambda_std_dev, size=(self.head_dim,)))
        self.groupnorm = nn.RMSNorm(2 * self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
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

        # Split V into two halves and broadcast each back to ``num_kv_heads`` heads (the dispatch's
        # ``repeat_kv`` will then expand them to ``num_heads`` like K).
        value_states1, value_states2 = (v.repeat(1, 2, 1, 1) for v in torch.chunk(value_states, 2, dim=1))

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        # The first call's weights are returned; the second's are mathematically identical
        # (shared Q/K). ``config.attention_dropout > 0`` is rejected in ``__init__`` because the
        # two calls cannot share a single softmax-dropout mask the way V-doubling did.
        attn_output1, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states1,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output2, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states2,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

        # Chunk along the head axis and apply the learned lambda — realises the differential
        # combination ``(softmax_1 - λ · softmax_2) · V`` head-pair by head-pair.
        attn_output1, attn_output2 = torch.chunk(attn_output, 2, dim=2)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1, dtype=torch.float32)).to(
            query_states.dtype
        )
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1, dtype=torch.float32)).to(
            query_states.dtype
        )
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_output = attn_output1 - lambda_full * attn_output2
        attn_output = (1 - self.lambda_init) * self.groupnorm(attn_output)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiffLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DiffLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = DiffLlamaAttention(config=config, layer_idx=layer_idx)


class DiffLlamaPreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DiffLlamaAttention):
            init.normal_(module.lambda_q1, 0, self.config.lambda_std_dev)
            init.normal_(module.lambda_k1, 0, self.config.lambda_std_dev)
            init.normal_(module.lambda_q2, 0, self.config.lambda_std_dev)
            init.normal_(module.lambda_k2, 0, self.config.lambda_std_dev)


class DiffLlamaModel(LlamaModel):
    pass


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
    "DiffLlamaModel",
    "DiffLlamaForCausalLM",
    "DiffLlamaForSequenceClassification",
    "DiffLlamaForQuestionAnswering",
    "DiffLlamaForTokenClassification",
]
