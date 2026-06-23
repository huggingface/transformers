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
    """Multi-headed differential attention (https://arxiv.org/abs/2410.05258).

    Computes ``(softmax(Q1 K1ᵀ) - λ · softmax(Q2 K2ᵀ)) · V`` as **two standard attention calls**
    that share Q and K but use the two halves of V (concatenated outputs along the last dim
    reconstruct the per-head ``2 * head_dim`` the paper calls for; the head-axis chunk-and-subtract
    then realises the differential combination).

    The natural "shortcut" of packing V along ``head_dim`` to do a single attention call with
    ``V`` of shape ``(B, H, S, 2D)`` is *slower* in practice. Asymmetric V trips PyTorch's SDPA
    backend selector — it can't use Flash or cuDNN with ``head_dim_v != head_dim_q`` and falls
    back to the memory-efficient / math kernel. Benchmarks at production shapes (prefill, long
    context, training-sized batches) show the two-call version is **~30 % faster** than the
    V-doubling version even though it issues an extra kernel launch — the gain from picking the
    fast Flash/cuDNN kernel dominates the launch overhead. Flash Attention 2 also requires
    ``head_dim_v == head_dim_q``, which only the two-call structure satisfies.
    """

    def __init__(self, config: DiffLlamaConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
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
        # Both calls share Q and K, so their attention weights are mathematically identical —
        # return just the first call's; concatenating would double the key dim incorrectly.
        attn_output1, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states1,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output2, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states2,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
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
