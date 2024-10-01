# coding=utf-8
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from ..gemma.configuration_gemma import GemmaConfig
from ..gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
)
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaModel,
    repeat_kv,
)
from ..phi3.modeling_phi3 import (
    Phi3MLP,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


class GlmConfig(GemmaConfig):

    model_type = "glm"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        resid_pdrop=0.0,
        attention_dropout=0.0,
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=0.00000015625,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        pad_token_id=151329,
        eos_token_id=[151329, 151336, 151338],
        bos_token_id=None,
        attention_bias=True,
        linear_bias=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        del self.hidden_activation


class GlmRMSNorm(Phi3RMSNorm):
    pass


class GlmRotaryEmbedding(Phi3RotaryEmbedding):
    pass


class GlmMLP(Phi3MLP):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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

    # Keep half for later concatenation
    q, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    # Apply rotary embeddings on the first half
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class GlmAttention(LlamaAttention):
    def __init__(self, config: GlmConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # Not used in the attention, only for BC
        self.rotary_emb = GlmRotaryEmbedding(
            dim=config.head_dim // 2, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )


class GlmFlashAttention2(GlmAttention, LlamaFlashAttention2):
    pass


class GlmSdpaAttention(GlmAttention, LlamaSdpaAttention):
    pass

GLM_ATTENTION_CLASSES = {
    "eager": GlmAttention,
    "flash_attention_2": GlmFlashAttention2,
    "sdpa": GlmSdpaAttention,
}


class GlmDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GlmConfig, layer_idx: int):
        super().__init__()

        self.mlp = GlmMLP(config)
        self.input_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GlmModel(LlamaModel):
    def __init__(self, config: GlmConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GlmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = GlmRotaryEmbedding(
            dim=config.head_dim // 2, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GlmForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()


class GlmForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self.post_init()
