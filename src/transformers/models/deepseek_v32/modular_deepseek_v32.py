# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import torch
from torch import nn

from ...cache_utils import Cache
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding, GlmMoeDsaMLP, GlmMoeDsaExperts, GlmMoeDsaRMSNorm, GlmMoeDsaDecoderLayer, GlmMoeDsaPreTrainedModel, GlmMoeDsaModel, GlmMoeDsaForCausalLM, GlmMoeDsaMoE
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
# TODO
# Use our rope and convert qkv with rope rotation to benefit from kernels\

@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V2-Lite")
class DeepseekV32Config(GlmMoeDsaConfig, RotaryEmbeddingConfigMixin):
    attribute_map = {
        "num_local_experts": "num_experts"
    } 
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_top_k: int = 2048
    max_seq_len: int = 2048
    mlp_bias: bool = False
    num_experts:int = 256
    head_dim: int = 64

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to a single tensor.

    This is the transformers equivalent of DeepSeek V3.2's `apply_rotary_emb(x, freqs_cis, interleaved)`.
    Instead of using complex-number `freqs_cis`, we use pre-split `(cos, sin)` tensors from RotaryEmbedding.

    Args:
        x (`torch.Tensor`): Input tensor of shape `[..., head_dim]`.
        cos (`torch.Tensor`): Cosine part from RotaryEmbedding, shape `[batch, seq_len, head_dim]`.
        sin (`torch.Tensor`): Sine part from RotaryEmbedding, shape `[batch, seq_len, head_dim]`.
        unsqueeze_dim (`int`): Dimension along which to unsqueeze cos/sin for broadcasting.
            Use `1` when x is `[B, H, S, D]` (BHSD) and `2` when x is `[B, S, H, D]` (BSHD).

    Returns:
        `torch.Tensor`: Tensor with rotary embeddings applied, same shape as input.
    """
    cos = cos[..., : x.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : x.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)

class DeepseekV32MoE(GlmMoeDsaMoE):
    pass


class DeepseekV32MLP(GlmMoeDsaMLP):
    pass


class DeepseekV32RMSNorm(GlmMoeDsaRMSNorm):
    pass


class DeepseekV32RotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    pass


class DeepseekV32Indexer(GlmMoeDsaIndexer):
    pass


class DeepseekV32Attention(GlmMoeDsaAttention):
    pass

class DeepseekV32DecoderLayer(GlmMoeDsaDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(GlmMoeDsaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]


class DeepseekV32Model(GlmMoeDsaModel):
    pass


class DeepseekV32ForCausalLM(GlmMoeDsaForCausalLM):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
