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
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding, GlmMoeDsaMLP, GlmMoeDsaExperts, GlmMoeDsaRMSNorm, GlmMoeDsaDecoderLayer, GlmMoeDsaPreTrainedModel, GlmMoeDsaModel, GlmMoeDsaForCausalLM
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
# TODO
# Use our rope and convert qkv with rope rotation to benefit from kernels\

@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V2-Lite")
class DeepseekV32Config(GlmMoeDsaConfig, RotaryEmbeddingConfigMixin):
    attribute_map = {
        "num_experts": "num_experts_per_tok"
    } 
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_top_k: int = 2048
    max_seq_len: int = 2048
    mlp_bias: bool = False



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
    pass


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
