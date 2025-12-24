# coding=utf-8
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

from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoeNaiveMoe,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
    Glm4MoeTopkRouter,
)
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class Glm4MoeLiteConfig(DeepseekV3Config):
    pass


class Glm4MoeLiteRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class Glm4MoeLiteAttention(DeepseekV3Attention):
    pass


class Glm4MoeLiteMLP(Glm4MoeMLP):
    pass


class Glm4MoeLiteTopkRouter(Glm4MoeTopkRouter):
    pass


class Glm4MoeLiteRMSNorm(Glm4MoeRMSNorm):
    pass


class Glm4MoeLiteNaiveMoe(Glm4MoeNaiveMoe):
    pass


class Glm4MoeLiteMoE(Glm4MoeMoE):
    pass


class Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer):
    pass


class Glm4MoeLitePreTrainedModel(Glm4MoePreTrainedModel):
    pass


class Glm4MoeLiteModel(Glm4MoeModel):
    pass


class Glm4MoeLiteForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "Glm4MoeLiteConfig",
    "Glm4MoeLitePreTrainedModel",
    "Glm4MoeLiteModel",
    "Glm4MoeLiteForCausalLM",
]
