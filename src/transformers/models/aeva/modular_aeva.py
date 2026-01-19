# Copyright 2026 the HuggingFace Team. All rights reserved.
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
    Glm4MoeAttention,
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


class AevaConfig(Glm4MoeConfig):
    pass


class AevaRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class AevaAttention(Glm4MoeAttention):
    pass


class AevaMLP(Glm4MoeMLP):
    pass


class AevaTopkRouter(Glm4MoeTopkRouter):
    pass


class AevaRMSNorm(Glm4MoeRMSNorm):
    pass


class AevaNaiveMoe(Glm4MoeNaiveMoe):
    pass


class AevaMoE(Glm4MoeMoE):
    pass


class AevaDecoderLayer(Glm4MoeDecoderLayer):
    pass


class AevaPreTrainedModel(Glm4MoePreTrainedModel):
    pass


class AevaModel(Glm4MoeModel):
    pass


class AevaForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "AevaConfig",
    "AevaPreTrainedModel",
    "AevaModel",
    "AevaForCausalLM",
]
