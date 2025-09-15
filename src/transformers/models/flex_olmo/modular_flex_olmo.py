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

from ..olmoe.configuration_olmoe import OlmoeConfig
from ..olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeDecoderLayer,
    OlmoeFlashAttention2,
    OlmoeForCausalLM,
    OlmoeMLP,
    OlmoeModel,
    OlmoePreTrainedModel,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
    OlmoeSdpaAttention,
    OlmoeSparseMoeBlock,
)


class FlexOlmoConfig(OlmoeConfig):
    pass


class FlexOlmoRMSNorm(OlmoeRMSNorm):
    pass


class FlexOlmoRotaryEmbedding(OlmoeRotaryEmbedding):
    pass


class FlexOlmoMLP(OlmoeMLP):
    pass


class FlexOlmoAttention(OlmoeAttention):
    pass


class FlexOlmoFlashAttention2(OlmoeFlashAttention2):
    pass


class FlexOlmoSdpaAttention(OlmoeSdpaAttention):
    pass


class FlexOlmoSparseMoeBlock(OlmoeSparseMoeBlock):
    pass


class FlexOlmoDecoderLayer(OlmoeDecoderLayer):
    pass


class FlexOlmoPreTrainedModel(OlmoePreTrainedModel):
    pass


class FlexOlmoModel(OlmoeModel):
    pass


class FlexOlmoForCausalLM(OlmoeForCausalLM):
    pass


__all__ = [
    "FlexOlmoConfig",
    "FlexOlmoForCausalLM",
    "FlexOlmoModel",
    "FlexOlmoPreTrainedModel",
]
