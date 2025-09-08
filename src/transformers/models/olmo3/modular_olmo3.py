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

from ..olmo2.configuration_olmo2 import Olmo2Config
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2MLP,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
    Olmo2RotaryEmbedding,
)


class Olmo3Config(Olmo2Config):
    pass


class Olmo3RMSNorm(Olmo2RMSNorm):
    pass


class Olmo3Attention(Olmo2Attention):
    pass


class Olmo3MLP(Olmo2MLP):
    pass


class Olmo3DecoderLayer(Olmo2DecoderLayer):
    pass


class Olmo3RotaryEmbedding(Olmo2RotaryEmbedding):
    pass


class Olmo3PreTrainedModel(Olmo2PreTrainedModel):
    pass


class Olmo3Model(Olmo2Model):
    pass


class Olmo3ForCausalLM(Olmo2ForCausalLM):
    pass


__all__ = [
    "Olmo3Config",
    "Olmo3ForCausalLM",
    "Olmo3Model",
    "Olmo3PreTrainedModel",
]
