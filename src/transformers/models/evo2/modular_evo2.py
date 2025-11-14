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

from ..mistral.configuration_mistral import MistralConfig
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralMLP,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)


class Evo2Config(MistralConfig):
    pass


class Evo2MLP(MistralMLP):
    pass


class Evo2Attention(MistralAttention):
    pass


class Evo2RMSNorm(MistralRMSNorm):
    pass


class Evo2DecoderLayer(MistralDecoderLayer):
    pass


class Evo2PreTrainedModel(MistralPreTrainedModel):
    pass


class Evo2RotaryEmbedding(MistralRotaryEmbedding):
    pass


class Evo2Model(MistralModel):
    pass


class Evo2ForCausalLM(MistralForCausalLM):
    pass


class Evo2ForTokenClassification(MistralForTokenClassification):
    pass


class Evo2ForSequenceClassification(MistralForSequenceClassification):
    pass


class Evo2ForQuestionAnswering(MistralForQuestionAnswering):
    pass


__all__ = [
    "Evo2Config",
    "Evo2ForCausalLM",
    "Evo2ForQuestionAnswering",
    "Evo2Model",
    "Evo2PreTrainedModel",
    "Evo2ForSequenceClassification",
    "Evo2ForTokenClassification",
]
