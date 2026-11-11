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

from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class NanbeigeConfig(LlamaConfig):
    pass


class NanbeigeRMSNorm(LlamaRMSNorm):
    pass


class NanbeigeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class NanbeigeMLP(LlamaMLP):
    pass


class NanbeigeAttention(LlamaAttention):
    pass


class NanbeigeDecoderLayer(LlamaDecoderLayer):
    pass


class NanbeigePreTrainedModel(LlamaPreTrainedModel):
    pass


class NanbeigeModel(LlamaModel):
    pass


class NanbeigeForCausalLM(LlamaForCausalLM):
    pass


class NanbeigeForSequenceClassification(LlamaForSequenceClassification):
    pass


class NanbeigeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class NanbeigeForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "NanbeigeModel",
    "NanbeigePreTrainedModel",
    "NanbeigeForSequenceClassification",
    "NanbeigeForQuestionAnswering",
    "NanbeigeForTokenClassification",
]
