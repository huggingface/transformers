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
from ..llama.tokenization_llama import LlamaTokenizer
from ..llama.tokenization_llama_fast import LlamaTokenizerFast


class PldrllmConfig(LlamaConfig):
    pass


class PldrllmRMSNorm(LlamaRMSNorm):
    pass


class PldrllmRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class PldrllmMLP(LlamaMLP):
    pass


class PldrllmAttention(LlamaAttention):
    pass


class PldrllmDecoderLayer(LlamaDecoderLayer):
    pass


class PldrllmPreTrainedModel(LlamaPreTrainedModel):
    pass


class PldrllmModel(LlamaModel):
    pass


class PldrllmForCausalLM(LlamaForCausalLM):
    pass


class PldrllmForSequenceClassification(LlamaForSequenceClassification):
    pass


class PldrllmForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class PldrllmForTokenClassification(LlamaForTokenClassification):
    pass


class PldrllmTokenizer(LlamaTokenizer):
    pass


class PldrllmTokenizerFast(LlamaTokenizerFast):
    pass


__all__ = [
    "PldrllmConfig",
    "PldrllmForCausalLM",
    "PldrllmModel",
    "PldrllmPreTrainedModel",
    "PldrllmForSequenceClassification",
    "PldrllmForQuestionAnswering",
    "PldrllmForTokenClassification",
    "PldrllmTokenizer",
    "PldrllmTokenizerFast",
]
