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

from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2ForSequenceClassification,
    Gemma2ForTokenClassification,
    Gemma2MLP,
    Gemma2Model,
    Gemma2PreTrainedModel,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    Gemma2TextScaledWordEmbedding,
)


class OnyxConfig(Gemma2Config):
    pass


class OnyxRMSNorm(Gemma2RMSNorm):
    pass


class OnyxMLP(Gemma2MLP):
    pass


class OnyxRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class OnyxAttention(Gemma2Attention):
    pass


class OnyxDecoderLayer(Gemma2DecoderLayer):
    pass


class OnyxTextScaledWordEmbedding(Gemma2TextScaledWordEmbedding):
    pass


class OnyxPreTrainedModel(Gemma2PreTrainedModel):
    pass


class OnyxModel(Gemma2Model):
    pass


class OnyxForCausalLM(Gemma2ForCausalLM):
    pass


class OnyxForSequenceClassification(Gemma2ForSequenceClassification):
    pass


class OnyxForTokenClassification(Gemma2ForTokenClassification):
    pass


__all__ = [
    "OnyxConfig",
    "OnyxForCausalLM",
    "OnyxModel",
    "OnyxPreTrainedModel",
    "OnyxForSequenceClassification",
    "OnyxForTokenClassification",
]
