# Copyright 2026 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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

from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3Model,
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
)
from .configuration_sarvam_mla import SarvamMLAConfig


class SarvamMLARMSNorm(DeepseekV3RMSNorm):
    pass


class SarvamMLARotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class SarvamMLAMLP(DeepseekV3MLP):
    pass


class SarvamMLATopkRouter(DeepseekV3TopkRouter):
    pass


class SarvamMLANaiveMoe(DeepseekV3NaiveMoe):
    pass


class SarvamMLAMoE(DeepseekV3MoE):
    pass


class SarvamMLAAttention(DeepseekV3Attention):
    pass


class SarvamMLADecoderLayer(DeepseekV3DecoderLayer):
    pass


class SarvamMLAPreTrainedModel(DeepseekV3PreTrainedModel):
    pass


class SarvamMLAModel(DeepseekV3Model):
    pass


class SarvamMLAForCausalLM(DeepseekV3ForCausalLM):
    pass


class SarvamMLAForSequenceClassification(GenericForSequenceClassification, SarvamMLAPreTrainedModel):
    pass


class SarvamMLAForTokenClassification(GenericForTokenClassification, SarvamMLAPreTrainedModel):
    pass


__all__ = [
    "SarvamMLAPreTrainedModel",
    "SarvamMLAModel",
    "SarvamMLAForCausalLM",
    "SarvamMLAForSequenceClassification",
    "SarvamMLAForTokenClassification",
]
