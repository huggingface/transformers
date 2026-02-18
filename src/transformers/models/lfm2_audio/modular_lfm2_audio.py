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

from ..moshi.configuration_moshi import MoshiConfig, MoshiDepthConfig
from ..moshi.modeling_moshi import (
    MoshiAttention,
    MoshiCausalLMOutputWithPast,
    MoshiConditionalGenerationGenerateOutput,
    MoshiConditionalGenerationOutputWithPast,
    MoshiDecoderLayer,
    MoshiDepthDecoder,
    MoshiFlashAttention2,
    MoshiFlexibleLinear,
    MoshiForCausalLM,
    MoshiForConditionalGeneration,
    MoshiGatingMLP,
    MoshiLinear,
    MoshiModel,
    MoshiPreTrainedModel,
    MoshiRMSNorm,
    MoshiRotaryEmbedding,
    MoshiSdpaAttention,
    MoshiUnconditionalInput,
)
from ..audioflamingo3.processing_audioflamingo3 import AudioFlamingo3Processor
from ...processing_utils import ProcessorMixin

class Lfm2AudioDepthConfig(MoshiDepthConfig):
    pass


class Lfm2AudioConfig(MoshiConfig):
    pass


class Lfm2AudioProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "ParakeetFeatureExtractor"
    tokenizer_class = "AutoTokenizer"


class Lfm2AudioConditionalGenerationGenerateOutput(MoshiConditionalGenerationGenerateOutput):
    pass


class Lfm2AudioCausalLMOutputWithPast(MoshiCausalLMOutputWithPast):
    pass


class Lfm2AudioConditionalGenerationOutputWithPast(MoshiConditionalGenerationOutputWithPast):
    pass


class Lfm2AudioUnconditionalInput(MoshiUnconditionalInput):
    pass


class Lfm2AudioRMSNorm(MoshiRMSNorm):
    pass


class Lfm2AudioFlexibleLinear(MoshiFlexibleLinear):
    pass


class Lfm2AudioLinear(MoshiLinear):
    pass


class Lfm2AudioRotaryEmbedding(MoshiRotaryEmbedding):
    pass


class Lfm2AudioGatingMLP(MoshiGatingMLP):
    pass


class Lfm2AudioAttention(MoshiAttention):
    pass


class Lfm2AudioFlashAttention2(MoshiFlashAttention2):
    pass


class Lfm2AudioSdpaAttention(MoshiSdpaAttention):
    pass


class Lfm2AudioDecoderLayer(MoshiDecoderLayer):
    pass


class Lfm2AudioPreTrainedModel(MoshiPreTrainedModel):
    pass


class Lfm2AudioDepthDecoder(MoshiDepthDecoder):
    pass


class Lfm2AudioModel(MoshiModel):
    pass


class Lfm2AudioForCausalLM(MoshiForCausalLM):
    pass


class Lfm2AudioForConditionalGeneration(MoshiForConditionalGeneration):
    pass


__all__ = [
    "Lfm2AudioConfig",
    "Lfm2AudioDepthConfig",
    "Lfm2AudioProcessor",
    "Lfm2AudioForCausalLM",
    "Lfm2AudioForConditionalGeneration",
    "Lfm2AudioModel",
    "Lfm2AudioPreTrainedModel",
]
