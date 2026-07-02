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

from ..gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
from ..gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from ..gemma4.image_processing_gemma4 import Gemma4ImageProcessor, Gemma4ImageProcessorKwargs
from ..gemma4.modeling_gemma4 import (
    Gemma4AudioAttention,
    Gemma4AudioCausalConv1d,
    Gemma4AudioFeedForward,
    Gemma4AudioLayer,
    Gemma4AudioLightConv1d,
    Gemma4AudioModel,
    Gemma4AudioModelOutput,
    Gemma4AudioRelPositionalEncoding,
    Gemma4AudioSubSampleConvProjection,
    Gemma4AudioSubSampleConvProjectionLayer,
    Gemma4CausalLMOutputWithPast,
    Gemma4ClippableLinear,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4Model,
    Gemma4ModelOutputWithPast,
    Gemma4MultimodalEmbedder,
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextExperts,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextRouter,
    Gemma4TextScaledWordEmbedding,
    Gemma4VisionAttention,
    Gemma4VisionEncoder,
    Gemma4VisionEncoderLayer,
    Gemma4VisionMLP,
    Gemma4VisionModel,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionRotaryEmbedding,
)
from ..gemma4.processing_gemma4 import Gemma4Processor, Gemma4ProcessorKwargs


class TmlAudioConfig(Gemma4AudioConfig):
    pass


class TmlTextConfig(Gemma4TextConfig):
    pass


class TmlVisionConfig(Gemma4VisionConfig):
    pass


class TmlConfig(Gemma4Config):
    pass


class TmlModelOutputWithPast(Gemma4ModelOutputWithPast):
    pass


class TmlCausalLMOutputWithPast(Gemma4CausalLMOutputWithPast):
    pass


class TmlTextModelOutputWithPast(Gemma4TextModelOutputWithPast):
    pass


class TmlAudioModelOutput(Gemma4AudioModelOutput):
    pass


class TmlClippableLinear(Gemma4ClippableLinear):
    pass


class TmlRMSNorm(Gemma4RMSNorm):
    pass


class TmlAudioRelPositionalEncoding(Gemma4AudioRelPositionalEncoding):
    pass


class TmlAudioAttention(Gemma4AudioAttention):
    pass


class TmlAudioSubSampleConvProjectionLayer(Gemma4AudioSubSampleConvProjectionLayer):
    pass


class TmlAudioSubSampleConvProjection(Gemma4AudioSubSampleConvProjection):
    pass


class TmlAudioFeedForward(Gemma4AudioFeedForward):
    pass


class TmlAudioCausalConv1d(Gemma4AudioCausalConv1d):
    pass


class TmlAudioLightConv1d(Gemma4AudioLightConv1d):
    pass


class TmlAudioLayer(Gemma4AudioLayer):
    pass


class TmlVisionPatchEmbedder(Gemma4VisionPatchEmbedder):
    pass


class TmlVisionPooler(Gemma4VisionPooler):
    pass


class TmlVisionMLP(Gemma4VisionMLP):
    pass


class TmlVisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    pass


class TmlVisionAttention(Gemma4VisionAttention):
    pass


class TmlVisionEncoderLayer(Gemma4VisionEncoderLayer):
    pass


class TmlVisionEncoder(Gemma4VisionEncoder):
    pass


class TmlTextMLP(Gemma4TextMLP):
    pass


class TmlTextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


class TmlTextAttention(Gemma4TextAttention):
    pass


class TmlTextExperts(Gemma4TextExperts):
    pass


class TmlTextRouter(Gemma4TextRouter):
    pass


class TmlTextDecoderLayer(Gemma4TextDecoderLayer):
    pass


class TmlTextScaledWordEmbedding(Gemma4TextScaledWordEmbedding):
    pass


class TmlPreTrainedModel(Gemma4PreTrainedModel):
    pass


class TmlTextModel(Gemma4TextModel):
    pass


class TmlForCausalLM(Gemma4ForCausalLM):
    pass


class TmlAudioModel(Gemma4AudioModel):
    pass


class TmlVisionModel(Gemma4VisionModel):
    pass


class TmlMultimodalEmbedder(Gemma4MultimodalEmbedder):
    pass


class TmlModel(Gemma4Model):
    pass


class TmlForConditionalGeneration(Gemma4ForConditionalGeneration):
    pass


class TmlImageProcessorKwargs(Gemma4ImageProcessorKwargs):
    pass


class TmlImageProcessor(Gemma4ImageProcessor):
    pass


class TmlAudioFeatureExtractor(Gemma4AudioFeatureExtractor):
    pass


class TmlProcessorKwargs(Gemma4ProcessorKwargs):
    pass


class TmlProcessor(Gemma4Processor):
    pass


__all__ = [
    "TmlAudioConfig",
    "TmlConfig",
    "TmlTextConfig",
    "TmlVisionConfig",
    "TmlAudioModel",
    "TmlForCausalLM",
    "TmlForConditionalGeneration",
    "TmlModel",
    "TmlPreTrainedModel",
    "TmlTextModel",
    "TmlVisionModel",
    "TmlImageProcessor",
    "TmlAudioFeatureExtractor",
    "TmlProcessor",
]
