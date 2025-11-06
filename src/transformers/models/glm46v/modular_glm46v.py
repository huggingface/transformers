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

from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vTextConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vForConditionalGeneration,
    Glm4VisionMlp,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vRMSNorm,
    Glm4vTextAttention,
    Glm4vTextDecoderLayer,
    Glm4vTextMLP,
    Glm4vTextModel,
    Glm4vTextRotaryEmbedding,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vVisionEmbeddings,
    Glm4vVisionModel,
    Glm4vVisionPatchEmbed,
    Glm4vVisionPatchMerger,
    Glm4vVisionRotaryEmbedding,
)
from ..glm4v.processing_glm4v import Glm4vProcessor


class Glm46VVisionConfig(Glm4vVisionConfig):
    pass


class Glm46VTextConfig(Glm4vTextConfig):
    pass


class Glm46VConfig(Glm4vConfig):
    pass


class Glm46VRMSNorm(Glm4vRMSNorm):
    pass


class Glm4VisionMlp(Glm4VisionMlp):
    pass


class Glm46VVisionPatchEmbed(Glm4vVisionPatchEmbed):
    pass


class Glm46VVisionRotaryEmbedding(Glm4vVisionRotaryEmbedding):
    pass


class Glm46VVisionPatchMerger(Glm4vVisionPatchMerger):
    pass


class Glm46VVisionEmbeddings(Glm4vVisionEmbeddings):
    pass


class Glm46VVisionAttention(Glm4vVisionAttention):
    pass


class Glm46VVisionBlock(Glm4vVisionBlock):
    pass


class Glm46VTextRotaryEmbedding(Glm4vTextRotaryEmbedding):
    pass


class Glm46VTextAttention(Glm4vTextAttention):
    pass


class Glm46VTextMLP(Glm4vTextMLP):
    pass


class Glm46VTextDecoderLayer(Glm4vTextDecoderLayer):
    pass


class Glm46VModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class Glm46VPreTrainedModel(Glm4vPreTrainedModel):
    pass


class Glm46VVisionModel(Glm4vVisionModel):
    pass


class Glm46VTextModel(Glm4vTextModel):
    pass


class Glm46VModel(Glm4vModel):
    pass


class Glm46VCausalLMOutputWithPast(Glm4vCausalLMOutputWithPast):
    pass


class Glm46VForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


class Glm46vProcessor(Glm4vProcessor):
    def replace_frame_token_id(self, timestamp_sec):
        return f"<|begin_of_image|>{self.image_token}<|end_of_image|>{round(timestamp_sec)} seconds"


__all__ = [
    "Glm46VConfig",
    "Glm46VTextConfig",
    "Glm46VForConditionalGeneration",
    "Glm46VModel",
    "Glm46VPreTrainedModel",
    "Glm46VTextModel",
]
