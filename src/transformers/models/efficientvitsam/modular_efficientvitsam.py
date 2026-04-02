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

from ..sam.configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
from ..sam.image_processing_pil_sam import SamImageProcessorPil
from ..sam.image_processing_sam import SamImageProcessor, SamImageProcessorKwargs
from ..sam.modeling_sam import (
    SamAttention,
    SamFeedForward,
    SamImageSegmentationOutput,
    SamLayerNorm,
    SamMaskDecoder,
    SamMaskEmbedding,
    SamMLPBlock,
    SamModel,
    SamPatchEmbeddings,
    SamPositionalEmbedding,
    SamPreTrainedModel,
    SamPromptEncoder,
    SamTwoWayAttentionBlock,
    SamTwoWayTransformer,
    SamVisionAttention,
    SamVisionEncoder,
    SamVisionEncoderOutput,
    SamVisionLayer,
    SamVisionModel,
    SamVisionNeck,
    SamVisionSdpaAttention,
)
from ..sam.processing_sam import SamImagesKwargs, SamProcessor, SamProcessorKwargs


class EfficientvitsamPromptEncoderConfig(SamPromptEncoderConfig):
    pass


class EfficientvitsamMaskDecoderConfig(SamMaskDecoderConfig):
    pass


class EfficientvitsamVisionConfig(SamVisionConfig):
    pass


class EfficientvitsamConfig(SamConfig):
    pass


class EfficientvitsamVisionEncoderOutput(SamVisionEncoderOutput):
    pass


class EfficientvitsamImageSegmentationOutput(SamImageSegmentationOutput):
    pass


class EfficientvitsamPatchEmbeddings(SamPatchEmbeddings):
    pass


class EfficientvitsamMLPBlock(SamMLPBlock):
    pass


class EfficientvitsamLayerNorm(SamLayerNorm):
    pass


class EfficientvitsamAttention(SamAttention):
    pass


class EfficientvitsamTwoWayAttentionBlock(SamTwoWayAttentionBlock):
    pass


class EfficientvitsamTwoWayTransformer(SamTwoWayTransformer):
    pass


class EfficientvitsamFeedForward(SamFeedForward):
    pass


class EfficientvitsamMaskDecoder(SamMaskDecoder):
    pass


class EfficientvitsamPositionalEmbedding(SamPositionalEmbedding):
    pass


class EfficientvitsamMaskEmbedding(SamMaskEmbedding):
    pass


class EfficientvitsamPromptEncoder(SamPromptEncoder):
    pass


class EfficientvitsamVisionAttention(SamVisionAttention):
    pass


class EfficientvitsamVisionSdpaAttention(SamVisionSdpaAttention):
    pass


class EfficientvitsamVisionLayer(SamVisionLayer):
    pass


class EfficientvitsamVisionNeck(SamVisionNeck):
    pass


class EfficientvitsamPreTrainedModel(SamPreTrainedModel):
    pass


class EfficientvitsamVisionEncoder(SamVisionEncoder):
    pass


class EfficientvitsamVisionModel(SamVisionModel):
    pass


class EfficientvitsamModel(SamModel):
    pass


class EfficientvitsamImageProcessorKwargs(SamImageProcessorKwargs):
    pass


class EfficientvitsamImageProcessor(SamImageProcessor):
    pass


class EfficientvitsamImageProcessorPil(SamImageProcessorPil):
    pass


class EfficientvitsamImagesKwargs(SamImagesKwargs):
    pass


class EfficientvitsamProcessorKwargs(SamProcessorKwargs):
    pass


class EfficientvitsamProcessor(SamProcessor):
    pass


__all__ = [
    "EfficientvitsamConfig",
    "EfficientvitsamMaskDecoderConfig",
    "EfficientvitsamPromptEncoderConfig",
    "EfficientvitsamVisionConfig",
    "EfficientvitsamVisionModel",
    "EfficientvitsamModel",
    "EfficientvitsamPreTrainedModel",
    "EfficientvitsamImageProcessor",
    "EfficientvitsamImageProcessorKwargs",
    "EfficientvitsamImageProcessorPil",
    "EfficientvitsamProcessor",
]
