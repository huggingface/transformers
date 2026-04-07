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

from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLRMSNorm,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2MLP,
)


class NewQwen25VlVisionConfig(Qwen2_5_VLVisionConfig):
    pass


class NewQwen25VlTextConfig(Qwen2_5_VLTextConfig):
    pass


class NewQwen25VlConfig(Qwen2_5_VLConfig):
    pass


class NewQwen25VlRMSNorm(Qwen2_5_VLRMSNorm):
    pass


class NewQwen25VlMLP(Qwen2_5_VLMLP):
    pass


class Qwen2_5_VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    pass


class Qwen2_5_VisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class NewQwen25VlPatchMerger(Qwen2_5_VLPatchMerger):
    pass


class NewQwen25VlVisionAttention(Qwen2_5_VLVisionAttention):
    pass


class NewQwen25VlVisionBlock(Qwen2_5_VLVisionBlock):
    pass


class NewQwen25VlPreTrainedModel(Qwen2_5_VLPreTrainedModel):
    pass


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):
    pass


class NewQwen25VlModelOutputWithPast(Qwen2_5_VLModelOutputWithPast):
    pass


class NewQwen25VlRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
    pass


class Qwen2MLP(Qwen2MLP):
    pass


class NewQwen25VlAttention(Qwen2_5_VLAttention):
    pass


class NewQwen25VlDecoderLayer(Qwen2_5_VLDecoderLayer):
    pass


class NewQwen25VlTextModel(Qwen2_5_VLTextModel):
    pass


class NewQwen25VlModel(Qwen2_5_VLModel):
    pass


class NewQwen25VlCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    pass


class NewQwen25VlForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    pass


__all__ = [
    "NewQwen25VlConfig",
    "NewQwen25VlTextConfig",
    "NewQwen25VlForConditionalGeneration",
    "NewQwen25VlModel",
    "NewQwen25VlPreTrainedModel",
    "NewQwen25VlTextModel",
]
