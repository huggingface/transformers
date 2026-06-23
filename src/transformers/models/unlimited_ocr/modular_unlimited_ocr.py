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
import torch
from torch import nn

from ..clip.configuration_clip import CLIPVisionConfig
from ..deepseek_ocr2.configuration_deepseek_ocr2 import (
    DeepseekOcr2Config,
    DeepseekOcr2TextConfig,
    DeepseekOcr2VisionConfig,
)
from ..deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2CausalLMOutputWithPast,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2Model,
    DeepseekOcr2ModelOutputWithPast,
    DeepseekOcr2ModelOutputWithPooling,
    DeepseekOcr2PreTrainedModel,
    DeepseekOcr2SamLayerNorm,
    DeepseekOcr2SamMLPBlock,
    DeepseekOcr2SamPatchEmbeddings,
    DeepseekOcr2SamVisionAttention,
    DeepseekOcr2SamVisionEncoder,
    DeepseekOcr2SamVisionLayer,
    DeepseekOcr2SamVisionNeck,
    DeepseekOcr2SamVisionProj,
    DeepseekOcr2SamVisionSdpaAttention,
    DeepseekOcr2TextAttention,
    DeepseekOcr2TextDecoderLayer,
    DeepseekOcr2TextExperts,
    DeepseekOcr2TextMLP,
    DeepseekOcr2TextModel,
    DeepseekOcr2TextMoe,
    DeepseekOcr2TextPreTrainedModel,
    DeepseekOcr2TextRMSNorm,
    DeepseekOcr2TextRotaryEmbedding,
    DeepseekOcr2VisionAttention,
    DeepseekOcr2VisionEncoder,
    DeepseekOcr2VisionEncoderLayer,
    DeepseekOcr2VisionMLP,
    DeepseekOcr2VisionModel,
    DeepseekOcr2VisionRMSNorm,
    DeepseekOcr2VisionRotaryEmbedding,
)
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig


class UnlimitedOcrSamVisionConfig(GotOcr2VisionConfig):
    pass


class UnlimitedOcrVisionEncoderConfig(CLIPVisionConfig):
    pass


class UnlimitedOcrVisionConfig(DeepseekOcr2VisionConfig):
    pass


class UnlimitedOcrTextConfig(DeepseekOcr2TextConfig):
    pass


class UnlimitedOcrConfig(DeepseekOcr2Config):
    pass


class UnlimitedOcrModelOutputWithPooling(DeepseekOcr2ModelOutputWithPooling):
    pass


class UnlimitedOcrModelOutputWithPast(DeepseekOcr2ModelOutputWithPast):
    pass


class UnlimitedOcrCausalLMOutputWithPast(DeepseekOcr2CausalLMOutputWithPast):
    pass


class UnlimitedOcrPreTrainedModel(DeepseekOcr2PreTrainedModel):
    pass


class UnlimitedOcrSamVisionAttention(DeepseekOcr2SamVisionAttention):
    pass


class UnlimitedOcrSamMLPBlock(DeepseekOcr2SamMLPBlock):
    pass


class UnlimitedOcrSamVisionSdpaAttention(DeepseekOcr2SamVisionSdpaAttention):
    pass


class UnlimitedOcrSamVisionLayer(DeepseekOcr2SamVisionLayer):
    pass


class UnlimitedOcrSamLayerNorm(DeepseekOcr2SamLayerNorm):
    pass


class UnlimitedOcrSamVisionNeck(DeepseekOcr2SamVisionNeck):
    pass


class UnlimitedOcrSamPatchEmbeddings(DeepseekOcr2SamPatchEmbeddings):
    pass


class UnlimitedOcrSamVisionProj(DeepseekOcr2SamVisionProj):
    pass


class UnlimitedOcrSamVisionEncoder(DeepseekOcr2SamVisionEncoder):
    pass


class UnlimitedOcrVisionMLP(DeepseekOcr2VisionMLP):
    pass


class UnlimitedOcrVisionRMSNorm(DeepseekOcr2VisionRMSNorm):
    pass


class UnlimitedOcrVisionRotaryEmbedding(DeepseekOcr2VisionRotaryEmbedding):
    pass


class UnlimitedOcrVisionAttention(DeepseekOcr2VisionAttention):
    pass


class UnlimitedOcrVisionEncoderLayer(DeepseekOcr2VisionEncoderLayer):
    pass


class UnlimitedOcrVisionEncoder(DeepseekOcr2VisionEncoder):
    pass


class UnlimitedOcrVisionModel(DeepseekOcr2VisionModel):
    pass


class UnlimitedOcrTextRotaryEmbedding(DeepseekOcr2TextRotaryEmbedding):
    pass


class UnlimitedOcrTextAttention(DeepseekOcr2TextAttention):
    pass


class UnlimitedOcrTextMLP(DeepseekOcr2TextMLP):
    pass


class UnlimitedOcrTextExperts(DeepseekOcr2TextExperts):
    pass


class UnlimitedOcrTextMoe(DeepseekOcr2TextMoe):
    pass


class UnlimitedOcrTextRMSNorm(DeepseekOcr2TextRMSNorm):
    pass


class UnlimitedOcrTextDecoderLayer(DeepseekOcr2TextDecoderLayer):
    pass


class UnlimitedOcrTextPreTrainedModel(DeepseekOcr2TextPreTrainedModel):
    pass


class UnlimitedOcrTextModel(DeepseekOcr2TextModel):
    pass


class UnlimitedOcrModel(DeepseekOcr2Model):
    def __init__(self, config: UnlimitedOcrConfig):
        super().__init__(config)
        n_embed = 1280
        self.multi_modal_projector = nn.Linear(
            config.vision_config.sam_config.hidden_size + config.vision_config.encoder_config.hidden_size, n_embed
        )
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)


class UnlimitedOcrForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    pass


__all__ = [
    "UnlimitedOcrConfig",
    "UnlimitedOcrTextConfig",
    "UnlimitedOcrVisionConfig",
    "UnlimitedOcrVisionEncoderConfig",
    "UnlimitedOcrSamVisionConfig",
    "UnlimitedOcrForConditionalGeneration",
    "UnlimitedOcrModel",
    "UnlimitedOcrPreTrainedModel",
    "UnlimitedOcrTextModel",
    "UnlimitedOcrTextPreTrainedModel",
    "UnlimitedOcrVisionModel",
]
