# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel, AutoModelForImageClassification
from ..pe_audio_video.modeling_pe_audio_video import (
    PeAudioVideoContrastiveHead,
    PeAudioVideoEncoder,
    PeAudioVideoEncoderPatchEmbedder,
    PeAudioVideoPreTrainedModel,
)
from .configuration_pe_video import PeVideoConfig, PeVideoEncoderConfig


# TODO: not sure about the typing for text_model_output
@dataclass
# @auto_docstring
class PeVideoOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_video_text: torch.FloatTensor | None = None
    text_video_embeds: torch.FloatTensor | None = None
    video_embeds: torch.FloatTensor | None = None
    text_outputs: BaseModelOutputWithPooling = None
    video_outputs: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_outputs", "video_outputs"] else getattr(self, k).to_tuple() for k in self.keys()
        )


class PeVideoContrastiveHead(PeAudioVideoContrastiveHead): ...


class PeVideoEncoderPatchEmbedder(PeAudioVideoEncoderPatchEmbedder): ...


class PeVideoEncoderEmbedder(nn.Module):
    def __init__(self, config: PeVideoEncoderConfig):
        super().__init__()
        self.vision_model = AutoModelForImageClassification.from_config(config.vision_config)
        self.proj = nn.Linear(config.vision_config.num_labels, config.hidden_size, bias=False)
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = pixel_values_videos.shape

        pixel_values_videos = pixel_values_videos.view(-1, *input_shape[2:])
        vision_encoder_outputs = self.vision_model(pixel_values_videos)

        logits = vision_encoder_outputs.logits.view(*input_shape[:2], -1)
        logits = F.normalize(logits, dim=-1)

        vision_features = self.proj(logits)
        inputs_embeds = self.data_proj(vision_features)

        return inputs_embeds, padding_mask


class PeVideoPreTrainedModel(PeAudioVideoPreTrainedModel):
    base_model_prefix = "video_model"
    main_input_name = "pixel_values_videos"


@auto_docstring(
    custom_intro="""
    The PeVideo Encoder model.
    """
)
class PeVideoEncoder(PeAudioVideoEncoder):
    base_model_prefix = "video_model.video_encoder"
    main_input_name = "pixel_values_videos"

    def __init__(self, config: PeVideoEncoderConfig):
        super().__init__(config)
        self.embedder = PeVideoEncoderEmbedder(config)

    @can_return_tuple
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        inputs_embeds, padding_mask = self.embedder(pixel_values_videos, padding_mask=padding_mask_videos)
        inputs_embeds, attention_mask = self.patch_embedder(inputs_embeds, padding_mask=padding_mask)

        if attention_mask is not None:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
        )


class PeVideoModel(PeVideoPreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config: PeVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.video_encoder = PeVideoEncoder(config.video_config)

        self.text_video_head = PeVideoContrastiveHead(config.text_config.hidden_size, config.text_config.hidden_size)
        self.video_head = PeVideoContrastiveHead(config.video_config.hidden_size, config.text_config.hidden_size)

        self.text_video_logit_scale = nn.Parameter(torch.zeros(1))
        self.text_video_logit_bias = nn.Parameter(torch.zeros(1))

        self.post_init()

        @can_return_tuple
        @auto_docstring
        def get_text_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
        ) -> tuple | BaseModelOutputWithPooling:
            text_outputs: BaseModelOutputWithPooling = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
            text_outputs.pooler_output = self.text_video_head(text_outputs.last_hidden_state)
            return text_outputs

        @can_return_tuple
        @auto_docstring
        def get_video_features(
            self,
            pixel_values_videos: torch.Tensor,
            padding_mask_videos: torch.Tensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
        ) -> tuple | BaseModelOutputWithPooling:
            video_outputs: BaseModelOutputWithPooling = self.video_encoder(
                pixel_values_videos=pixel_values_videos,
                padding_mask_videos=padding_mask_videos,
                return_dict=True,
                **kwargs,
            )
            video_outputs.pooler_output = self.video_head(video_outputs.pooler_output)
            return video_outputs

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask_videos: torch.Tensor | None = None,
        return_loss: bool | None = None,
        **kwargs,
    ) -> PeVideoOutput:
        video_outputs: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos, padding_mask_videos=padding_mask_videos, **kwargs
        )
        kwargs["output_hidden_states"] = True
        text_outputs: MaskedLMOutput = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        video_embeds = video_outputs.pooler_output
        video_embeds = self.video_head(video_embeds)

        text_video_embeds = text_outputs.hidden_states[-1][:, 0]
        text_video_embeds = self.text_video_head(text_video_embeds)

        logits_video_text = video_embeds @ text_video_embeds.T
        logits_video_text = logits_video_text * self.text_video_logit_scale + self.text_video_logit_bias

        loss = None
        if return_loss:
            labels = torch.eye(logits_video_text.shape[0], device=logits_video_text.device)
            loss = -F.logsigmoid(labels * logits_video_text).sum() / logits_video_text.shape[0]

        return PeVideoOutput(
            logits_video_text=logits_video_text,
            text_video_embeds=text_video_embeds,
            video_embeds=video_embeds,
            text_outputs=text_outputs,
            video_outputs=video_outputs,
            loss=loss,
        )


__all__ = [
    "PeVideoEncoder",
    "PeVideoModel",
]
