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

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel
from ..dac.modeling_dac import DacEncoder, DacEncoderBlock, Snake1d
from ..pe_audio_video.modeling_pe_audio_video import (
    PeAudioVideoContrastiveHead,
    PeAudioVideoEncoder,
    PeAudioVideoPreTrainedModel,
)
from .configuration_pe_audio import PeAudioConfig, PeAudioEncoderConfig


class PeAudioDacEncoderBlock(DacEncoderBlock):
    def __init__(self, config: PreTrainedConfig, stride: int = 1, stride_index: int = 1):
        super().__init__(config, stride=stride, stride_index=stride_index)


class PeAudioDacEncoder(DacEncoder):
    def __init__(self, config: PreTrainedConfig):
        super().__init__(config)


class PeAudioEncoderEmbedder(nn.Module):
    def __init__(self, config: PeAudioEncoderConfig):
        super().__init__()
        self.dac_encoder = PeAudioDacEncoder(config.dac_config)
        self.bottleneck = nn.Conv1d(config.dac_config.hidden_size, config.dac_config.codebook_dim, 1)
        self.data_proj = nn.Linear(config.dac_config.codebook_dim, config.hidden_size)
        self.config = config

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            hidden_states = self.dac_encoder(input_values)
            hidden_states = self.bottleneck(hidden_states)

        codec_features = hidden_states.transpose(1, 2)
        inputs_embeds = self.data_proj(codec_features)

        if padding_mask is not None:
            padding_mask = padding_mask[:, :: self.config.dac_config.hop_length]

        return inputs_embeds, padding_mask


class PeAudioContrastiveHead(PeAudioVideoContrastiveHead): ...


class PeAudioPreTrainedModel(PeAudioVideoPreTrainedModel):
    base_model_prefix = "audio_model"

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv1d):
            init.trunc_normal_(module.weight, std=0.02)
            init.constant_(module.bias, 0)
        elif isinstance(module, Snake1d):
            init.ones_(module.alpha)
        elif isinstance(module, nn.ConvTranspose1d):
            module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=0.02)


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PeAudioEncoder`].
    """
)
class PeAudioEncoderOutput(BaseModelOutputWithPooling):
    codec_features: torch.FloatTensor | None = None
    output_mask: tuple[torch.FloatTensor] | None = None


# TODO: add the capture of codec features?
@auto_docstring(
    custom_intro="""
    The PeAudio Encoder model.
    """
)
class PeAudioEncoder(PeAudioVideoEncoder):
    base_model_prefix = "audio_model.audio_encoder"

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, padding_mask = self.embedder(input_values, padding_mask=padding_mask)
        inputs_embeds, attention_mask = self.patch_embedder(inputs_embeds, padding_mask=padding_mask)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

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

        return PeAudioEncoderOutput(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
            output_mask=padding_mask,
        )


# TODO: not sure about the typing for text_model_output
@dataclass
# @auto_docstring
class PeAudioOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_audio_text: torch.FloatTensor | None = None
    text_audio_embeds: torch.FloatTensor | None = None
    audio_embeds: torch.FloatTensor | None = None
    text_outputs: BaseModelOutputWithPooling = None
    audio_outputs: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_outputs", "audio_outputs"] else getattr(self, k).to_tuple() for k in self.keys()
        )


class PeAudioModel(PeAudioPreTrainedModel):
    def __init__(self, config: PeAudioConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = PeAudioEncoder(config.audio_config)

        self.text_audio_head = PeAudioContrastiveHead(config.text_config.hidden_size, config.text_config.hidden_size)
        self.audio_head = PeAudioContrastiveHead(config.audio_config.hidden_size, config.text_config.hidden_size)

        self.text_audio_logit_scale = nn.Parameter(torch.zeros(1))
        self.text_audio_logit_bias = nn.Parameter(torch.zeros(1))

        self.post_init()

    def get_text_audio_embeds(self, input_ids, attention_mask=None):
        # TODO: naming can be improved here...
        text_outputs: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_audio_embeds = text_outputs.hidden_states[-1][:, 0]
        return self.text_audio_head(text_audio_embeds)

    def get_audio_embeds(self, input_values, padding_mask=None):
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            return_dict=True,
        )
        audio_embeds = audio_outputs.pooler_output
        return self.audio_head(audio_embeds)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_loss: bool | None = None,
        **kwargs,
    ) -> PeAudioOutput:
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values, padding_mask=padding_mask, **kwargs
        )

        kwargs["output_hidden_states"] = True
        text_outputs: MaskedLMOutput = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        audio_embeds = audio_outputs.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        text_audio_embeds = text_outputs.hidden_states[-1][:, 0]
        text_audio_embeds = self.text_audio_head(text_audio_embeds)

        logits_audio_text = audio_embeds @ text_audio_embeds.T
        logits_audio_text = logits_audio_text * self.text_audio_logit_scale + self.text_audio_logit_bias

        loss = None
        if return_loss:
            labels = torch.eye(logits_audio_text.shape[0], device=logits_audio_text.device)
            loss = -F.logsigmoid(labels * logits_audio_text).sum() / logits_audio_text.shape[0]

        return PeAudioOutput(
            logits_audio_text=logits_audio_text,
            text_audio_embeds=text_audio_embeds,
            audio_embeds=audio_embeds,
            text_outputs=text_outputs,
            audio_outputs=audio_outputs,
            loss=loss,
        )


# TODO: underline in documentation that logits output shape is
# 1. Model: (n_audio, n_text)
# 2. Frame-level: (n_audio, n_text, n_frames)
class PeAudioFrameLevelModel(PeAudioModel):
    def get_audio_embeds(self, input_values, padding_mask=None):
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            return_dict=True,
        )
        audio_embeds = audio_outputs.last_hidden_state
        audio_embeds = self.audio_head(audio_embeds)
        return audio_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_loss: bool | None = None,
        **kwargs,
    ) -> PeAudioOutput:
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values, padding_mask=padding_mask, **kwargs
        )
        kwargs["output_hidden_states"] = True
        text_outputs: MaskedLMOutput = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        audio_embeds = audio_outputs.last_hidden_state
        audio_embeds = self.audio_head(audio_embeds)

        text_audio_embeds = text_outputs.hidden_states[-1][:, 0]
        text_audio_embeds = self.text_audio_head(text_audio_embeds)

        logits_audio_text = (audio_embeds @ text_audio_embeds.T).transpose(1, 2)
        logits_audio_text = logits_audio_text * self.text_audio_logit_scale + self.text_audio_logit_bias

        loss = None
        if return_loss:
            labels = torch.eye(logits_audio_text.shape[0], device=logits_audio_text.device)
            loss = -F.logsigmoid(labels * logits_audio_text).sum() / logits_audio_text.shape[0]

        return PeAudioOutput(
            logits_audio_text=logits_audio_text,
            text_audio_embeds=text_audio_embeds,
            audio_embeds=audio_embeds,
            text_outputs=text_outputs,
            audio_outputs=audio_outputs,
            loss=loss,
        )


__all__ = [
    "PeAudioFrameLevelModel",
    "PeAudioModel",
    "PeAudioEncoder",
]
