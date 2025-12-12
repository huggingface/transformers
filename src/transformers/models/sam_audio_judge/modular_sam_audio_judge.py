# coding=utf-8
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
from typing import Any, Optional

import torch
import torch.nn as nn

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput
from ..auto import AutoModel
from ..pe_audio_video.modular_pe_audio_video import PeAudioVideoEncoder
from .configuration_sam_audio_judge import SamAudioJudgeConfig


class SamAudioJudgeEmbedder(nn.Module):
    def __init__(self, config: SamAudioJudgeConfig):
        super().__init__()
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = AutoModel.from_config(config.audio_config)

        self.cat_audio_proj = nn.Linear(2 * config.audio_config.hidden_size, config.bottleneck_dim)
        self.text_proj_1 = nn.Linear(
            in_features=config.text_config.hidden_size, out_features=config.audio_config.hidden_size, bias=False
        )
        self.text_proj_2 = nn.Linear(in_features=config.audio_config.hidden_size, out_features=config.bottleneck_dim)
        self.audio_text_proj_1 = nn.Linear(2 * config.bottleneck_dim, config.bottleneck_dim)
        self.audio_text_proj_2 = nn.Linear(config.bottleneck_dim, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.bottleneck_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            **{**kwargs, "return_dict": True},
        )

        text_outputs: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )

        audio_embeds, audio_hyp_embeds = audio_outputs.last_hidden_state.chunk(2, 0)
        audio_embeds = torch.cat([audio_hyp_embeds, audio_embeds], dim=2)
        audio_embeds = self.cat_audio_proj(audio_embeds)

        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_proj_1(text_embeds)
        text_embeds = self.text_proj_2(text_embeds)
        text_embeds = self.layer_norm(text_embeds)
        text_embeds = text_embeds[:, None, :].expand_as(audio_embeds)

        audio_text_embeds = torch.cat([audio_embeds, text_embeds], dim=2)
        audio_text_embeds = self.audio_text_proj_1(audio_text_embeds)
        audio_text_embeds = self.audio_text_proj_2(audio_text_embeds)

        output_mask = audio_outputs.output_mask

        return audio_text_embeds, output_mask, audio_outputs, text_outputs


@dataclass
class SamAudioJudgeOutput(BaseModelOutput):
    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class SamAudioJudgeModel(PeAudioVideoEncoder):
    def __init__(self, config: SamAudioJudgeConfig):
        super().__init__(config)
        self.output_proj = nn.Linear(config.hidden_size, 4, bias=False)
        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("std", torch.ones(4))

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SamAudioJudgeOutput:
        audio_text_embeds, output_mask, audio_outputs, text_outputs = self.embedder(
            input_ids=input_ids,
            input_values=input_values,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        inputs_embeds, attention_mask = self.patch_embedder(audio_text_embeds, padding_mask=output_mask)

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

        logits = self.output_proj(hidden_states[:, 1:])
        pooled_logits = torch.masked.mean(logits, mask=output_mask, dim=1)
        de_normalized_logits = pooled_logits * self.std + self.mean

        overall, recall, precision, faithfulness = de_normalized_logits.chunk(4, dim=1)

        return SamAudioJudgeOutput(
            overall=overall,
            recall=recall,
            precision=precision,
            faithfulness=faithfulness,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            last_hidden_state=hidden_states[:, 1:],
        )


__all__ = ["SamAudioJudgeModel", "SamAudioJudgeConfig"]
