from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from ..auto import AutoModel
from ..pe_audio.modeling_pe_audio import PEAudioTransformer
from .configuration_sam_audio_judge import SamAudioJudgeConfig


class SamAudioJudgeTransformer(PEAudioTransformer): ...


class SamAudioJudgePretrainedModel(PreTrainedModel):
    config: SamAudioJudgeConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@dataclass
class SamAudioJudgeOutput(ModelOutput):
    r"""
    overall (torch.Tensor, optional): Overall score tensor of shape (batch_size, 1).
    recall (torch.Tensor, optional): Recall score tensor of shape (batch_size, 1).
    precision (torch.Tensor, optional): Precision score tensor of shape (batch_size, 1).
    faithfulness (torch.Tensor, optional): Faithfulness score tensor of shape (batch_size, 1).
    text_model_output (BaseModelOutputWithPooling): Output from the text model.
    audio_model_output (BaseModelOutputWithPooling): Output from the audio model.
    """

    overall: Optional[torch.Tensor] = None
    recall: Optional[torch.Tensor] = None
    precision: Optional[torch.Tensor] = None
    faithfulness: Optional[torch.Tensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class SamAudioJudgeModel(SamAudioJudgePretrainedModel):
    def __init__(self, config: SamAudioJudgeConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = AutoModel.from_config(config.audio_config)
        self.transformer = SamAudioJudgeTransformer(config)

        self.text_proj1 = nn.Linear(
            in_features=config.text_config.hidden_size, out_features=config.audio_config.hidden_size, bias=False
        )
        self.text_proj2 = nn.Linear(in_features=config.audio_config.hidden_size, out_features=config.bottleneck_dim)

        self.cat_audio_proj = nn.Linear(2 * config.audio_config.hidden_size, config.bottleneck_dim)
        self.proj_audio_and_text = nn.Linear(2 * config.bottleneck_dim, config.bottleneck_dim)
        self.data_proj = nn.Linear(config.bottleneck_dim, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, 4, bias=False)
        self.layer_norm = nn.LayerNorm(config.bottleneck_dim)

        self.register_buffer("mean", torch.zeros(4))
        self.register_buffer("std", torch.ones(4))

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        separated_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> SamAudioJudgeOutput:
        stacked_audio = torch.cat([input_values, separated_values], dim=0)

        audio_output: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=stacked_audio,
            padding_mask=padding_mask,
            return_dict=True,
        )

        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[:, :: self.config.audio_config.dac_config.hop_length]

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.nth_text_layer,
            return_dict=True,
        )

        if self.config.nth_text_layer is not None:
            text_embeds = text_output.hidden_states[self.config.nth_text_layer]
        else:
            text_embeds = text_output.last_hidden_state

        text_embeds = self.text_proj1(text_embeds)

        audio_embeds, audio_hyp_embeds = audio_output.last_hidden_state.chunk(2, 0)
        audio_embeds = torch.cat([audio_hyp_embeds, audio_embeds], dim=2)
        audio_embeds = self.cat_audio_proj(audio_embeds)

        text_embeds = self.text_proj2(text_embeds.unsqueeze(1).expand_as(audio_embeds))
        text_embeds = self.layer_norm(text_embeds)

        audio_text_embeds = torch.cat([audio_embeds, text_embeds], dim=2)
        audio_text_embeds = self.proj_audio_and_text(audio_text_embeds)
        audio_text_embeds = self.data_proj(audio_text_embeds)
        audio_text_output = self.transformer(audio_text_embeds, padding_mask=feature_padding_mask)

        logits = self.head(audio_text_output.last_hidden_state)
        pooled = torch.masked.mean(logits, mask=feature_padding_mask, dim=1)
        de_normalized = pooled * self.std + self.mean

        return SamAudioJudgeOutput(*de_normalized.chunk(4, dim=1))


__all__ = ["SamAudioJudgeModel"]
