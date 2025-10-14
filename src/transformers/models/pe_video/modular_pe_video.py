from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ..auto import AutoModel, AutoModelForImageClassification
from ..clip.modeling_clip import CLIPOutput
from ..pe_audio.modeling_pe_audio import (
    PEAudioContrastiveHead,
    PEAudioTransformer,
)
from .configuration_pe_video import PEVideoConfig, PEVideoEncoderConfig


class PEVideoTransformer(PEAudioTransformer): ...


class PEVideoContrastiveHead(PEAudioContrastiveHead): ...


class PEVideoPretrainedModel(PreTrainedModel):
    config: PEVideoConfig
    base_model_prefix = "pe_video"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


# TODO: not sure about the typing for text_model_output
class PEVideoOutput(CLIPOutput): ...


class PEVideoEncoder(PEVideoPretrainedModel):
    config_class = PEVideoEncoderConfig
    base_model_prefix = "video_encoder"

    def __init__(self, config: PEVideoEncoderConfig):
        super().__init__(config)
        # NOTE: we use `AutoModelForImageClassification` instead of `AutoModel`
        # because `TimmWrapperModel` forces `num_classes=0` (https://github.com/huggingface/transformers/blob/53838edde77cb10f3a360150aa85a457637e9ac3/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L163)
        # which drops the final linear projection
        self.vision_encoder = AutoModelForImageClassification.from_config(config.vision_encoder_config)
        self.proj = nn.Linear(config.vision_encoder_config.num_classes, config.hidden_size, bias=False)
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.transformer = PEVideoTransformer(config)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        pixel_values_videos = pixel_values_videos.view(-1, *pixel_values_videos.shape[2:])

        vision_encoder_outputs = self.vision_encoder(pixel_values_videos)
        logits = vision_encoder_outputs.logits.view(*pixel_values_videos.shape[:2], -1)
        logits = F.normalize(logits, dim=-1)
        projected = self.proj(logits)
        outputs = self.transformer(self.data_proj(projected), attention_mask=padding_mask_videos)

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PEVideoModel(PEVideoPretrainedModel):
    def __init__(self, config: PEVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.video_encoder = PEVideoEncoder(config.video_config)

        self.text_head_video = PEVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.video_head = PEVideoContrastiveHead(config.video_config.hidden_size, config.projection_dim)

        self.logit_scale = nn.Parameter(torch.tensor([config.logit_scale_init_value]).log())
        self.logit_bias = nn.Parameter(torch.tensor([config.logit_bias_init_value]))

    def get_text_features(self, input_ids, attention_mask=None):
        # TODO: should it be named feature or embeds
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if self.config.nth_text_layer is not None:
            text_features = text_outputs.hidden_states[self.config.nth_text_layer]
        else:
            text_features = text_outputs.last_hidden_state

        text_features = self.text_head(text_features)
        return text_features

    def get_video_features(self, pixel_values_videos, padding_mask_videos=None):
        # TODO: should it be named feature or embeds
        video_outputs: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            return_dict=True,
        )
        video_features = self.video_head(video_outputs.pooler_output)
        return video_features

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
    ) -> PEVideoOutput:
        video_output: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            return_dict=True,
        )

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.nth_text_layer,
            return_dict=True,
        )

        video_embeds = video_output.pooler_output
        if self.config.nth_video_layer is not None:
            text_embeds = text_output.hidden_states[self.config.nth_video_layer]
        else:
            text_embeds = text_output.last_hidden_state

        video_embeds = self.video_head(video_embeds)
        text_embeds = self.text_head(text_embeds)

        # TODO: there is not logits per video?

        loss = None
        if return_loss:
            logits_per_text = text_embeds @ video_embeds.t()
            logits_per_text = logits_per_text * self.logit_scale + self.logit_bias
            labels = torch.eye(text_embeds.shape[0], device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.shape[0]

        return PEVideoOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_output,
            video_model_output=video_output,
        )


__all__ = [
    "PEVideoModel",
    "PEVideoEncoder",
]
