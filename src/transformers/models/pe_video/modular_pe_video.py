from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel, AutoModelForImageClassification
from ..pe_audio.modeling_pe_audio import PEAudioTransformer, PEAudioContrastiveHead, PEAudioPretrainedModel
from .configuration_pe_video import PEVideoConfig, PEVideoEncoderConfig


class PEVideoTransformer(PEAudioTransformer): ...


class PEVideoContrastiveHead(PEAudioContrastiveHead): ...


class PEVideoPretrainedModel(PEAudioPretrainedModel): ...


# TODO: not sure about the typing for text_model_output
@dataclass
@auto_docstring
class PEVideoOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for video-text similarity.
    logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
        The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
        The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`PEVideoTextModel`].
    video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The video embeddings obtained by applying the projection layer to the pooled output of [`PEVideoEncoder`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PEVideoTextModel`].
    video_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PEVideoEncoder`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_video: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    video_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "video_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PEVideoEncoder`].
    """
)
class PEVideoEncoderOutput(BaseModelOutputWithPooling):
    vision_features: Optional[torch.FloatTensor] = None
    outputs_mask: Optional[tuple[torch.FloatTensor]] = None


class PEVideoEncoder(PEVideoPretrainedModel):
    config_class = PEVideoEncoderConfig
    main_input_name = "pixel_values_videos"
    base_model_prefix = "video_encoder"

    def __init__(self, config: PEVideoEncoderConfig):
        super().__init__(config)
        # NOTE: we use `AutoModelForImageClassification` instead of `AutoModel`
        # because `TimmWrapperModel` forces `num_classes=0` (https://github.com/huggingface/transformers/blob/53838edde77cb10f3a360150aa85a457637e9ac3/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L163)
        # which drops the final linear projection
        self.vision_encoder = AutoModelForImageClassification.from_config(config.vision_encoder_config)
        self.proj = nn.Linear(config.vision_encoder_config.num_labels, config.hidden_size, bias=False)
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.transformer = PEVideoTransformer(config)

        self.post_init()

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> PEVideoEncoderOutput:
        input_shape = pixel_values_videos.shape

        pixel_values_videos = pixel_values_videos.view(-1, *input_shape[2:])
        vision_encoder_outputs = self.vision_encoder(pixel_values_videos)

        logits = vision_encoder_outputs.logits.view(*input_shape[:2], -1)
        logits = F.normalize(logits, dim=-1)
        
        vision_features = self.proj(logits)
        projected = self.data_proj(vision_features)
        outputs = self.transformer(projected, attention_mask=padding_mask_videos, **kwargs)

        return PEVideoEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_features=vision_features,
            outputs_mask=padding_mask_videos,
        )


class PEVideoModel(PEVideoPretrainedModel):
    def __init__(self, config: PEVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.video_encoder = PEVideoEncoder(config.video_config)

        self.text_head_video = PEVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.video_head = PEVideoContrastiveHead(config.video_config.hidden_size, config.projection_dim)

        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_bias = nn.Parameter(torch.zeros(1))

        self.post_init()

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

        text_features = self.text_head_video(text_features)
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

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        **kwargs,
    ) -> PEVideoOutput:
        video_output: BaseModelOutputWithPooling = self.video_encoder(
            pixel_values_videos=pixel_values_videos,
            padding_mask_videos=padding_mask_videos,
            **{**kwargs, "return_dict": True},
        )

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
        )

        video_embeds = video_output.pooler_output
        if self.config.nth_text_layer is not None:
            text_embeds = text_output.hidden_states[self.config.nth_text_layer]
        else:
            text_embeds = text_output.last_hidden_state

        video_embeds = self.video_head(video_embeds)
        text_embeds = self.text_head_video(text_embeds)

        # TODO: something is wrong in the logic here, should be using pooler?
        logits_per_text = text_embeds @ video_embeds.t()
        logits_per_text = logits_per_text * self.logit_scale.exp() + self.logit_bias
        # TODO: there is not logits per video?

        loss = None
        if return_loss:
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
