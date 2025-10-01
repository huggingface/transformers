from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel, AutoModelForImageClassification
from ..pe_audio.configuration_pe_audio import PEAudioTransformerConfig
from ..pe_audio.modeling_pe_audio import (
    PEAudioContrastiveHead,
    PEAudioTransformer,
)
from ..timm_wrapper import TimmWrapperConfig


class PEVideoTransformerConfig(PEAudioTransformerConfig): ...


class PEVideoEncoderConfig(PretrainedConfig):
    r"""
    PEVideoEncoderConfig
    """

    sub_configs = {"transformer": PEVideoTransformerConfig, "clip_vision_model": TimmWrapperConfig}
    model_type = "pe_video"

    def __init__(
        self,
        clip_vision_model: Optional[dict] = None,
        transformer: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        transformer = transformer or {}
        clip_vision_model = clip_vision_model or {}

        self.clip_vision_model = TimmWrapperConfig(**clip_vision_model)
        self.transformer = PEVideoTransformerConfig(**transformer)


class PEVideoConfig(PretrainedConfig):
    r"""
    PEVideoConfig
    """

    sub_configs = {"video_model": PEVideoEncoderConfig, "text_model": AutoConfig}

    def __init__(
        self,
        video_model: Optional[dict] = None,
        text_model: Optional[dict] = None,
        output_dim: int = 1024,
        nth_text_layer: Optional[int] = 22,
        **kwargs,
    ):
        super().__init__(**kwargs)
        text_model = text_model or {}
        video_model = video_model or {}

        self.video_model = PEVideoEncoderConfig(**video_model)
        self.text_model = CONFIG_MAPPING[text_model.get("model_type", "modernbert")](**text_model)
        self.output_dim = output_dim
        self.nth_text_layer = nth_text_layer


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


class PEVideoModel(PEVideoPretrainedModel):
    def __init__(self, config: PEVideoEncoderConfig):
        super().__init__(config)
        # Synchronize _attn_implementations
        config.transformer._attn_implementation = config._attn_implementation
        # NOTE: we use `AutoModelForImageClassification` instead of `AutoModelForImageClassification`
        # because `TimmWrapperModel` forces `num_classes=0` (https://github.com/huggingface/transformers/blob/53838edde77cb10f3a360150aa85a457637e9ac3/src/transformers/models/timm_wrapper/modeling_timm_wrapper.py#L163)
        # which drops the final linear projection
        self.clip_vision_model = AutoModelForImageClassification.from_config(config.clip_vision_model)
        self.proj = torch.nn.Linear(config.clip_vision_model.num_labels, config.transformer.hidden_size, bias=False)
        self.data_proj = torch.nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.transformer = PEVideoTransformer(config.transformer)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        B, N, C, H, W = pixel_values_videos.shape
        backbone_output = self.clip_vision_model(pixel_values_videos.view(B * N, C, H, W)).logits.view(B, N, -1)
        projected = self.proj(F.normalize(backbone_output, dim=-1))
        return self.transformer(self.data_proj(projected), attention_mask=padding_mask_videos)


@dataclass
class PEVideoTextOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    video_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class PEVideoWithTextModel(PEVideoPretrainedModel):
    def __init__(self, config: PEVideoConfig):
        super().__init__(config)
        self.video_model = PEVideoModel(config.video_model)
        self.text_model = AutoModel.from_config(config.text_model)
        self.text_head = PEVideoContrastiveHead(config.text_model.hidden_size, config.output_dim)
        self.logit_scale = torch.nn.Parameter(torch.tensor([10.0]).log())
        self.logit_bias = torch.nn.Parameter(torch.tensor([-10.0]))

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=nth_layer is not None
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0])

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> PEVideoTextOutput:
        video_model_output = self.video_model(pixel_values_videos, padding_mask_videos)
        text_model_output = self._get_text_output(input_ids, attention_mask)

        text_embeds = self.video_text_head(text_model_output.pooler_output)
        video_embeds = self.video_head(video_model_output.pooler_output)

        loss = None
        if return_loss:
            logits_per_text = text_embeds @ video_embeds.t()
            logits_per_text = logits_per_text * self.logit_scale.exp()
            labels = torch.eye(text_embeds.size(0), device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.size(0)

        return PEVideoTextOutput(
            loss=loss,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_model_output,
            video_model_output=video_model_output,
        )


__all__ = ["PEVideoModel", "PEVideoWithTextModel", "PEVideoConfig", "PEVideoEncoderConfig"]
