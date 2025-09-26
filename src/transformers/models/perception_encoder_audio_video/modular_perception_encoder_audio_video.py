from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..perception_encoder_audio.configuration_perception_encoder_audio import PerceptionEncoderAudioTransformerConfig
from ..perception_encoder_audio.modeling_perception_encoder_audio import (
    PerceptionEncoderAudioContrastiveHead,
    PerceptionEncoderAudioTransformer,
)


class PerceptionEncoderAudioVideoTransformerConfig(PerceptionEncoderAudioTransformerConfig): ...


class PerceptionEncoderAudioVideoConfig(PretrainedConfig):
    sub_config = {"video_model": AutoConfig, "audio_model": AutoConfig, "text_model": AutoConfig}
    model_type = "perception_encoder_audio_video"

    def __init__(
        self,
        transformer: Optional[dict] = None,
        video_model: Optional[dict] = None,
        audio_model: Optional[dict] = None,
        text_model: Optional[dict] = None,
        output_dim: int = 1024,
        nth_text_layer: Optional[int] = 22,
        **kwargs,
    ):
        super().__init__(**kwargs)
        text_model = text_model or {}
        video_model = video_model or {}
        transformer = transformer or {}
        audio_model = audio_model or {}

        self.video_model = CONFIG_MAPPING[video_model.get("model_type", "perception_encoder_video")](**video_model)
        self.text_model = CONFIG_MAPPING[text_model.get("model_type", "modernbert")](**text_model)
        self.audio_model = CONFIG_MAPPING[audio_model.get("model_type", "perception_encoder_audio")](**audio_model)
        self.transformer = PerceptionEncoderAudioVideoTransformerConfig(**transformer)
        self.output_dim = output_dim
        self.nth_text_layer = nth_text_layer


class AlignModalities(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        btc: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.btc = btc
        self.conv = torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        if self.normalize:
            self.layer_norm = torch.nn.LayerNorm(self.out_channels)

    def get_sizes(self, seq, mask):
        if mask is not None:
            sizes = mask.sum(-1)
        else:
            sizes = torch.full((seq.size(0),), seq.size(-1), device=seq.device)
        if sizes.dim() > 1:
            sizes = sizes.squeeze(1)
        return sizes.long()

    def interpolate(self, tgt, tgt_sizes, src_sizes) -> torch.Tensor:
        result = torch.zeros(tgt.size(0), tgt.size(1), src_sizes.max(), device=tgt.device)
        for i, (tgt_row, tgt_size, src_size) in enumerate(zip(tgt, tgt_sizes, src_sizes)):
            tgt_row = tgt_row[:, :tgt_size]
            interpolated = F.interpolate(tgt_row[None], size=src_size, mode="nearest")
            result[i, :, :src_size] = interpolated[0]
        return result

    def forward(self, src, src_mask, tgt, tgt_mask):
        # BxTxC -> BxCxT
        src = src.transpose(1, 2)
        tgt = tgt.transpose(1, 2)

        tgt = self.conv(tgt)

        src_sizes = self.get_sizes(src, src_mask)
        tgt_sizes = self.get_sizes(tgt, tgt_mask)
        if all(src_sizes == tgt_sizes):
            upsampled = tgt
        else:
            upsampled = self.interpolate(tgt, tgt_sizes, src_sizes)

        upsampled = upsampled.permute(0, 2, 1)  # BxCxT -> BxTxC
        if self.normalize:
            upsampled = self.layer_norm(upsampled)
        return upsampled, src_mask


class PerceptionEncoderAudioVideoTransformer(PerceptionEncoderAudioTransformer): ...


class PerceptionEncoderAudioVideoContrastiveHead(PerceptionEncoderAudioContrastiveHead): ...


## Audio Video Encoder
class AudioVideoEncoder(PerceptionEncoderAudioVideoTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.modality_aligner = AlignModalities(
            self.config.hidden_size, self.config.hidden_size, normalize=True, btc=True
        )
        self.concat_modality_proj = torch.nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.data_proj = torch.nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        video_padding_mask: Optional[torch.Tensor] = None,
    ):
        video, video_padding_mask = self.modality_aligner(audio, audio_padding_mask, video, video_padding_mask)
        x = torch.cat([audio, video], dim=-1)
        x = self.concat_modality_proj(x)
        return super().forward(self.data_proj(x), padding_mask=video_padding_mask)


@dataclass
class AudioVideoModelOutput(BaseModelOutputWithPooling):
    audio_model_output: Optional[BaseModelOutputWithPooling] = None
    video_model_output: Optional[BaseModelOutputWithPooling] = None


class PerceptionEncoderAudioVideoPretrainedModel(PreTrainedModel):
    config: PerceptionEncoderAudioVideoConfig
    base_model_prefix = "perception_encoder_audio_video"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


class PerceptionEncoderAudioVideoModel(PerceptionEncoderAudioVideoPretrainedModel):
    def __init__(self, config: PerceptionEncoderAudioVideoConfig):
        super().__init__(config)
        # Synchronize _attn_implementations
        config.audio_model.transformer_attn_implementation = config._attn_implementation
        config.transformer.transformer_attn_implementation = config._attn_implementation
        config.transformer._attn_implementation = config._attn_implementation
        self.audio_model = AutoModel.from_config(config.audio_model)
        self.video_model = AutoModel.from_config(config.video_model)
        self.transformer = AudioVideoEncoder(config.transformer)

    def forward(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> AudioVideoModelOutput:
        audio_output = self.audio_model(input_values, padding_mask=padding_mask)
        video_output = self.video_model(pixel_values_videos, padding_mask_videos=padding_mask_videos)
        av_output = self.transformer(
            audio_output.last_hidden_state,
            video_output.last_hidden_state,
            audio_padding_mask=audio_output.audio_feature_padding_mask,
            video_padding_mask=padding_mask_videos,
        )
        return AudioVideoModelOutput(
            last_hidden_state=av_output.last_hidden_state,
            pooler_output=av_output.pooler_output,
            audio_model_output=audio_output,
            video_model_output=video_output,
        )


@dataclass
class PerceptionEncoderAudioVideoTextOutput(ModelOutput):
    audio_video_loss: Optional[torch.FloatTensor] = None
    text_audio_loss: Optional[torch.FloatTensor] = None
    text_audio_video_loss: Optional[torch.FloatTensor] = None
    text_video_loss: Optional[torch.FloatTensor] = None
    # embeddings
    audio_embeds: Optional[torch.FloatTensor] = None
    audio_video_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    audio_text_embeds: Optional[torch.FloatTensor] = None
    audio_video_text_embeds: Optional[torch.FloatTensor] = None
    video_text_embeds: Optional[torch.FloatTensor] = None
    # model outputs
    audio_model_output: Optional[BaseModelOutputWithPooling] = None
    audio_video_model_output: Optional[BaseModelOutputWithPooling] = None
    text_model_output: Optional[BaseModelOutputWithPooling] = None
    video_model_output: Optional[BaseModelOutputWithPooling] = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class PerceptionEncoderAudioVideoWithTextModel(PerceptionEncoderAudioVideoModel):
    def __init__(self, config: PerceptionEncoderAudioVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_model)
        self.audio_video_text_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.audio_text_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.video_text_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.text_model.hidden_size, config.output_dim
        )
        self.audio_video_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.transformer.hidden_size, config.output_dim
        )
        self.audio_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.audio_model.transformer.hidden_size, config.output_dim
        )
        self.video_head = PerceptionEncoderAudioVideoContrastiveHead(
            config.video_model.transformer.hidden_size, config.output_dim
        )
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

    def _maybe_compute_loss(
        self, embeds1: Optional[torch.Tensor], embeds2: Optional[torch.Tensor], return_loss: bool
    ) -> Optional[torch.Tensor]:
        if return_loss and embeds1 is not None and embeds2 is not None:
            logits = embeds1 @ embeds2.t()
            logits = logits * self.logit_scale + self.logit_bias
            labels = torch.eye(embeds1.size(0), device=embeds1.device)
            return -F.logsigmoid(labels * logits).sum() / embeds1.size(0)
        return None

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, padding_mask_videos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.video_head(self.video_model(pixel_values_videos, padding_mask_videos).pooler_output)

    def get_audio_features(
        self, input_values: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.audio_head(self.audio_model(input_values, padding_mask).pooler_output)

    def get_audio_video_features(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = super().forward(
            input_values=input_values,
            pixel_values_videos=pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )
        return self.audio_video_head(output.pooler_output)

    def get_audio_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.audio_text_head(self._get_text_output(input_ids, attention_mask).pooler_output)

    def get_video_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.video_text_head(self._get_text_output(input_ids, attention_mask).pooler_output)

    def get_audio_video_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.audio_video_text_head(self._get_text_output(input_ids, attention_mask).pooler_output)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> PerceptionEncoderAudioVideoTextOutput:
        # text embeddings
        audio_text_embeds = video_text_embeds = audio_video_text_embeds = None
        # media embeddings (audio, video, audio_video)
        audio_embeds = video_embeds = audio_video_embeds = None
        # model outputs
        audio_outputs = video_outputs = audio_video_outputs = text_outputs = None

        # Compute model outputs and embeddings for each modality
        if input_ids is not None:
            text_outputs = self._get_text_output(input_ids, attention_mask)
        if input_values is not None and pixel_values_videos is not None:
            # If we compute audio/video outputs, then extract the intermedia audio and video outputs
            audio_video_outputs = super().forward(
                input_values, pixel_values_videos, padding_mask=padding_mask, padding_mask_videos=padding_mask_videos
            )
            audio_outputs = audio_video_outputs.audio_model_output
            video_outputs = audio_video_outputs.video_model_output

            audio_embeds = self.audio_head(audio_outputs.pooler_output)
            video_embeds = self.video_head(video_outputs.pooler_output)
            audio_video_embeds = self.audio_video_head(audio_video_outputs.pooler_output)
            if text_outputs is not None:
                # Compute the corresponding text embeddings
                audio_text_embeds = self.audio_text_head(text_outputs.pooler_output)
                video_text_embeds = self.video_text_head(text_outputs.pooler_output)
                audio_video_text_embeds = self.audio_video_text_head(text_outputs.pooler_output)
        else:
            if pixel_values_videos is not None:
                video_outputs = self.video_model(pixel_values_videos, padding_mask_videos=padding_mask_videos)
                video_embeds = self.video_head(video_outputs.pooler_output)
                if text_outputs is not None:
                    video_text_embeds = self.video_text_head(text_outputs.pooler_output)
            if input_values is not None:
                audio_outputs = self.audio_model(input_values, padding_mask=padding_mask)
                if text_outputs is not None:
                    audio_text_embeds = self.audio_text_head(text_outputs.pooler_output)

        return PerceptionEncoderAudioVideoTextOutput(
            audio_video_loss=self._maybe_compute_loss(audio_embeds, video_embeds, return_loss),
            text_audio_loss=self._maybe_compute_loss(audio_text_embeds, audio_embeds, return_loss),
            text_audio_video_loss=self._maybe_compute_loss(audio_video_text_embeds, audio_video_embeds, return_loss),
            text_video_loss=self._maybe_compute_loss(video_text_embeds, video_embeds, return_loss),
            audio_embeds=audio_embeds,
            audio_video_embeds=audio_video_embeds,
            video_embeds=video_embeds,
            audio_text_embeds=audio_text_embeds,
            audio_video_text_embeds=audio_video_text_embeds,
            video_text_embeds=video_text_embeds,
            audio_model_output=audio_outputs,
            audio_video_model_output=audio_video_outputs,
            text_model_output=text_outputs,
            video_model_output=video_outputs,
        )


__all__ = [
    "PerceptionEncoderAudioVideoModel",
    "PerceptionEncoderAudioVideoWithTextModel",
    "PerceptionEncoderAudioVideoTextOutput",
    "PerceptionEncoderAudioVideoConfig",
]
