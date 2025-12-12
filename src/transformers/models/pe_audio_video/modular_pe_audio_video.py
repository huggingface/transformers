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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, eager_attention_forward
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from .configuration_pe_audio_video import PeAudioVideoConfig, PeAudioVideoEncoderConfig


class PeAudioVideoMaskedGroupNorm(nn.GroupNorm):
    def forward(self, x, padding_mask=None):
        if padding_mask is None:
            return super().forward(x)

        batch_size, hidden_size, seq_len = x.shape
        group_size = hidden_size // self.num_groups
        grouped_shape = (batch_size, -1, group_size, seq_len)

        x_grouped = x.view(grouped_shape)
        padding_mask_grouped = padding_mask.reshape(grouped_shape).bool()

        mean = torch.masked.mean(x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True)
        var = torch.masked.var(x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True, unbiased=False)

        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(x.shape)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)

        return x_norm * padding_mask


class PeAudioVideoConvBlock1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groupnorm = PeAudioVideoMaskedGroupNorm(num_groups=1, num_channels=config.hidden_size)
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x, padding_mask=None):
        x = self.groupnorm(x, padding_mask=padding_mask)
        x = self.activation(x)
        return self.project(x)


class PeAudioVideoResnetBlock1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = PeAudioVideoConvBlock1d(config)
        self.block2 = PeAudioVideoConvBlock1d(config)

    def forward(self, hidden_states, padding_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            padding_mask: (batch_size, seq_len)
        Returns:
            hidden_states: (batch_size, seq_len, hidden_size)
        """
        # transpose for convolutions
        # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len)
        hidden_states = hidden_states.transpose(1, 2)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).expand_as(hidden_states)

        residual = hidden_states
        hidden_states = self.block1(hidden_states, padding_mask=padding_mask)
        hidden_states = self.block2(hidden_states, padding_mask=padding_mask)
        hidden_states = residual + hidden_states

        return hidden_states.transpose(1, 2)


class PeAudioVideoEncoderPatchEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet_block = PeAudioVideoResnetBlock1d(config)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, inputs_embeds, padding_mask=None):
        # Embedding step: prepend class token and run the ResNet block.
        hidden_states = torch.cat(
            [self.class_embedding.expand(inputs_embeds.size(0), -1, -1), inputs_embeds],
            dim=1,
        )

        if padding_mask is not None:
            # TODO: any reason why we take padding_mask[0] and not just 1?
            padding_mask = torch.cat([padding_mask[:, [0]], padding_mask], dim=1)

        hidden_states = self.resnet_block(hidden_states, padding_mask=padding_mask)
        return hidden_states, padding_mask


class PeAudioVideoContrastiveHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.layer_norm(x))


class PeAudioVideoEncoderEmbedder(nn.Module):
    def __init__(self, config: PeAudioVideoEncoderConfig):
        super().__init__()
        self.audio_encoder = AutoModel.from_config(config.audio_config)
        self.video_encoder = AutoModel.from_config(config.video_config)

        self.video_proj = nn.Conv1d(config.video_config.hidden_size, config.audio_config.hidden_size, 1)
        self.video_norm = nn.LayerNorm(config.audio_config.hidden_size)

        self.concat_modality_proj = nn.Linear(
            config.audio_config.hidden_size + config.video_config.hidden_size,
            config.hidden_size,
        )
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def _align_video_hidden_state(
        self,
        video_hidden_state: torch.Tensor,
        audio_hidden_state: torch.Tensor,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Align video_hidden_state to audio_hidden_state by nearest neighbor interpolation.
        """
        if video_hidden_state.shape[1] == audio_hidden_state.shape[1]:
            return video_hidden_state

        if padding_mask_videos is not None:
            video_lengths = padding_mask_videos.sum(dim=-1)
        else:
            video_lengths = video_hidden_state.shape[1] * video_hidden_state.new_ones(
                video_hidden_state.shape[0], dtype=torch.long
            )

        if padding_mask is not None:
            audio_lengths = padding_mask.sum(dim=-1)
        else:
            audio_lengths = audio_hidden_state.shape[1] * audio_hidden_state.new_ones(
                audio_hidden_state.shape[0], dtype=torch.long
            )

        if (audio_lengths == video_hidden_state.shape[1]).all() or (
            video_lengths == audio_hidden_state.shape[1]
        ).all():
            # no need to align taking into account the padding masks
            # note: when one of the above is true, we can expect the other to be true as there is no reason
            # to have masked audio without masked video and vice versa

            return F.interpolate(video_hidden_state, size=audio_hidden_state.shape[1], mode="nearest")

        aligned_shape = (*audio_hidden_state.shape[:2], video_hidden_state.shape[-1])
        aligned_hidden_state = audio_hidden_state.new_zeros(aligned_shape)

        for i, (hidden_state, video_length, audio_length) in enumerate(
            zip(video_hidden_state, video_lengths, audio_lengths)
        ):
            hidden_state = hidden_state[:video_length]
            if hidden_state.numel() > 0 and audio_length > 0:
                interpolated_hidden_state = F.interpolate(
                    hidden_state[None].transpose(1, 2), size=audio_length, mode="nearest"
                ).transpose(1, 2)[0]
                aligned_hidden_state[i, :audio_length, :] = interpolated_hidden_state

        return aligned_hidden_state

    def forward(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ):
        audio_output = self.audio_encoder(input_values, padding_mask=padding_mask)
        video_output = self.video_encoder(pixel_values_videos, padding_mask_videos=padding_mask_videos)

        audio_hidden_state = audio_output.last_hidden_state
        video_hidden_state = video_output.last_hidden_state
        padding_mask = audio_output.output_mask

        video_hidden_state = self.video_proj(video_hidden_state.transpose(1, 2)).transpose(1, 2)
        video_hidden_state = self._align_video_hidden_state(
            video_hidden_state=video_hidden_state,
            audio_hidden_state=audio_hidden_state,
            padding_mask_videos=padding_mask_videos,
            padding_mask=padding_mask,
        )
        video_hidden_state = self.video_norm(video_hidden_state)
        inputs_embeds = torch.cat([audio_hidden_state, video_hidden_state], dim=-1)
        inputs_embeds = self.concat_modality_proj(inputs_embeds)

        return inputs_embeds, padding_mask, audio_output, video_output


class PeAudioVideoEncoderAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False
        del self.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PeAudioVideoEncoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class PeAudioVideoEncoderRMSNorm(Qwen3RMSNorm): ...


def stack_freqs(cos: torch.Tensor, sin: torch.Tensor):
    dim = cos.size(-1)
    cos = cos.narrow(-1, 0, dim // 2)
    sin = sin.narrow(-1, 0, dim // 2)
    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*cos.size(), 2, 2)
    return freqs_cis


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    freqs_cis = stack_freqs(cos, sin)
    freqs_cis = freqs_cis.unsqueeze(unsqueeze_dim)
    q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q_ * freqs_cis).sum(5).flatten(3), (k_ * freqs_cis).sum(5).flatten(3)


class PeAudioVideoEncoderRotaryEmbedding(Qwen3RotaryEmbedding): ...


@auto_docstring
class PeAudioVideoPretrainedModel(PreTrainedModel):
    config: PeAudioVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PeAudioVideoEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PeAudioVideoEncoderLayer,
        "attentions": PeAudioVideoEncoderAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, PeAudioVideoEncoderPatchEmbedder):
            embed_dim = module.class_embedding.shape[-1]
            nn.init.normal_(module.class_embedding, mean=0.0, std=embed_dim**-0.5 * std)


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PeAudioVideoEncoder`].
    """
)
class PeAudioVideoEncoderOutput(BaseModelOutputWithPooling):
    audio_model_output: Optional[BaseModelOutputWithPooling] = None
    video_model_output: Optional[BaseModelOutputWithPooling] = None


@auto_docstring(
    custom_intro="""
    The PeAudioVideo Encoder model.
    """
)
class PeAudioVideoEncoder(PeAudioVideoPretrainedModel):
    config: PeAudioVideoEncoderConfig
    main_input_name = "input_values"
    base_model_prefix = "audio_video_encoder"

    def __init__(self, config: PeAudioVideoEncoderConfig):
        super().__init__(config)
        self.embedder = PeAudioVideoEncoderEmbedder(config)
        self.patch_embedder = PeAudioVideoEncoderPatchEmbedder(config)
        self.layers = nn.ModuleList(
            [PeAudioVideoEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PeAudioVideoEncoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PeAudioVideoEncoderRotaryEmbedding(config=config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.gradient_checkpointing = False

        self.post_init()

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PeAudioVideoEncoderOutput:
        inputs_embeds, padding_mask, audio_output, video_output = self.embedder(
            input_values,
            pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )
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

        return PeAudioVideoEncoderOutput(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
            audio_model_output=audio_output,
            video_model_output=video_output,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PeAudioVideoModel`] when using text, audio, and/or video.
    """
)
class PeAudioVideoOutput(ModelOutput):
    # embeddings
    audio_embeds: Optional[torch.FloatTensor] = None
    audio_video_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    text_audio_embeds: Optional[torch.FloatTensor] = None
    text_video_embeds: Optional[torch.FloatTensor] = None
    text_audio_video_embeds: Optional[torch.FloatTensor] = None

    # model outputs
    # TODO: update types to the correct ones
    audio_video_model_output: Optional[BaseModelOutputWithPooling] = None
    text_model_output: Optional[BaseModelOutputWithPooling] = None

    # logits
    logits_audio_text: Optional[torch.FloatTensor] = None
    logits_video_text: Optional[torch.FloatTensor] = None
    logits_audio_video_text: Optional[torch.FloatTensor] = None
    logits_audio_video: Optional[torch.FloatTensor] = None

    loss: Optional[torch.FloatTensor] = None
    audio_video_loss: Optional[torch.FloatTensor] = None
    text_audio_loss: Optional[torch.FloatTensor] = None
    text_video_loss: Optional[torch.FloatTensor] = None
    text_audio_video_loss: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class PeAudioVideoModel(PeAudioVideoPretrainedModel):
    def __init__(self, config: PeAudioVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_video_encoder = PeAudioVideoEncoder(config.audio_video_config)

        text_hidden_size = config.text_config.hidden_size
        audio_hidden_size = config.audio_video_config.audio_config.hidden_size
        video_hidden_size = config.audio_video_config.video_config.hidden_size

        # audio
        self.audio_head = PeAudioVideoContrastiveHead(audio_hidden_size, text_hidden_size)
        self.text_audio_head = PeAudioVideoContrastiveHead(text_hidden_size, text_hidden_size)
        self.audio_logit_scale = nn.Parameter(torch.zeros(1))
        self.audio_logit_bias = nn.Parameter(torch.zeros(1))
        self.text_audio_logit_scale = nn.Parameter(torch.zeros(1))
        self.text_audio_logit_bias = nn.Parameter(torch.zeros(1))

        # video
        self.video_head = PeAudioVideoContrastiveHead(video_hidden_size, text_hidden_size)
        self.text_video_head = PeAudioVideoContrastiveHead(text_hidden_size, text_hidden_size)
        self.video_logit_scale = nn.Parameter(torch.zeros(1))
        self.video_logit_bias = nn.Parameter(torch.zeros(1))
        self.text_video_logit_scale = nn.Parameter(torch.zeros(1))
        self.text_video_logit_bias = nn.Parameter(torch.zeros(1))

        # audio-video
        self.audio_video_head = PeAudioVideoContrastiveHead(config.audio_video_config.hidden_size, text_hidden_size)
        self.text_audio_video_head = PeAudioVideoContrastiveHead(text_hidden_size, text_hidden_size)
        self.audio_video_logit_scale = nn.Parameter(torch.zeros(1))
        self.audio_video_logit_bias = nn.Parameter(torch.zeros(1))
        self.text_audio_video_logit_scale = nn.Parameter(torch.zeros(1))
        self.text_audio_video_logit_bias = nn.Parameter(torch.zeros(1))

        # text-audio
        self.audio_plus_text_head = PeAudioVideoContrastiveHead(audio_hidden_size + text_hidden_size, text_hidden_size)

        # text-video
        self.video_plus_text_head = PeAudioVideoContrastiveHead(video_hidden_size + text_hidden_size, text_hidden_size)

        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss=False,
        **kwargs,
    ) -> PeAudioVideoOutput:
        audio_video_outputs = self.audio_video_encoder(
            input_values=input_values,
            pixel_values_videos=pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
            **{**kwargs, "return_dict": True},
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )

        audio_video_embeds = audio_video_outputs.pooler_output
        audio_video_embeds = self.audio_video_head(audio_video_embeds)

        audio_embeds = audio_video_outputs.audio_model_output.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        video_embeds = audio_video_outputs.video_model_output.pooler_output
        video_embeds = self.video_head(video_embeds)

        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_audio_embeds = self.text_audio_head(text_embeds)
        text_video_embeds = self.text_video_head(text_embeds)
        text_audio_video_embeds = self.text_audio_video_head(text_embeds)

        logits_audio_video = audio_video_embeds @ text_embeds.T
        logits_audio_video = logits_audio_video * self.audio_video_logit_scale + self.audio_video_logit_bias

        logits_audio_text = audio_embeds @ text_audio_embeds.T
        logits_audio_text = logits_audio_text * self.text_audio_logit_scale + self.text_audio_logit_bias

        logits_video_text = video_embeds @ text_video_embeds.T
        logits_video_text = logits_video_text * self.text_video_logit_scale + self.text_video_logit_bias

        logits_audio_video_text = audio_video_embeds @ text_audio_video_embeds.T
        logits_audio_video_text = (
            logits_audio_video_text * self.text_audio_video_logit_scale + self.text_audio_video_logit_bias
        )

        loss, audio_video_loss, audio_text_loss, video_text_loss, audio_video_text_loss = None, None, None, None, None
        if return_loss:
            audio_video_labels = torch.eye(audio_video_embeds.shape[0], device=audio_video_embeds.device)
            audio_text_labels = torch.eye(audio_embeds.shape[0], device=audio_embeds.device)
            video_text_labels = torch.eye(video_embeds.shape[0], device=video_embeds.device)
            audio_video_text_labels = torch.eye(audio_video_embeds.shape[0], device=audio_video_embeds.device)

            audio_video_loss = (
                -F.logsigmoid(audio_video_labels * logits_audio_video).sum() / audio_video_embeds.shape[0]
            )
            audio_text_loss = -F.logsigmoid(audio_text_labels * logits_audio_text).sum() / audio_embeds.shape[0]
            video_text_loss = -F.logsigmoid(video_text_labels * logits_video_text).sum() / video_embeds.shape[0]
            audio_video_text_loss = (
                -F.logsigmoid(audio_video_text_labels * logits_audio_video_text).sum() / audio_video_embeds.shape[0]
            )

            loss = audio_video_loss + audio_text_loss + video_text_loss + audio_video_text_loss

        return PeAudioVideoOutput(
            logits_audio_video=logits_audio_video,
            logits_audio_text=logits_audio_text,
            logits_video_text=logits_video_text,
            logits_audio_video_text=logits_audio_video_text,
            audio_embeds=audio_embeds,
            video_embeds=video_embeds,
            text_audio_embeds=text_audio_embeds,
            text_video_embeds=text_video_embeds,
            text_audio_video_embeds=text_audio_video_embeds,
            text_model_output=text_outputs,
            audio_video_model_output=audio_video_outputs,
            loss=loss,
            audio_video_loss=audio_video_loss,
            text_audio_loss=audio_text_loss,
            text_video_loss=video_text_loss,
            text_audio_video_loss=audio_video_text_loss,
        )

    def forward_text_audio(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> PeAudioVideoOutput:
        # Audio encoding
        audio_outputs = self.audio_video_encoder.embedder.audio_encoder(input_values, padding_mask=padding_mask)
        audio_embeds = audio_outputs.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        # Text encoding
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )
        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_audio_head(text_embeds)

        # Compute logits
        logits_per_audio = audio_embeds @ text_embeds.T
        logits_per_audio = logits_per_audio * self.audio_logit_scale + self.audio_logit_bias

        # Compute loss
        loss = None
        if return_loss:
            labels = torch.eye(audio_embeds.shape[0], device=audio_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_audio).sum() / audio_embeds.shape[0]

        return PeAudioVideoOutput(
            logits_per_text=logits_per_audio.t(),
            logits_per_audio_text=logits_per_audio,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            loss=loss,
        )

    def forward_text_video(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> PeAudioVideoOutput:
        # Video encoding
        video_outputs = self.audio_video_encoder.embedder.video_encoder(
            pixel_values_videos, padding_mask_videos=padding_mask_videos
        )
        video_embeds = video_outputs.pooler_output
        video_embeds = self.video_head(video_embeds)

        # Text encoding
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{**kwargs, "return_dict": True},
            output_hidden_states=True,
        )
        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_video_head(text_embeds)

        # Compute logits
        logits_per_video = video_embeds @ text_embeds.T
        logits_per_video = logits_per_video * self.video_logit_scale + self.video_logit_bias

        # Compute loss
        loss = None
        if return_loss:
            labels = torch.eye(video_embeds.shape[0], device=video_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_video).sum() / video_embeds.shape[0]

        return PeAudioVideoOutput(
            logits_per_text=logits_per_video.t(),
            logits_per_video_text=logits_per_video,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_outputs,
            video_model_output=video_outputs,
            loss=loss,
        )

    def forward_audio_video(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs,
    ) -> PeAudioVideoOutput:
        # Audio encoding
        audio_outputs = self.audio_video_encoder.embedder.audio_encoder(input_values, padding_mask=padding_mask)
        audio_embeds = audio_outputs.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        # Video encoding
        video_outputs = self.audio_video_encoder.embedder.video_encoder(
            pixel_values_videos, padding_mask_videos=padding_mask_videos
        )
        video_embeds = video_outputs.pooler_output
        video_embeds = self.video_head(video_embeds)

        # Compute logits
        logits_audio_video = audio_embeds @ video_embeds.T
        logits_audio_video = logits_audio_video * self.audio_video_logit_scale + self.audio_video_logit_bias

        # Compute loss
        loss = None
        if return_loss:
            labels = torch.eye(audio_embeds.shape[0], device=audio_embeds.device)
            loss = -F.logsigmoid(labels * logits_audio_video).sum() / audio_embeds.shape[0]

        return PeAudioVideoOutput(
            logits_per_audio_video=logits_audio_video,
            audio_embeds=audio_embeds,
            video_embeds=video_embeds,
            audio_model_output=audio_outputs,
            video_model_output=video_outputs,
            loss=loss,
        )


__all__ = [
    "PeAudioVideoModel",
    "PeAudioVideoEncoder",
]
