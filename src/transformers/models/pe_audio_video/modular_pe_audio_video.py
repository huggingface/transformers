from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..clip.modeling_clip import CLIPOutput
from ..pe_audio.modeling_pe_audio import (
    PEAudioContrastiveHead,
    PEAudioEncoderEmbeddings,
)
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding


class PEAudioVideoEncoderConfig(Qwen3Config):
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        video_config=None,
        hidden_size=1792,
        intermediate_size=4800,
        num_hidden_layers=6,
        num_attention_heads=14,
        num_key_value_heads=None,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_parameters={
            "rope_theta": 20000,
        },
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        layer_types=None,
        tie_word_embeddings=False,
        vocab_size=None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            vocab_size=vocab_size,
            layer_types=layer_types,
            tie_word_embeddings=tie_word_embeddings,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            **kwargs,
        )

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "pe_audio_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["pe_audio_encoder"]()
            # TODO: add log

        self.audio_config = audio_config

        if isinstance(video_config, dict):
            video_config["model_type"] = video_config.get("model_type", "pe_video_encoder")
            video_config = CONFIG_MAPPING[video_config["model_type"]](**video_config)
        elif video_config is None:
            video_config = CONFIG_MAPPING["pe_video_encoder"]()
            # TODO: add log

        self.video_config = video_config


class PEAudioVideoConfig(PretrainedConfig):
    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_video_config": PEAudioVideoEncoderConfig}

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
    }

    def __init__(
        self,
        text_config=None,
        audio_video_config=None,
        projection_dim=1024,
        nth_text_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](
                **self._default_text_config_kwargs
            )

        if isinstance(audio_video_config, dict):
            audio_video_config = PEAudioVideoEncoderConfig(**audio_video_config)
        elif audio_video_config is None:
            audio_video_config = PEAudioVideoEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.audio_video_config = audio_video_config

        self.projection_dim = projection_dim


class PEAudioVideoContrastiveHead(PEAudioContrastiveHead): ...


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PEAudioVideoEncoder`].
    """
)
class PEAudioVideoEncoderOutput(BaseModelOutputWithPooling):
    audio_features: Optional[torch.FloatTensor] = None
    video_features: Optional[torch.FloatTensor] = None
    outputs_mask: Optional[tuple[torch.FloatTensor]] = None
    audio_model_output: Optional[BaseModelOutputWithPooling] = None
    video_model_output: Optional[BaseModelOutputWithPooling] = None


class PEAudioVideoEncoderEmbeddings(PEAudioEncoderEmbeddings): ...


class PEAudioVideoAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None


class PEAudioVideoEncoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class PEAudioVideoRMSNorm(Qwen3RMSNorm): ...


class PEAudioVideoRotaryEmbedding(Qwen3RotaryEmbedding): ...

@auto_docstring
class PEAudioVideoPretrainedModel(PreTrainedModel):
    config: PEAudioVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PEAudioVideoEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PEAudioVideoEncoderLayer,
        "attentions": PEAudioVideoAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, PEAudioVideoEncoderEmbeddings):
            embed_dim = module.class_embedding.shape[-1]
            nn.init.normal_(module.class_embedding, mean=0.0, std=embed_dim**-0.5 * std)


@auto_docstring(
    custom_intro="""
    The PEAudioVideo Encoder model.
    """
)
class PEAudioVideoEncoder(PEAudioVideoPretrainedModel):
    config: PEAudioVideoEncoderConfig
    main_input_name = "input_values"
    base_model_prefix = "audio_video_encoder"

    def __init__(self, config: PEAudioVideoEncoderConfig):
        super().__init__(config)
        self.audio_encoder = AutoModel.from_config(config.audio_config)
        self.video_encoder = AutoModel.from_config(config.video_config)

        self.video_proj = nn.Conv1d(config.video_config.hidden_size, config.audio_config.hidden_size, 1)
        self.video_norm = nn.LayerNorm(config.audio_config.hidden_size)

        self.concat_modality_proj = nn.Linear(
            config.audio_config.hidden_size + config.video_config.hidden_size,
            config.hidden_size,
        )
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.embeddings = PEAudioVideoEncoderEmbeddings(config)
        self.layers = nn.ModuleList(
            [PEAudioVideoEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PEAudioVideoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PEAudioVideoRotaryEmbedding(config=config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.post_init()

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

        video_lengths = (
            padding_mask_videos.sum(dim=-1)
            if padding_mask_videos is not None
            else video_hidden_state.shape[1] * video_hidden_state.new_ones(video_hidden_state.shape[0], dtype=torch.long)
        )
        audio_lengths = (
            padding_mask.sum(dim=-1)
            if padding_mask is not None
            else audio_hidden_state.shape[1] * audio_hidden_state.new_ones(audio_hidden_state.shape[0], dtype=torch.long)
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
                    hidden_state[None].transpose(1, 2),
                    size=audio_length,
                    mode="nearest"
                ).transpose(1, 2)[0]
                aligned_hidden_state[i, :audio_length, :] = interpolated_hidden_state

        return aligned_hidden_state

    def get_audio_video_features(
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

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PEAudioVideoEncoderOutput:
        inputs_embeds, padding_mask, audio_output, video_output = self.get_audio_video_features(
            input_values,
            pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )

        inputs_embeds, attention_mask = self.embeddings(inputs_embeds, padding_mask=padding_mask)

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

        return PEAudioVideoEncoderOutput(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
            audio_model_output=audio_output,
            video_model_output=video_output,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PEAudioVideoModel`] when using text, audio, and/or video.
    """
)
class PEAudioVideoOutput(ModelOutput):
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


class PEAudioVideoModel(PEAudioVideoPretrainedModel):
    def __init__(self, config: PEAudioVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_video_encoder = PEAudioVideoEncoder(config.audio_video_config)

        # audio
        self.audio_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.audio_config.hidden_size, config.projection_dim
        )
        self.text_audio_head = PEAudioVideoContrastiveHead(
            config.text_config.hidden_size, config.projection_dim
        )
        self.audio_logit_scale = nn.Parameter(torch.zeros((1)))
        self.audio_logit_bias = nn.Parameter(torch.zeros((1)))
        self.text_audio_logit_scale = nn.Parameter(torch.zeros((1)))
        self.text_audio_logit_bias = nn.Parameter(torch.zeros((1)))

        # video
        self.video_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.video_config.hidden_size, config.projection_dim
        ) 
        self.text_video_head = PEAudioVideoContrastiveHead(
            config.text_config.hidden_size, config.projection_dim
        )
        self.video_logit_scale = nn.Parameter(torch.zeros((1)))
        self.video_logit_bias = nn.Parameter(torch.zeros((1)))
        self.text_video_logit_scale = nn.Parameter(torch.zeros((1)))
        self.text_video_logit_bias = nn.Parameter(torch.zeros((1)))

        # audio-video
        self.audio_video_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.hidden_size, config.projection_dim
        )
        self.text_audio_video_head = PEAudioVideoContrastiveHead(
            config.text_config.hidden_size, config.projection_dim
        )
        self.audio_video_logit_scale = nn.Parameter(torch.zeros((1)))
        self.audio_video_logit_bias = nn.Parameter(torch.zeros((1)))
        self.text_audio_video_logit_scale = nn.Parameter(torch.zeros((1)))
        self.text_audio_video_logit_bias = nn.Parameter(torch.zeros((1)))

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
    ) -> PEAudioVideoOutput:
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
        logits_audio_video_text = logits_audio_video_text * self.text_audio_video_logit_scale + self.text_audio_video_logit_bias

        loss, audio_video_loss, audio_text_loss, video_text_loss, audio_video_text_loss = None, None, None, None, None
        if return_loss:
            audio_video_labels = torch.eye(audio_video_embeds.shape[0], device=audio_video_embeds.device)
            audio_text_labels = torch.eye(audio_embeds.shape[0], device=audio_embeds.device)
            video_text_labels = torch.eye(video_embeds.shape[0], device=video_embeds.device)
            audio_video_text_labels = torch.eye(audio_video_embeds.shape[0], device=audio_video_embeds.device)

            audio_video_loss = -F.logsigmoid(audio_video_labels * logits_audio_video).sum() / audio_video_embeds.shape[0]
            audio_text_loss = -F.logsigmoid(audio_text_labels * logits_audio_text).sum() / audio_embeds.shape[0]
            video_text_loss = -F.logsigmoid(video_text_labels * logits_video_text).sum() / video_embeds.shape[0]
            audio_video_text_loss = -F.logsigmoid(audio_video_text_labels * logits_audio_video_text).sum() / audio_video_embeds.shape[0]

            loss = audio_video_loss + audio_text_loss + video_text_loss + audio_video_text_loss

        return PEAudioVideoOutput(
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
    ) -> PEAudioVideoOutput:
        # Audio encoding
        audio_outputs = self.audio_video_encoder.audio_encoder(
            input_values,
            padding_mask=padding_mask
        )
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

        return PEAudioVideoOutput(
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
    ) -> PEAudioVideoOutput:
        # Video encoding
        video_outputs = self.audio_video_encoder.video_encoder(
            pixel_values_videos,
            padding_mask_videos=padding_mask_videos
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

        return PEAudioVideoOutput(
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
    ) -> PEAudioVideoOutput:
        # Audio encoding
        audio_outputs = self.audio_video_encoder.audio_encoder(
            input_values,
            padding_mask=padding_mask
        )
        audio_embeds = audio_outputs.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        # Video encoding
        video_outputs = self.audio_video_encoder.video_encoder(
            pixel_values_videos,
            padding_mask_videos=padding_mask_videos
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

        return PEAudioVideoOutput(
            logits_per_audio_video=logits_audio_video,
            audio_embeds=audio_embeds,
            video_embeds=video_embeds,
            audio_model_output=audio_outputs,
            video_model_output=video_outputs,
            loss=loss,
        )


__all__ = [
    "PEAudioVideoModel",
    "PEAudioVideoEncoder",
    "PEAudioVideoEncoderConfig",
    "PEAudioVideoConfig",
]
