from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
from ...modeling_outputs import BaseModelOutputWithPooling, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring, can_return_tuple, TransformersKwargs
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..dac.modeling_dac import DacEncoder
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from .configuration_pe_audio import PeAudioConfig, PeAudioEncoderConfig


class PeAudioEncoderConfig(Qwen3Config):
    _default_dac_config_kwargs = {
        "downsampling_ratios": [2, 8, 10, 12],
        "encoder_hidden_size": 64,
        "codebook_dim": 128,
    }
    sub_configs = {"dac_config": AutoConfig}

    def __init__(
        self,
        dac_config=None,
        hidden_size=1792,
        intermediate_size=4800,
        num_hidden_layers=28,
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
        if isinstance(dac_config, dict):
            dac_config['model_type'] = dac_config.get('model_type', 'dac')
            dac_config = CONFIG_MAPPING[dac_config['model_type']](
                **{**self._default_dac_config_kwargs, **dac_config}
            )
        elif dac_config is None:
            dac_config = CONFIG_MAPPING['dac'](
                **self._default_dac_config_kwargs
            )

        self.dac_config = dac_config

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


class PeAudioConfig(PretrainedConfig):
    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_config": PeAudioEncoderConfig}

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    def __init__(
        self,
        text_config=None,
        audio_config=None,
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

        if isinstance(audio_config, dict):
            audio_config = PeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = PeAudioEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.audio_config = audio_config


# TODO: not sure about the typing for text_model_output
@dataclass
# @auto_docstring
class PeAudioOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_audio (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-image
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`PeAudioTextModel`].
    audio_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The image embeddings obtained by applying the projection layer to the pooled output of [`PeAudioVisionModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PeAudioTextModel`].
    audio_model_output (`BaseModelOutputWithPooling`):
        The output of the [`PeAudioVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_audio: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    audio_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class PeAudioMaskedGroupNorm(nn.GroupNorm):
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


class PeAudioConvBlock1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.groupnorm = PeAudioMaskedGroupNorm(num_groups=1, num_channels=config.hidden_size)
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


class PeAudioResnetBlock1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = PeAudioConvBlock1d(config)
        self.block2 = PeAudioConvBlock1d(config)

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


class PeAudioDacEncoder(DacEncoder): ...


class PeAudioContrastiveHead(nn.Module):
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


class PeAudioEncoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet_block = PeAudioResnetBlock1d(config)
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


class PeAudioAttention(Qwen3Attention):
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


class PeAudioEncoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class PeAudioRMSNorm(Qwen3RMSNorm): ...


class PeAudioRotaryEmbedding(Qwen3RotaryEmbedding): ...


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


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PeAudioEncoder`].
    """
)
class PeAudioEncoderOutput(BaseModelOutputWithPooling):
    codec_features: Optional[torch.FloatTensor] = None
    output_mask: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring
class PeAudioPretrainedModel(PreTrainedModel):
    config: PeAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PeAudioEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PeAudioEncoderLayer,
        "attentions": PeAudioAttention,
    }
    _checkpoint_conversion_mapping = {
        r"^audio_video_encoder\.audio_encoder": "audio_encoder",
    }

    def _init_weights(self, module):
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, PeAudioEncoderEmbeddings):
            embed_dim = module.class_embedding.shape[-1]
            nn.init.normal_(module.class_embedding, mean=0.0, std=embed_dim**-0.5 * std)


@auto_docstring(
    custom_intro="""
    The PeAudio Encoder model.
    """
)
class PeAudioEncoder(PeAudioPretrainedModel):
    config: PeAudioEncoderConfig
    main_input_name = "input_values"
    base_model_prefix = "audio_encoder"

    def __init__(self, config: PeAudioEncoderConfig):
        super().__init__(config)
        self.dac_encoder = PeAudioDacEncoder(config.dac_config)
        self.bottleneck = nn.Conv1d(
            config.dac_config.hidden_size,
            config.dac_config.codebook_dim * 2,
            1,
        )
        self.data_proj = nn.Linear(config.dac_config.codebook_dim, config.hidden_size)

        # TODO: should it be named patch_embedding?
        self.embeddings =  PeAudioEncoderEmbeddings(config)
        self.layers = nn.ModuleList(
            [PeAudioEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PeAudioRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PeAudioRotaryEmbedding(config=config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.post_init()

    def get_audio_features(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-encoding step
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            hidden_states = self.dac_encoder(input_values)  # (batch_size, hidden_size, seq_len)
            hidden_states = self.bottleneck(hidden_states)  # (batch_size, hidden_size, seq_len)
            # TODO: we might actually be able to remove half the channels
            hidden_states, _ = hidden_states.chunk(2, dim=1)

        codec_features = hidden_states.transpose(1, 2)

        if padding_mask is not None:
            padding_mask = padding_mask[:, :: self.config.dac_config.hop_length]

        # Project codec features
        inputs_embeds = self.data_proj(codec_features)

        return inputs_embeds, padding_mask

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, padding_mask = self.get_audio_features(
            input_values,
            padding_mask=padding_mask,
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

        return PeAudioEncoderOutput(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
            output_mask=padding_mask,
        )


class PeAudioModel(PeAudioPretrainedModel):
    def __init__(self, config: PeAudioConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = PeAudioEncoder(config.audio_config)

        self.text_audio_head = PeAudioContrastiveHead(config.text_config.hidden_size, config.text_config.hidden_size)
        self.audio_head = PeAudioContrastiveHead(config.audio_config.hidden_size, config.text_config.hidden_size)

        self.audio_logit_scale = nn.Parameter(torch.zeros(1))
        self.audio_logit_bias = nn.Parameter(torch.zeros(1))

        self.post_init()

    def get_text_features(self, input_ids, attention_mask=None):
        # TODO: should it be named feature or embeds
        text_outputs: MaskedLMOutput = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_features = text_outputs.last_hidden_state
        text_features = self.text_head(text_features)
        return text_features

    def _get_audio_embeds(self, audio_outputs: BaseModelOutputWithPooling):
        return self.audio_head(audio_outputs.pooler_output)

    def get_audio_features(self, input_values, padding_mask=None):
        # TODO: should it be named feature or embeds
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            return_dict=True,
        )
        audio_features = self._get_audio_embeds(audio_outputs)
        return audio_features

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        **kwargs,
    ) -> PeAudioOutput:
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

        audio_embeds = audio_outputs.pooler_output
        audio_embeds = self.audio_head(audio_embeds)

        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_audio_head(text_embeds)

        logits_per_audio = audio_embeds @ text_embeds.T
        logits_per_audio = logits_per_audio * self.audio_logit_scale + self.audio_logit_bias
        logits_per_text = logits_per_audio.t()

        loss = None
        if return_loss:
            labels = torch.eye(text_embeds.shape[0], device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.shape[0]

        return PeAudioOutput(
            logits_per_text=logits_per_text,
            logits_per_audio=logits_per_audio,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            loss=loss,
        )


# TODO: underline in documentation that logits output shape is 
# 1. Model: (n_audio, n_text)
# 2. Frame-level: (n_audio, n_text, n_frames)
class PeAudioFrameLevelModel(PeAudioModel):
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        **kwargs,
    ) -> PeAudioOutput:
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

        audio_embeds = audio_outputs.last_hidden_state
        audio_embeds = self.audio_head(audio_embeds)

        text_embeds = text_outputs.hidden_states[-1][:, 0]
        text_embeds = self.text_audio_head(text_embeds)

        logits_per_audio = (audio_embeds @ text_embeds.T).transpose(1, 2)
        logits_per_audio = logits_per_audio * self.audio_logit_scale + self.audio_logit_bias
        logits_per_text = logits_per_audio.transpose(0, 1)

        loss = None
        if return_loss:
            labels = torch.eye(text_embeds.shape[0], device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.shape[0]

        return PeAudioOutput(
            logits_per_text=logits_per_text,
            logits_per_audio=logits_per_audio,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
            loss=loss,
        )


__all__ = [
    "PeAudioFrameLevelModel",
    "PeAudioModel",
    "PeAudioEncoder",
    "PeAudioEncoderConfig",
    "PeAudioConfig",
]
