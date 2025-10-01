from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ...utils.generic import check_model_inputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..dac.configuration_dac import DacConfig
from ..dac.modeling_dac import DacEncoder
from ..qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)


class PerceptionEncoderAudioTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_window_layers = max_window_layers
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        super().__init__(**kwargs)


class DACVAEConfig(DacConfig): ...


class PerceptionEncoderAudioConfig(PretrainedConfig):
    r"""
    PerceptionEncoderAudioConfig
    """

    sub_configs = {"text_model": AutoConfig}
    model_type = "perception_encoder_audio"

    def __init__(
        self,
        dac_vae_encoder: Optional[dict] = None,
        transformer: Optional[dict] = None,
        text_model: Optional[dict] = None,
        output_dim: int = 1024,
        nth_text_layer: Optional[int] = 22,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dac_vae_encoder = dac_vae_encoder or {}
        transformer = transformer or {}
        text_model = text_model or {}
        self.dac_vae_encoder = DACVAEConfig(**dac_vae_encoder)
        self.transformer = PerceptionEncoderAudioTransformerConfig(**transformer)
        self.text_model = CONFIG_MAPPING[kwargs.get("model_type", "modernbert")](**text_model)
        self.output_dim = output_dim
        self.nth_text_layer = nth_text_layer


## Patcher
class MaskedGroupNorm(torch.nn.GroupNorm):
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(x)
        B, C, L = x.shape
        G = self.num_groups
        x_grouped = x.view(B, G, C // G, L)
        padding_mask_grouped = padding_mask.view(B, G, C // G, L).bool()
        mean = torch.masked.mean(x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True)
        var = torch.masked.var(x_grouped, mask=padding_mask_grouped, dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, L)
        if self.affine:
            x_norm = x_norm * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x_norm * padding_mask


class ConvBlock1d(torch.nn.Module):
    def __init__(self, config: PerceptionEncoderAudioTransformerConfig):
        super().__init__()
        self.groupnorm = MaskedGroupNorm(num_groups=1, num_channels=config.hidden_size)
        self.activation = torch.nn.SiLU()
        self.project = torch.nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.groupnorm(x, padding_mask=padding_mask)
        x = self.activation(x)
        return self.project(x)


class ResnetBlock1d(torch.nn.Module):
    def __init__(self, config: PerceptionEncoderAudioTransformerConfig):
        super().__init__()
        self.block1 = ConvBlock1d(config)
        self.block2 = ConvBlock1d(config)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x, padding_mask=padding_mask)
        h = self.block2(h, padding_mask=padding_mask)
        return h + x


class PerceptionEncoderAudioContrastiveHead(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.layer_norm(x))


## Audio Codec
class VAEBottleneck(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        bottleneck_dim: int = 512,
    ):
        super().__init__()
        self.in_proj = torch.nn.Conv1d(input_dim, bottleneck_dim * 2, kernel_size=1)
        self.out_proj = torch.nn.Conv1d(bottleneck_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, scale = self.in_proj(z).chunk(2, dim=1)
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        latents = torch.randn_like(mean) * stdev + mean
        return latents


class DACVAEEncoder(DacEncoder): ...


class DacEncoderVAE(torch.nn.Module):
    def __init__(self, config: DACVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DACVAEEncoder(config)
        self.bottleneck = VAEBottleneck(config.codebook_size, config.codebook_dim)
        self.hop_length = config.hop_length
        self.sampling_rate = config.sampling_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            z = self.encoder(self._pad(waveform))
            encoded_frames = self.bottleneck(z)
        return encoded_frames

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs


## Transformer
class PerceptionEncoderAudioMLP(Qwen3MLP): ...


class PerceptionEncoderAudioRMSNorm(Qwen3RMSNorm): ...


class PerceptionEncoderAudioRotaryEmbedding(Qwen3RotaryEmbedding): ...


def stack_freqs(cos: torch.Tensor, sin: torch.Tensor):
    dim = cos.size(-1)
    cos = cos.narrow(-1, 0, dim // 2)
    sin = sin.narrow(-1, 0, dim // 2)
    freqs_cis = torch.stack((cos, -sin, sin, cos), dim=-1).view(*cos.size(), 2, 2)
    return freqs_cis


def apply_rotary_pos_emb(q, k, freqs_cis, unsqueeze_dim=1):
    freqs_cis = freqs_cis.unsqueeze(unsqueeze_dim)
    q_ = q.reshape(*q.shape[:-1], -1, 1, 2)
    k_ = k.reshape(*k.shape[:-1], -1, 1, 2)
    return (q_ * freqs_cis).sum(5).flatten(3), (k_ * freqs_cis).sum(5).flatten(3)


class PerceptionEncoderAudioAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False

    def _reshape_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        B, T, C = x.shape
        # B x T x C -> B x T x C/H x H
        x = x.reshape(B, T, C // heads, heads)
        # B x T x C/H x H -> B x H x T x C/H
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        **kwargs,
    ):
        # The only difference from `Qwen3Attention` is the reshape of Q/K/V
        # We reshape # B x T x C -> B x T x C/H x H, and then permute...
        # Qwen3 reshapes # B x T x C -> B x T x H x C/H

        input_shape = hidden_states.shape[:-1]
        nheads = hidden_states.size(-1) // self.head_dim

        query_states = self.q_norm(self._reshape_heads(self.q_proj(hidden_states), nheads))
        key_states = self.k_norm(self._reshape_heads(self.k_proj(hidden_states), nheads))
        value_states = self._reshape_heads(self.v_proj(hidden_states), nheads)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embeddings)

        attention_interface = eager_attention_forward
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


class PerceptionEncoderAudioDecoderLayer(Qwen3DecoderLayer): ...


class PerceptionEncoderAudioEmbeddings(torch.nn.Module):
    def __init__(self, config: PerceptionEncoderAudioTransformerConfig):
        super().__init__()
        self.config = config
        self.resnet_block = ResnetBlock1d(config)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        if padding_mask is not None:
            padding_mask = torch.cat([padding_mask[:, [0]], padding_mask], dim=1).unsqueeze(1).expand_as(x)
        x = rearrange(x, "b l c-> b c l")
        h = self.resnet_block(x, padding_mask=padding_mask)
        return rearrange(h, "b c l -> b l c")


class PerceptionEncoderAudioTransformer(torch.nn.Module):
    _can_record_outputs = {
        "attentions": PerceptionEncoderAudioAttention,
        "hidden_states": PerceptionEncoderAudioDecoderLayer,
    }

    def __init__(self, config: PerceptionEncoderAudioTransformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = PerceptionEncoderAudioEmbeddings(config)
        self.layers = torch.nn.ModuleList(
            [PerceptionEncoderAudioDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PerceptionEncoderAudioRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope_embeddings = PerceptionEncoderAudioRotaryEmbedding(config)
        self.output = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        inputs_embeds = self.embeddings(inputs_embeds, padding_mask=attention_mask)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        cos, sin = self.rope_embeddings(inputs_embeds, position_ids)
        position_embeddings = stack_freqs(cos, sin)

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

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
        )


class PerceptionEncoderAudioPretrainedModel(PreTrainedModel):
    config: PerceptionEncoderAudioConfig
    base_model_prefix = "perception_encoder_audio"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@dataclass
class PerceptionEncoderAudioOutput(BaseModelOutputWithPooling):
    # Padding mask for the encoded audio (shortened by `hop_length` of the dac_vae_encoder)
    audio_feature_padding_mask: Optional[torch.Tensor] = None
    dac_vae_features: Optional[torch.Tensor] = None


class PerceptionEncoderAudioModel(PerceptionEncoderAudioPretrainedModel):
    def __init__(self, config: PerceptionEncoderAudioConfig):
        super().__init__(config)
        # Synchronize _attn_implementations
        config.transformer._attn_implementation = config._attn_implementation
        self.data_proj = torch.nn.Linear(config.dac_vae_encoder.codebook_dim, config.transformer.hidden_size)
        self.dac_vae_encoder = DacEncoderVAE(config.dac_vae_encoder)
        self.transformer = PerceptionEncoderAudioTransformer(config.transformer)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> PerceptionEncoderAudioOutput:
        codec_features = self.dac_vae_encoder(input_values).transpose(1, 2)
        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[:, :: self.dac_vae_encoder.config.hop_length]
        outputs = self.transformer(self.data_proj(codec_features), padding_mask=feature_padding_mask)
        return PerceptionEncoderAudioOutput(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            audio_feature_padding_mask=feature_padding_mask,
            dac_vae_features=codec_features,
        )


@dataclass
class PerceptionEncoderAudioTextOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    audio_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class PerceptionEncoderAudioWithTextModel(PerceptionEncoderAudioModel):
    def __init__(self, config: PerceptionEncoderAudioConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_model)
        self.audio_text_head = PerceptionEncoderAudioContrastiveHead(config.text_model.hidden_size, config.output_dim)
        self.audio_head = PerceptionEncoderAudioContrastiveHead(config.transformer.hidden_size, config.output_dim)
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

    def _get_audio_head_input(self, audio_model_output):
        return audio_model_output.pooler_output

    def forward(
        self,
        input_ids: torch.Tensor,  # tokenized text
        input_values: torch.Tensor,  # audio waveform
        attention_mask: Optional[torch.Tensor] = None,  # text attention mask
        padding_mask: Optional[torch.Tensor] = None,  # audio padding mask
        return_loss=False,
    ) -> PerceptionEncoderAudioTextOutput:
        audio_model_output = super().forward(input_values, padding_mask)
        text_model_output = self._get_text_output(input_ids, attention_mask)

        text_embeds = self.audio_text_head(text_model_output.pooler_output)
        audio_embeds = self.audio_head(self._get_audio_head_input(audio_model_output))

        loss = None
        if return_loss:
            logits_per_text = text_embeds @ audio_embeds.t()
            logits_per_text = logits_per_text * self.logit_scale + self.logit_bias
            labels = torch.eye(text_embeds.size(0), device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.size(0)

        return PerceptionEncoderAudioTextOutput(
            loss=loss,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_model_output,
            audio_model_output=audio_model_output,
        )


class PerceptionEncoderAudioFrameWithText(PerceptionEncoderAudioWithTextModel):
    def _get_audio_head_input(self, audio_model_output):
        return audio_model_output.last_hidden_state


__all__ = [
    "PerceptionEncoderAudioModel",
    "PerceptionEncoderAudioWithTextModel",
    "PerceptionEncoderAudioConfig",
    "PerceptionEncoderAudioFrameWithText",
]
