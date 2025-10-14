from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import check_model_inputs
from ..auto import AutoModel
from ..clip.modeling_clip import CLIPOutput
from ..dac.modeling_dac import DacEncoder
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)
from .configuration_pe_audio import PEAudioConfig, PEAudioEncoderConfig


class PEAudioMaskedGroupNorm(nn.GroupNorm):
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


class PEAudioConvBlock1d(nn.Module):
    def __init__(self, config: PEAudioEncoderConfig):
        super().__init__()
        self.groupnorm = PEAudioMaskedGroupNorm(num_groups=1, num_channels=config.hidden_size)
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


class PEAudioResnetBlock1d(nn.Module):
    def __init__(self, config: PEAudioEncoderConfig):
        super().__init__()
        self.block1 = PEAudioConvBlock1d(config)
        self.block2 = PEAudioConvBlock1d(config)

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


class PEAudioEmbeddings(nn.Module):
    def __init__(self, config: PEAudioEncoderConfig):
        super().__init__()
        self.resnet_block = PEAudioResnetBlock1d(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, inputs_embeds, padding_mask=None):
        hidden_states = torch.cat([self.cls_token.expand(inputs_embeds.size(0), -1, -1), inputs_embeds], dim=1)

        if padding_mask is not None:
            # TODO: any reason why we take padding_mask[0] and not just 1?
            padding_mask = torch.cat([padding_mask[:, [0]], padding_mask], dim=1)

        hidden_states = self.resnet_block(hidden_states, padding_mask=padding_mask)
        return hidden_states, padding_mask


class PEAudioRotaryEmbedding(Qwen3RotaryEmbedding): ...


class PEAudioAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.is_causal = False
        self.sliding_window = None


class PEAudioDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        del self.attention_type


class PEAudioRMSNorm(Qwen3RMSNorm): ...


class PEAudioTransformer(nn.Module):
    _can_record_outputs = {
        "attentions": PEAudioAttention,
        "hidden_states": PEAudioDecoderLayer,
    }

    def __init__(self, config: PEAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = PEAudioEmbeddings(config)
        self.layers = nn.ModuleList(
            [PEAudioDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PEAudioRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope_embeddings = PEAudioRotaryEmbedding(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    @check_model_inputs
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, attention_mask = self.embeddings(inputs_embeds, padding_mask=attention_mask)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rope_embeddings(inputs_embeds, position_ids)

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


class PEAudioDacEncoder(DacEncoder): ...


class PEAudioContrastiveHead(nn.Module):
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


class PEAudioPretrainedModel(PreTrainedModel):
    config: PEAudioConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True


# TODO: not sure about the typing for text_model_output
class PEAudioOutput(CLIPOutput): ...


class PEAudioEncoder(PEAudioPretrainedModel):
    config_class = PEAudioEncoderConfig
    base_model_prefix = "audio_encoder"

    def __init__(self, config: PEAudioEncoderConfig):
        super().__init__(config)

        self.encoder = PEAudioDacEncoder(config.dac_config)
        self.bottleneck = nn.Conv1d(config.dac_config.hidden_size, config.hidden_size, 1)
        self.data_proj = nn.Linear(config.dac_config.codebook_dim, config.hidden_size)
        self.transformer = PEAudioTransformer(config)

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            hidden_states = self.encoder(input_values)  # (batch_size, hidden_size, seq_len)
            hidden_states = self.bottleneck(hidden_states)  # (batch_size, hidden_size, seq_len)

        codec_features = hidden_states.transpose(1, 2)
        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = padding_mask[:, :: self.dac_vae_encoder.config.hop_length]
        outputs = self.transformer(self.data_proj(codec_features), attention_mask=feature_padding_mask)

        # TODO: shoud we return codec features?
        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PEAudioModel(PEAudioPretrainedModel):
    def __init__(self, config: PEAudioConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = PEAudioEncoder(config.audio_config)

        self.text_head_audio = PEAudioContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.audio_head = PEAudioContrastiveHead(config.audio_config.hidden_size, config.projection_dim)

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

    def get_audio_features(self, input_values, padding_mask=None):
        # TODO: should it be named feature or embeds
        audio_outputs: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            return_dict=True,
        )
        audio_features = self.audio_head(audio_outputs.pooler_output)
        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
    ) -> PEAudioOutput:
        audio_output: BaseModelOutputWithPooling = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            return_dict=True,
        )

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.config.nth_text_layer,
            return_dict=True,
        )

        audio_embeds = audio_output.pooler_output
        if self.config.nth_audio_layer is not None:
            text_embeds = text_output.hidden_states[self.config.nth_audio_layer]
        else:
            text_embeds = text_output.last_hidden_state

        audio_embeds = self.audio_head(audio_embeds)
        text_embeds = self.text_head(text_embeds)

        # TODO: there is not logits per audio?

        loss = None
        if return_loss:
            logits_per_text = text_embeds @ audio_embeds.t()
            logits_per_text = logits_per_text * self.logit_scale + self.logit_bias
            labels = torch.eye(text_embeds.shape[0], device=text_embeds.device)
            loss = -F.logsigmoid(labels * logits_per_text).sum() / text_embeds.shape[0]

        return PEAudioOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            text_model_output=text_output,
            audio_model_output=audio_output,
        )


__all__ = [
    "PEAudioModel",
    "PEAudioEncoder",
]
