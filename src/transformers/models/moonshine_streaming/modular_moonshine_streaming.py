# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from torch import Tensor

from ...cache_utils import Cache
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ProcessingKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..llama.modeling_llama import LlamaMLP, eager_attention_forward
from ..moonshine.modeling_moonshine import (
    MoonshineDecoder,
    MoonshineEncoderLayer,
    MoonshineEncoderMLP,
    MoonshineForConditionalGeneration,
    MoonshineModel,
    MoonshinePreTrainedModel,
)
from ..wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from .configuration_moonshine_streaming import MoonshineStreamingConfig, MoonshineStreamingEncoderConfig


logger = logging.get_logger(__name__)


class MoonshineStreamingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "pad_to_multiple_of": 80,
            "padding": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class MoonshineStreamingProcessor(Wav2Vec2Processor): ...


@dataclass
@auto_docstring(
    custom_intro="""
    Extends [~modeling_outputs.BaseModelOutput] to include the output attention mask since sequence length is not preserved in the model's forward.
    """
)
class MoonshineStreamingEncoderModelOutput(BaseModelOutput):
    attention_mask: torch.Tensor | None = None


class MoonshineStreamingFrameCMVN(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        rms = (centered.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return centered / rms


class MoonshineStreamingAsinhCompression(nn.Module):
    def __init__(self, k_init: float = 0.75):
        super().__init__()
        self.log_k = nn.Parameter(torch.log(torch.tensor(k_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(torch.exp(self.log_k) * x)


class MoonshineStreamingCausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        x = super().forward(x)

        if mask is not None:
            mask = nn.functional.pad(mask, (self.left_pad, 0))[:, None, :]
            weight = torch.ones(1, 1, self.kernel_size[0], device=mask.device)
            mask = nn.functional.conv1d(mask.float(), weight, stride=self.stride)
            mask = mask > 0
            x *= mask

        if mask is not None:
            mask = mask.squeeze(1)
        return x, mask


class MoonshineStreamingLayerNorm(nn.Module):
    def __init__(self, dim: int, unit_offset: bool = True, device=None, dtype=None):
        super().__init__()
        self.unit_offset = float(unit_offset)
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, device=device, dtype=dtype)
        self.gamma = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        normed = self.ln(x)
        gamma = self.gamma + self.unit_offset
        return normed * gamma


class MoonshineStreamingEncoderMLP(MoonshineEncoderMLP): ...


class MoonshineStreamingEncoderAttention(nn.Module):
    def __init__(self, config: MoonshineStreamingConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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


class MoonshineStreamingEncoderLayer(MoonshineEncoderLayer):
    def __init__(self, config: MoonshineStreamingConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MoonshineStreamingEncoderAttention(config, layer_idx)
        self.mlp = MoonshineStreamingEncoderMLP(config, config.hidden_act)
        self.input_layernorm = MoonshineStreamingLayerNorm(config.hidden_size)
        self.post_attention_layernorm = MoonshineStreamingLayerNorm(config.hidden_size)


class MoonshineStreamingEncoderEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cmvn = MoonshineStreamingFrameCMVN()
        self.comp = MoonshineStreamingAsinhCompression()
        self.conv1 = MoonshineStreamingCausalConv1d(
            config.hidden_size, config.hidden_size * 2, kernel_size=5, stride=2
        )
        self.conv2 = MoonshineStreamingCausalConv1d(
            config.hidden_size * 2, config.hidden_size, kernel_size=5, stride=2
        )
        self.frame_len = int(round(config.sample_rate * config.frame_ms / 1000.0))
        self.linear = nn.Linear(self.frame_len, config.hidden_size, bias=False)

    def forward(self, input_values, padding_mask=None):
        hidden_states = self.cmvn(input_values.reshape(input_values.shape[0], -1, self.frame_len))
        hidden_states = self.comp(hidden_states)
        hidden_states = nn.functional.silu(self.linear(hidden_states))

        if padding_mask is not None:
            num_frames = padding_mask.sum(-1) // self.frame_len
            padding_mask = (
                torch.arange(hidden_states.shape[1], device=padding_mask.device)[None, :] < num_frames[:, None]
            )
            hidden_states *= padding_mask[..., None]

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states, padding_mask = self.conv1(hidden_states, padding_mask)
        hidden_states = nn.functional.silu(hidden_states)
        hidden_states, padding_mask = self.conv2(hidden_states, padding_mask)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states, padding_mask


class MoonshineStreamingPreTrainedModel(MoonshinePreTrainedModel):
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        frame_len = int(round(self.config.encoder_config.sample_rate * self.config.encoder_config.frame_ms / 1000.0))
        output_lengths = input_lengths // frame_len
        output_lengths = (output_lengths - 1) // 2 + 1
        output_lengths = (output_lengths - 1) // 2 + 1
        return output_lengths

    def _init_weights(self, module: nn.Module):
        if isinstance(module, MoonshineStreamingLayerNorm):
            nn.init.constant_(module.gamma, 1.0 - module.unit_offset)
        else:
            super()._init_weights(module)


def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        left_window_size, right_window_size = sliding_window

        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class MoonshineStreamingEncoder(MoonshineStreamingPreTrainedModel):
    config: MoonshineStreamingEncoderConfig
    _can_record_outputs = {
        "attentions": OutputRecorder(MoonshineStreamingEncoderAttention, index=1, layer_name="self_attn"),
        "hidden_states": MoonshineStreamingEncoderLayer,
    }

    def __init__(self, config: MoonshineStreamingEncoderConfig):
        super().__init__(config)
        self.embedder = MoonshineStreamingEncoderEmbedder(config)
        self.layers = nn.ModuleList(
            [MoonshineStreamingEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.final_norm = MoonshineStreamingLayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
                Float values of the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
                `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or
                the soundfile library (`pip install soundfile`). To prepare the array into
                `input_values`, the [`AutoFeatureExtractor`] should be used for padding
                and conversion into a tensor of type `torch.FloatTensor`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding indices in `input_values`. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
        """
        inputs_embeds, attention_mask = self.embedder(input_values, padding_mask=attention_mask)

        if attention_mask is not None:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            per_layer_attention_mask = [
                create_bidirectional_mask(
                    and_mask_function=sliding_window_mask_function(self.config.sliding_windows[layer_idx]),
                    **mask_kwargs,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]

        hidden_states = inputs_embeds
        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=per_layer_attention_mask[layer_idx] if attention_mask is not None else None,
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return MoonshineStreamingEncoderModelOutput(last_hidden_state=hidden_states, attention_mask=attention_mask)


class MoonshinMoonshineStreamingDecoderMLP(LlamaMLP): ...


class MoonshineStreamingDecoder(MoonshineDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.pos_emb = nn.Embedding(self.config.max_position_embeddings, config.encoder_config.hidden_size)

        if config.encoder_config.hidden_size != self.config.hidden_size:
            self.proj = nn.Linear(config.encoder_config.hidden_size, self.config.hidden_size, bias=False)
        else:
            self.proj = nn.Identity()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding indices in `encoder_hidden_states`. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        """
        position_embeddings = self.pos_emb(
            torch.arange(encoder_hidden_states.shape[1], device=encoder_hidden_states.device)
        )
        encoder_hidden_states += position_embeddings
        encoder_hidden_states = self.proj(encoder_hidden_states)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )


class MoonshineStreamingModel(MoonshineModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = MoonshineStreamingEncoder(config.encoder_config)
        self.decoder = MoonshineStreamingDecoder(config)


class MoonshineStreamingForConditionalGeneration(MoonshineForConditionalGeneration): ...


__all__ = [
    "MoonshineStreamingPreTrainedModel",
    "MoonshineStreamingModel",
    "MoonshineStreamingForConditionalGeneration",
    "MoonshineStreamingProcessor",
]
