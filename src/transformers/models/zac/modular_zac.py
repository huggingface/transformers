# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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
#
# Ported from ZAC-1 (https://github.com/Zyphra/ZAC-1, Apache-2.0, Copyright Zyphra), which itself adapts
# descript-audio-codec (MIT, Copyright (c) 2023-present Descript) and EnCodec (MIT, Copyright (c) Meta Platforms, Inc.
# and affiliates).
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_causal_mask, sliding_window_overlay
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedAudioTokenizerBase
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoConfig, AutoModel
from ..dac.modeling_dac import (
    DacEncoderBlock,
    DacResidualVectorQuantizer,
    DacVectorQuantize,
    Snake1d,
)
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..mimi.modeling_mimi import (
    MimiAttention,
    MimiLayerScale,
    MimiMLP,
    MimiRotaryEmbedding,
    MimiTransformerLayer,
    eager_attention_forward,
)
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCausalConvNet, Qwen3OmniMoeConvNeXtBlock
from ..xcodec2.modeling_xcodec2 import Xcodec2ISTFTHead


@auto_docstring(checkpoint="Zyphra/ZAC-1-hf")  # TODO: move to final Zyphra repo
@strict
class ZacConfig(PreTrainedConfig):
    r"""
    semantic_model_config (`Union[dict, Wav2Vec2BertConfig]`, *optional*):
        Configuration of the frozen Wav2Vec2-BERT 2.0 teacher (its first 16 layers), whose features feed the semantic
        codebook.
    internal_sampling_rate (`int`, *optional*, defaults to 44160):
        Sampling rate the codec runs at internally. Audio is resampled from `sampling_rate` on the way in and back on
        the way out.
    hop_length (`int`, *optional*, defaults to 2940):
        Number of `sampling_rate` samples per code frame.
    internal_hop_length (`int`, *optional*, defaults to 2944):
        Number of `internal_sampling_rate` samples per code frame.
    num_quantizers (`int`, *optional*, defaults to 10):
        Number of codebooks: one semantic codebook followed by `num_quantizers - 1` acoustic residual codebooks.
    semantic_hidden_size (`int`, *optional*, defaults to 768):
        Width of the semantic branch (semantic encoder, semantic codebook and semantic head).
    semantic_intermediate_size (`int`, *optional*, defaults to 2048):
        Hidden size of the ConvNeXt blocks of the semantic encoder and semantic head.
    semantic_num_layers (`int`, *optional*, defaults to 4):
        Number of ConvNeXt blocks of the semantic encoder and of the semantic head.
    semantic_window_ratio (`float`, *optional*, defaults to 2.0):
        Width, in code frames, of the box filter that averages the teacher features onto the code frames.
    encoder_strides (`list[int]`, *optional*, defaults to `[2, 4]`):
        Strides of the causal convolutional encoder blocks (the channels double at each block).
    encoder_patch_sizes (`list[int]`, *optional*, defaults to `[23, 2, 2, 2, 2]`):
        Time-to-channel patch size applied before each encoder transformer stage.
    encoder_input_sizes (`list[int]`, *optional*, defaults to `[5888, 768, 768, 768, 1280]`):
        Input width of each encoder transformer stage.
    encoder_hidden_sizes (`list[int]`, *optional*, defaults to `[768, 768, 768, 768, 1280]`):
        Hidden width of each encoder transformer stage.
    encoder_output_sizes (`list[int]`, *optional*, defaults to `[384, 384, 384, 640, 1024]`):
        Output width of each encoder transformer stage.
    encoder_num_attention_heads (`list[int]`, *optional*, defaults to `[12, 12, 12, 12, 20]`):
        Number of attention heads of each encoder transformer stage.
    encoder_num_hidden_layers (`list[int]`, *optional*, defaults to `[12, 12, 12, 12, 32]`):
        Number of layers of each encoder transformer stage.
    encoder_intermediate_sizes (`list[int]`, *optional*, defaults to `[3072, 3072, 3072, 3072, 5120]`):
        MLP hidden size of each encoder transformer stage.
    encoder_sliding_windows (`list[int]`, *optional*, defaults to `[480, 480, 480, 300, 150]`):
        Causal attention window (in frames of the stage) of each encoder transformer stage.
    decoder_unpatch_size (`int`, *optional*, defaults to 2):
        Channel-to-time un-patch size applied between the decoder transformer stages.
    decoder_input_sizes (`list[int]`, *optional*, defaults to `[1024, 640, 384, 384, 384]`):
        Input width of each decoder transformer stage.
    decoder_hidden_sizes (`list[int]`, *optional*, defaults to `[1280, 768, 768, 768, 768]`):
        Hidden width of each decoder transformer stage.
    decoder_output_sizes (`list[int | None]`, *optional*, defaults to `[1280, 768, 768, 768, None]`):
        Output width of each decoder transformer stage; `None` means no output projection.
    decoder_num_attention_heads (`list[int]`, *optional*, defaults to `[20, 12, 12, 12, 12]`):
        Number of attention heads of each decoder transformer stage.
    decoder_num_hidden_layers (`list[int]`, *optional*, defaults to `[2, 2, 2, 2, 2]`):
        Number of layers of each decoder transformer stage (`[2, 2, 2, 2, 2]` for the `keep2_plain` decoder,
        `[32, 12, 12, 12, 12]` for `full_mr`).
    decoder_intermediate_sizes (`list[int]`, *optional*, defaults to `[5120, 3072, 3072, 3072, 3072]`):
        MLP hidden size of each decoder transformer stage.
    decoder_sliding_windows (`list[int]`, *optional*, defaults to `[150, 300, 480, 480, 480]`):
        Causal attention window (in frames of the stage) of each decoder transformer stage.
    convnext_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the layer norms of the ConvNeXt blocks and of the iSTFT head.
    head_num_layers (`int`, *optional*, defaults to 8):
        Number of ConvNeXt blocks of the iSTFT head.
    head_intermediate_size (`int`, *optional*, defaults to 2048):
        Hidden size of the ConvNeXt blocks of the iSTFT head.
    head_hop_length (`int`, *optional*, defaults to 184):
        Hop length of the iSTFT branches, in `internal_sampling_rate` samples.
    head_n_ffts (`list[int]`, *optional*, defaults to `[736, 2208]`):
        FFT size of each iSTFT branch.
    head_mixer_kernel_size (`int`, *optional*, defaults to 127):
        Kernel size of the linear FIR filter that mixes the iSTFT branches.
    head_max_magnitude (`float`, *optional*, defaults to 100.0):
        Upper bound of the predicted spectral magnitudes.
    resample_lowpass_filter_width (`int`, *optional*, defaults to 16):
        Low-pass filter width of the sinc resamplers between `sampling_rate` and `internal_sampling_rate`.
    resample_rolloff (`float`, *optional*, defaults to 0.99):
        Roll-off of the sinc resamplers between `sampling_rate` and `internal_sampling_rate`.
    lane_frames (`int`, *optional*, defaults to 300):
        Number of code frames the encoder processes at once (20 s). Longer audio is encoded in several lanes.
    lane_context_frames (`int`, *optional*, defaults to 150):
        Number of already-encoded frames each lane after the first re-reads as left context (10 s).

    Example:

    ```python
    >>> from transformers import ZacConfig, ZacModel

    >>> # Initializing a ZAC-1 configuration
    >>> configuration = ZacConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ZacModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "zac"
    sub_configs_defaults = {
        "semantic_model_config": SubConfigSpec(
            config_class=AutoConfig, model_type="wav2vec2-bert", init_kwargs={"num_hidden_layers": 16}
        ),
    }

    semantic_model_config: dict | PreTrainedConfig | None = None
    sampling_rate: int = 44100
    internal_sampling_rate: int = 44160
    hop_length: int = 2940
    internal_hop_length: int = 2944
    num_quantizers: int = 10
    codebook_size: int = 4096
    codebook_dim: int = 8
    hidden_size: int = 1024
    semantic_hidden_size: int = 768
    semantic_intermediate_size: int = 2048
    semantic_num_layers: int = 4
    semantic_window_ratio: float = 2.0
    encoder_hidden_size: int = 64
    encoder_strides: list[int] | tuple[int, ...] = (2, 4)
    encoder_patch_sizes: list[int] | tuple[int, ...] = (23, 2, 2, 2, 2)
    encoder_input_sizes: list[int] | tuple[int, ...] = (5888, 768, 768, 768, 1280)
    encoder_hidden_sizes: list[int] | tuple[int, ...] = (768, 768, 768, 768, 1280)
    encoder_output_sizes: list[int] | tuple[int, ...] = (384, 384, 384, 640, 1024)
    encoder_num_attention_heads: list[int] | tuple[int, ...] = (12, 12, 12, 12, 20)
    encoder_num_hidden_layers: list[int] | tuple[int, ...] = (12, 12, 12, 12, 32)
    encoder_intermediate_sizes: list[int] | tuple[int, ...] = (3072, 3072, 3072, 3072, 5120)
    encoder_sliding_windows: list[int] | tuple[int, ...] = (480, 480, 480, 300, 150)
    decoder_unpatch_size: int = 2
    decoder_input_sizes: list[int] | tuple[int, ...] = (1024, 640, 384, 384, 384)
    decoder_hidden_sizes: list[int] | tuple[int, ...] = (1280, 768, 768, 768, 768)
    decoder_output_sizes: list[int | None] | tuple[int | None, ...] = (1280, 768, 768, 768, None)
    decoder_num_attention_heads: list[int] | tuple[int, ...] = (20, 12, 12, 12, 12)
    decoder_num_hidden_layers: list[int] | tuple[int, ...] = (2, 2, 2, 2, 2)
    decoder_intermediate_sizes: list[int] | tuple[int, ...] = (5120, 3072, 3072, 3072, 3072)
    decoder_sliding_windows: list[int] | tuple[int, ...] = (150, 300, 480, 480, 480)
    head_dim: int = 64
    max_position_embeddings: int = 8192
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "gelu"
    norm_eps: float = 1e-5
    convnext_norm_eps: float = 1e-6
    layer_scale_initial_scale: float = 0.01
    attention_dropout: float | int = 0.0
    head_num_layers: int = 8
    head_intermediate_size: int = 2048
    head_hop_length: int = 184
    head_n_ffts: list[int] | tuple[int, ...] = (736, 2208)
    head_mixer_kernel_size: int = 127
    head_max_magnitude: float = 100.0
    resample_lowpass_filter_width: int = 16
    resample_rolloff: float = 0.99
    lane_frames: int = 300
    lane_context_frames: int = 150
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        encoder_hop = math.prod(self.encoder_strides) * math.prod(self.encoder_patch_sizes)
        if encoder_hop != self.internal_hop_length:
            raise ValueError(
                f"prod(encoder_strides) * prod(encoder_patch_sizes) = {encoder_hop} must equal "
                f"internal_hop_length = {self.internal_hop_length}."
            )
        decoder_hop = self.decoder_unpatch_size ** (len(self.decoder_hidden_sizes) - 1) * self.head_hop_length
        if decoder_hop != self.internal_hop_length:
            raise ValueError(
                f"decoder_unpatch_size ** (len(decoder_hidden_sizes) - 1) * head_hop_length = {decoder_hop} must "
                f"equal internal_hop_length = {self.internal_hop_length}."
            )
        if self.lane_context_frames >= self.lane_frames:
            raise ValueError(
                f"lane_context_frames ({self.lane_context_frames}) must be smaller than lane_frames ({self.lane_frames})."
            )

    @property
    def frame_rate(self) -> float:
        """Number of code frames per second."""
        return self.sampling_rate / self.hop_length

    @property
    def frame_size(self) -> int:
        """Number of audio samples per code frame."""
        return self.hop_length


@auto_docstring
@dataclass
class ZacOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`, *optional*):
        Reconstructed audio at the raw decoder level.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
        Discrete codes computed with `model.encode`: codebook 0 is semantic, codebooks 1+ are acoustic.
    audio_codes_mask (`torch.LongTensor` of shape `(batch_size, codes_length)`, *optional*):
        Mask of the valid frames of `audio_codes`.
    """

    audio_values: torch.FloatTensor | None = None
    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.LongTensor | None = None


@auto_docstring
@dataclass
class ZacEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`, *optional*):
        Discrete codes: codebook 0 is semantic, codebooks 1+ are acoustic. Padded frames are 0.
    audio_codes_mask (`torch.LongTensor` of shape `(batch_size, codes_length)`, *optional*):
        Mask of the valid frames of `audio_codes`.
    """

    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.LongTensor | None = None


@auto_docstring
@dataclass
class ZacDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, 1, codes_length * hop_length)`, *optional*):
        Decoded audio at the raw decoder level (ZAC-1 applies a 3 dB headroom gain for playback).
    """

    audio_values: torch.FloatTensor | None = None


class ZacCausalConv1d(Qwen3OmniMoeCausalConvNet):
    pass


class ZacSnake1d(Snake1d):
    pass


class ZacResidualUnit(nn.Module):
    """A residual unit of Snake1d and causal weight-normed convolutions (DAC's, with left-only padding)."""

    def __init__(self, dimension: int, dilation: int):
        super().__init__()
        self.snake1 = ZacSnake1d(dimension)
        self.conv1 = ZacCausalConv1d(dimension, dimension, kernel_size=7, dilation=dilation)
        self.snake2 = ZacSnake1d(dimension)
        self.conv2 = ZacCausalConv1d(dimension, dimension, kernel_size=1)

    def forward(self, hidden_state):
        return hidden_state + self.conv2(self.snake2(self.conv1(self.snake1(hidden_state))))


class ZacEncoderBlock(DacEncoderBlock):
    def __init__(self, config: ZacConfig, stride: int = 1, stride_index: int = 1):
        super().__init__(config, stride, stride_index)
        dimension = config.encoder_hidden_size * 2**stride_index
        self.res_unit1 = ZacResidualUnit(dimension // 2, dilation=1)
        self.res_unit2 = ZacResidualUnit(dimension // 2, dilation=3)
        self.res_unit3 = ZacResidualUnit(dimension // 2, dilation=9)
        self.snake1 = ZacSnake1d(dimension // 2)
        self.conv1 = ZacCausalConv1d(dimension // 2, dimension, kernel_size=2 * stride, stride=stride)


class ZacConvNeXtBlock(Qwen3OmniMoeConvNeXtBlock):
    """Causal ConvNeXt block (left padding 6) without layer scale."""

    def __init__(self, dim: int, intermediate_size: int, eps: float):
        super().__init__(dim)
        self.dwconv = ZacCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_size)
        self.pwconv2 = nn.Linear(intermediate_size, dim)
        del self.gamma

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.pwconv2(self.act(self.pwconv1(self.norm(hidden_states))))
        return residual + hidden_states.transpose(1, 2)


class ZacRotaryEmbedding(MimiRotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(config: ZacConfig, **kwargs) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies the way ZAC-1 does (`exp(-2 i log(theta) / dim)`, which differs from
        `theta ** (-2 i / dim)` in the last float32 bits).
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = torch.exp(torch.arange(dim // 2, dtype=torch.float) * (-math.log(base) * 2 / dim))
        return inv_freq, 1.0


class ZacAttention(MimiAttention):
    def __init__(self, config: ZacConfig, stage_idx: int, is_decoder: bool, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        if is_decoder:
            hidden_size = config.decoder_hidden_sizes[stage_idx]
            num_heads = config.decoder_num_attention_heads[stage_idx]
            self.sliding_window = config.decoder_sliding_windows[stage_idx]
        else:
            hidden_size = config.encoder_hidden_sizes[stage_idx]
            num_heads = config.encoder_num_attention_heads[stage_idx]
            self.sliding_window = config.encoder_sliding_windows[stage_idx]
        self.attention_dropout = config.attention_dropout
        self.head_dim = config.head_dim
        self.num_key_value_groups = 1
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ZAC-1 rotates adjacent pairs, in float32
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_interleave(query_states.float(), key_states.float(), cos, sin)
        query_states, key_states = query_states.to(value_states.dtype), key_states.to(value_states.dtype)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ZacMLP(MimiMLP):
    def __init__(self, config: ZacConfig, stage_idx: int, is_decoder: bool):
        nn.Module.__init__(self)
        if is_decoder:
            hidden_size = config.decoder_hidden_sizes[stage_idx]
            intermediate_size = config.decoder_intermediate_sizes[stage_idx]
        else:
            hidden_size = config.encoder_hidden_sizes[stage_idx]
            intermediate_size = config.encoder_intermediate_sizes[stage_idx]
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)


class ZacLayerScale(MimiLayerScale):
    def __init__(self, config: ZacConfig, stage_idx: int, is_decoder: bool):
        nn.Module.__init__(self)
        channels = (config.decoder_hidden_sizes if is_decoder else config.encoder_hidden_sizes)[stage_idx]
        self.scale = nn.Parameter(torch.full((channels,), config.layer_scale_initial_scale))


class ZacTransformerLayer(MimiTransformerLayer):
    def __init__(self, config: ZacConfig, stage_idx: int, is_decoder: bool, layer_idx: int):
        nn.Module.__init__(self)
        hidden_size = (config.decoder_hidden_sizes if is_decoder else config.encoder_hidden_sizes)[stage_idx]
        self.hidden_size = hidden_size
        self.self_attn = ZacAttention(config, stage_idx, is_decoder, layer_idx)
        self.mlp = ZacMLP(config, stage_idx, is_decoder)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = ZacLayerScale(config, stage_idx, is_decoder)
        self.mlp_layer_scale = ZacLayerScale(config, stage_idx, is_decoder)


class ZacTransformerStage(nn.Module):
    """Causal sliding-window transformer over channels-first features, with input and output projections."""

    def __init__(self, config: ZacConfig, stage_idx: int, is_decoder: bool):
        super().__init__()
        self.config = config
        if is_decoder:
            input_size = config.decoder_input_sizes[stage_idx]
            hidden_size = config.decoder_hidden_sizes[stage_idx]
            output_size = config.decoder_output_sizes[stage_idx]
            num_layers = config.decoder_num_hidden_layers[stage_idx]
            self.sliding_window = config.decoder_sliding_windows[stage_idx]
        else:
            input_size = config.encoder_input_sizes[stage_idx]
            hidden_size = config.encoder_hidden_sizes[stage_idx]
            output_size = config.encoder_output_sizes[stage_idx]
            num_layers = config.encoder_num_hidden_layers[stage_idx]
            self.sliding_window = config.encoder_sliding_windows[stage_idx]
        self.input_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [ZacTransformerLayer(config, stage_idx, is_decoder, layer_idx) for layer_idx in range(num_layers)]
        )
        self.output_proj = None if output_size is None else nn.Linear(hidden_size, output_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: nn.Module, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        hidden_states = self.input_proj(hidden_states.transpose(1, 2))
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            cache=None,
            position_ids=position_ids,
            and_mask_function=sliding_window_overlay(self.sliding_window),
        )
        position_embeddings = rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs
            )
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states.transpose(1, 2)


class ZacVectorQuantize(DacVectorQuantize):
    def __init__(self, config: ZacConfig, is_semantic: bool):
        nn.Module.__init__(self)
        channels = config.semantic_hidden_size if is_semantic else config.hidden_size
        self.codebook_dim = config.codebook_dim
        self.in_proj = nn.Conv1d(channels, config.codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(config.codebook_dim, channels, kernel_size=1)
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)

    def decode_latents(self, hidden_states):
        batch_size, hidden_dim, sequence_length = hidden_states.shape
        encodings = F.normalize(hidden_states.transpose(1, 2).reshape(batch_size * sequence_length, hidden_dim))
        codebook = F.normalize(self.codebook.weight)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(batch_size, -1)
        quantized_representation = self.codebook(indices).transpose(1, 2)
        return quantized_representation, indices


class ZacResidualVectorQuantizer(DacResidualVectorQuantizer):
    """The semantic quantizer (one codebook on the semantic features) or the acoustic one (the residual codebooks)."""

    def __init__(self, config: ZacConfig, is_semantic: bool):
        nn.Module.__init__(self)
        self.n_codebooks = 1 if is_semantic else config.num_quantizers - 1
        self.quantizers = nn.ModuleList([ZacVectorQuantize(config, is_semantic) for _ in range(self.n_codebooks)])
        self.quantizer_dropout = 0.0


class ZacISTFTBranch(Xcodec2ISTFTHead):
    """One iSTFT branch of the multi-resolution head: log-magnitude and phase, periodic Hann, centered iSTFT."""

    def __init__(self, config: ZacConfig, n_fft: int):
        nn.Module.__init__(self)
        self.linear = nn.Linear(config.decoder_hidden_sizes[-1], n_fft + 2)
        self.n_fft = n_fft
        self.hop_length = config.head_hop_length
        self.max_log_magnitude = math.log(config.head_max_magnitude)
        self.window = nn.Buffer(torch.hann_window(n_fft, periodic=True), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_frames = hidden_states.shape[1]
        log_magnitude, phase = self.linear(hidden_states).float().chunk(2, dim=-1)
        magnitude = torch.exp(log_magnitude.clamp(max=self.max_log_magnitude)).transpose(1, 2)
        phase = phase.transpose(1, 2)
        spectrogram = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        return torch.istft(
            spectrogram,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.float(),
            center=True,
            length=num_frames * self.hop_length,
        )


class ZacISTFTHead(nn.Module):
    """Multi-resolution iSTFT head: a causal ConvNeXt trunk, one iSTFT branch per FFT size and a linear FIR mixer."""

    def __init__(self, config: ZacConfig):
        super().__init__()
        hidden_size = config.decoder_hidden_sizes[-1]
        self.layers = nn.ModuleList(
            [
                ZacConvNeXtBlock(hidden_size, config.head_intermediate_size, config.convnext_norm_eps)
                for _ in range(config.head_num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=config.convnext_norm_eps)
        self.branches = nn.ModuleList([ZacISTFTBranch(config, n_fft) for n_fft in config.head_n_ffts])
        kernel_size = config.head_mixer_kernel_size
        self.mixer = nn.Conv1d(len(config.head_n_ffts), 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states.transpose(1, 2))
        audio = torch.stack([branch(hidden_states) for branch in self.branches], dim=1)
        return self.mixer(audio.to(self.mixer.weight.dtype))


def _sinc_resample_kernel(
    orig_freq: int, new_freq: int, lowpass_filter_width: int, rolloff: float
) -> tuple[torch.Tensor, int]:
    """
    The `sinc_interp_hann` kernel of `torchaudio.transforms.Resample` (built in float64, returned in float32), for
    rates already divided by their gcd. Returns the kernel `[new_freq, 1, 2 * width + orig_freq]` and `width`.
    Adapted from torchaudio (BSD-2-Clause, Copyright (c) 2017 Facebook Inc.).
    """
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    # torchaudio builds the phase offsets in float32 and the taps in float64
    t = torch.arange(0, -new_freq, -1, dtype=torch.float32)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernel *= window * (base_freq / orig_freq)
    return kernel.to(torch.float32), width


class ZacResampler(nn.Module):
    """Sinc resampler with an integer-exact output length `ceil(length * new_freq / orig_freq)`, in float32."""

    def __init__(self, orig_freq: int, new_freq: int, lowpass_filter_width: int, rolloff: float):
        super().__init__()
        gcd = math.gcd(orig_freq, new_freq)
        self.orig_freq = orig_freq // gcd
        self.new_freq = new_freq // gcd
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        kernel, self.width = _sinc_resample_kernel(self.orig_freq, self.new_freq, lowpass_filter_width, rolloff)
        self.kernel = nn.Buffer(kernel, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shape = waveform.shape
        waveform = waveform.reshape(-1, shape[-1]).float()
        waveform = F.pad(waveform, (self.width, self.width + self.orig_freq))
        resampled = F.conv1d(waveform[:, None], self.kernel.float(), stride=self.orig_freq)
        resampled = resampled.transpose(1, 2).reshape(waveform.shape[0], -1)
        target_length = -(-shape[-1] * self.new_freq // self.orig_freq)
        return resampled[:, :target_length].reshape(*shape[:-1], target_length)


def average_ssl_frames(features: torch.Tensor, num_frames: int, window_ratio: float) -> torch.Tensor:
    """
    Box averages of the teacher features `[batch, teacher_frames, channels]` onto `num_frames` code frames, in float32:
    each window is `window_ratio` code frames wide, centered on its frame, clipped at the edges and integrated exactly
    (fractional teacher frames contribute in proportion to their overlap). Returns `[batch, channels, num_frames]`.
    Adapted from ZAC-1 `zac/semantic.py::average_ssl_frames` (Apache-2.0, Copyright Zyphra).
    """
    values = features.float().transpose(1, 2)
    teacher_frames = values.shape[-1]
    ratio = teacher_frames / num_frames
    centers = (torch.arange(num_frames, device=features.device, dtype=torch.float32) + 0.5) * ratio
    half_width = 0.5 * window_ratio * ratio
    left = (centers - half_width).clamp(0, teacher_frames)
    right = (centers + half_width).clamp(0, teacher_frames)
    prefix = F.pad(values.cumsum(dim=-1), (1, 0))

    def integral(position):
        index = position.floor().long()
        fraction = position - index
        return prefix.index_select(-1, index) + values.index_select(-1, index.clamp(max=teacher_frames - 1)) * fraction

    return (integral(right) - integral(left)) / (right - left)


class ZacSemanticEncoder(nn.Module):
    def __init__(self, config: ZacConfig):
        super().__init__()
        self.in_proj = nn.Conv1d(config.semantic_model_config.hidden_size, config.semantic_hidden_size, 1)
        self.layers = nn.ModuleList(
            [
                ZacConvNeXtBlock(
                    config.semantic_hidden_size, config.semantic_intermediate_size, config.convnext_norm_eps
                )
                for _ in range(config.semantic_num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_proj(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ZacSemanticHead(nn.Module):
    def __init__(self, config: ZacConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ZacConvNeXtBlock(
                    config.semantic_hidden_size, config.semantic_intermediate_size, config.convnext_norm_eps
                )
                for _ in range(config.semantic_num_layers)
            ]
        )
        self.out_proj = nn.Conv1d(config.semantic_hidden_size, config.hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.out_proj(hidden_states)


class ZacEncoder(nn.Module):
    """Causal convolutional front end followed by patched transformer stages: internal-rate audio -> code frames."""

    def __init__(self, config: ZacConfig):
        super().__init__()
        self.conv_in = ZacCausalConv1d(1, config.encoder_hidden_size, kernel_size=7)
        self.blocks = nn.ModuleList(
            [
                ZacEncoderBlock(config, stride, stride_index + 1)
                for stride_index, stride in enumerate(config.encoder_strides)
            ]
        )
        self.patch_sizes = list(config.encoder_patch_sizes)
        self.stages = nn.ModuleList(
            [ZacTransformerStage(config, stage_idx, is_decoder=False) for stage_idx in range(len(self.patch_sizes))]
        )
        self.rotary_emb = ZacRotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        for patch_size, stage in zip(self.patch_sizes, self.stages):
            batch_size, channels, length = hidden_states.shape
            hidden_states = (
                hidden_states.reshape(batch_size, channels, length // patch_size, patch_size)
                .permute(0, 1, 3, 2)
                .reshape(batch_size, channels * patch_size, length // patch_size)
            )
            hidden_states = stage(hidden_states, self.rotary_emb, **kwargs)
        return hidden_states


class ZacDecoder(nn.Module):
    """Transformer stages with un-patching in between, then the multi-resolution iSTFT head."""

    def __init__(self, config: ZacConfig):
        super().__init__()
        self.unpatch_size = config.decoder_unpatch_size
        self.stages = nn.ModuleList(
            [
                ZacTransformerStage(config, stage_idx, is_decoder=True)
                for stage_idx in range(len(config.decoder_hidden_sizes))
            ]
        )
        self.rotary_emb = ZacRotaryEmbedding(config)
        self.head = ZacISTFTHead(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                batch_size, channels, length = hidden_states.shape
                size = self.unpatch_size
                hidden_states = (
                    hidden_states.reshape(batch_size, channels // size, size, length)
                    .permute(0, 1, 3, 2)
                    .reshape(batch_size, channels // size, size * length)
                )
            hidden_states = stage(hidden_states, self.rotary_emb, **kwargs)
        return self.head(hidden_states)


@auto_docstring
class ZacPreTrainedModel(PreTrainedAudioTokenizerBase):
    config: ZacConfig
    base_model_prefix = "zac"
    main_input_name = "input_values"
    input_modalities = ("audio",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["ZacTransformerLayer"]
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, ZacSnake1d):
            init.ones_(module.alpha)
        elif isinstance(module, ZacISTFTBranch):
            init.copy_(module.window, torch.hann_window(module.n_fft, periodic=True))
        elif isinstance(module, ZacResampler):
            kernel, _ = _sinc_resample_kernel(
                module.orig_freq, module.new_freq, module.lowpass_filter_width, module.rolloff
            )
            init.copy_(module.kernel, kernel)
        elif isinstance(module, ZacLayerScale):
            init.constant_(module.scale, self.config.layer_scale_initial_scale)


@auto_docstring(custom_intro="ZAC-1 (Zyphra Audio Codec 1), the 44.1 kHz speech codec of ZONOS3.")
class ZacModel(ZacPreTrainedModel):
    def __init__(self, config: ZacConfig):
        super().__init__(config)
        self.encoder = ZacEncoder(config)
        self.encoder_resampler = ZacResampler(
            config.sampling_rate,
            config.internal_sampling_rate,
            config.resample_lowpass_filter_width,
            config.resample_rolloff,
        )
        self.semantic_model = AutoModel.from_config(config.semantic_model_config)
        self.semantic_encoder = ZacSemanticEncoder(config)
        self.semantic_quantizer = ZacResidualVectorQuantizer(config, is_semantic=True)
        self.semantic_head = ZacSemanticHead(config)
        self.acoustic_quantizer = ZacResidualVectorQuantizer(config, is_semantic=False)
        self.decoder = ZacDecoder(config)
        self.decoder_resampler = ZacResampler(
            config.internal_sampling_rate,
            config.sampling_rate,
            config.resample_lowpass_filter_width,
            config.resample_rolloff,
        )
        self.post_init()

    def _encode_lane(
        self,
        audio: torch.Tensor,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Codes `[1, num_quantizers, lane_frames]` of one lane of internal-rate audio `[1, 1, samples]`."""
        acoustic = self.encoder(audio, **kwargs)
        ssl = self.semantic_model(input_features, attention_mask=input_features_mask).last_hidden_state
        pooled = average_ssl_frames(ssl, acoustic.shape[-1], self.config.semantic_window_ratio)
        semantic = self.semantic_encoder(pooled.to(acoustic.dtype))
        semantic_quantized, semantic_codes = self.semantic_quantizer(semantic)[:2]
        residual = acoustic - self.semantic_head(semantic_quantized)
        acoustic_codes = self.acoustic_quantizer(residual)[1]
        return torch.cat([semantic_codes, acoustic_codes], dim=1)

    @auto_docstring
    @can_return_tuple
    def encode(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ZacEncoderOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
            Audio at `config.sampling_rate`, padded to a multiple of `config.hop_length` (see [`ZacFeatureExtractor`]).
        input_features (`torch.FloatTensor` of shape `(batch_size, num_lanes, num_frames, 160)`):
            Stacked log-mel features of each encoder lane for the Wav2Vec2-BERT teacher (see [`ZacFeatureExtractor`]).
        padding_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask of the valid samples of `input_values`.
        input_features_mask (`torch.LongTensor` of shape `(batch_size, num_lanes, num_frames)`, *optional*):
            Mask of the valid frames of `input_features`.
        """
        config = self.config
        batch_size, _, sequence_length = input_values.shape
        if padding_mask is None:
            lengths = [sequence_length] * batch_size
        else:
            lengths = padding_mask.sum(-1).tolist()
        num_frames = [-(-int(length) // config.hop_length) for length in lengths]
        audio_codes = torch.zeros(
            (batch_size, config.num_quantizers, max(num_frames)), dtype=torch.long, device=input_values.device
        )
        audio_codes_mask = torch.zeros((batch_size, max(num_frames)), dtype=torch.long, device=input_values.device)
        internal_hop = config.internal_hop_length
        for batch_idx, (length, frames) in enumerate(zip(lengths, num_frames)):
            internal = self.encoder_resampler(input_values[batch_idx, :, : int(length)]).to(input_values.dtype)
            internal = F.pad(internal, (0, frames * internal_hop - internal.shape[-1]))
            # Lanes of `lane_frames` code frames: every lane after the first re-reads `lane_context_frames` frames
            start, lane_idx = 0, 0
            while start < frames:
                context = 0 if start == 0 else config.lane_context_frames
                count = min(frames - start, config.lane_frames - context)
                lane = internal[:, (start - context) * internal_hop : (start + count) * internal_hop]
                lane = F.pad(lane, (0, config.lane_frames * internal_hop - lane.shape[-1]))
                lane_codes = self._encode_lane(
                    lane[None].to(self.dtype),
                    input_features[batch_idx, lane_idx][None].to(self.semantic_model.dtype),
                    None if input_features_mask is None else input_features_mask[batch_idx, lane_idx][None],
                    **kwargs,
                )
                audio_codes[batch_idx, :, start : start + count] = lane_codes[0, :, context : context + count]
                start += count
                lane_idx += 1
            audio_codes_mask[batch_idx, :frames] = 1

        return ZacEncoderOutput(audio_codes=audio_codes, audio_codes_mask=audio_codes_mask)

    @auto_docstring
    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ZacDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_quantizers, codes_length)`):
            Discrete codes computed with `model.encode`, in `[0, config.codebook_size)`.
        """
        if bool(((audio_codes < 0) | (audio_codes >= self.config.codebook_size)).any()):
            raise ValueError(f"`audio_codes` must be in [0, {self.config.codebook_size}).")
        semantic = self.semantic_quantizer.from_codes(audio_codes[:, :1])[0]
        latents = self.semantic_head(semantic) + self.acoustic_quantizer.from_codes(audio_codes[:, 1:])[0]
        audio_values = self.decoder_resampler(self.decoder(latents, **kwargs))
        audio_values = audio_values[..., : audio_codes.shape[-1] * self.config.hop_length]
        return ZacDecoderOutput(audio_values=audio_values)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_values: torch.Tensor,
        input_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ZacOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
            Audio at `config.sampling_rate`, padded to a multiple of `config.hop_length` (see [`ZacFeatureExtractor`]).
        input_features (`torch.FloatTensor` of shape `(batch_size, num_lanes, num_frames, 160)`):
            Stacked log-mel features of each encoder lane for the Wav2Vec2-BERT teacher (see [`ZacFeatureExtractor`]).
        padding_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask of the valid samples of `input_values`.
        input_features_mask (`torch.LongTensor` of shape `(batch_size, num_lanes, num_frames)`, *optional*):
            Mask of the valid frames of `input_features`.

        Examples:

        ```python
        >>> from datasets import Audio, load_dataset
        >>> from transformers import AutoFeatureExtractor, AutoModel

        >>> model_id = "Zyphra/ZAC-1-hf"  # TODO: move to final Zyphra repo
        >>> model = AutoModel.from_pretrained(model_id)
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
        >>> audio = dataset[0]["audio"]["array"]

        >>> inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        encoder_outputs = self.encode(
            input_values,
            input_features,
            padding_mask=padding_mask,
            input_features_mask=input_features_mask,
            **kwargs,
        )
        audio_values = self.decode(encoder_outputs.audio_codes, **kwargs).audio_values
        return ZacOutput(
            audio_values=audio_values[..., : input_values.shape[-1]],
            audio_codes=encoder_outputs.audio_codes,
            audio_codes_mask=encoder_outputs.audio_codes_mask,
        )


__all__ = ["ZacConfig", "ZacModel", "ZacPreTrainedModel"]
