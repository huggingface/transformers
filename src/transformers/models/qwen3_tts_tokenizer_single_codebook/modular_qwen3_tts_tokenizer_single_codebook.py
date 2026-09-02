# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Modular Qwen3-TTS single-codebook tokenizer."""

from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..cohere.modeling_cohere import CohereRotaryEmbedding
from ..qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniToken2WavConfig,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    DiTAttention,
    DiTDecoderLayer,
    Qwen2_5OmniAMPBlock,
    Qwen2_5OmniAntiAliasedActivation1d,
    Qwen2_5OmniSnakeBeta,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
    Qwen2_5OmniToken2WavModel,
    apply_rotary_pos_emb,
)
from ..qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention, Qwen2AudioEncoder, Qwen2AudioEncoderLayer
from ..voxtral_realtime.modeling_voxtral_realtime import VoxtralRealtimeCausalConv1d
from ..xcodec.modeling_xcodec import XcodecEuclideanCodebook, XcodecVectorQuantization


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookDiTConfig(Qwen2_5OmniDiTConfig):
    model_type = "qwen3_tts_tokenizer_single_codebook_decoder_dit"
    base_config_key = "dit_config"


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig(Qwen2_5OmniBigVGANConfig):
    r"""
    mel_dim (`int`, *optional*, defaults to 80):
        The dimension of the mel-spectrogram.
    upsample_initial_channel (`int`, *optional*, defaults to 1536):
        The number of channels in the initial upsampling layer.
    resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
        A list of kernel sizes for each residual block.
    resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
        A list of dilation sizes for each residual block.
    upsample_rates (`list[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
        A list of upsampling rates for each upsampling layer.
    upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
        A list of kernel sizes for each upsampling layer.
    conv_pre_kernel_size (`int`, *optional*, defaults to 5):
        Kernel size of the vocoder input convolution.
    conv_pre_stride (`int`, *optional*, defaults to 1):
        Stride of the vocoder input convolution.
    conv_pre_padding (`int`, *optional*, defaults to 2):
        Padding of the vocoder input convolution.
    resblock_causal_modes (`list[str]`, *optional*):
        Per-upsample residual-block mode. `"full_causal"` uses causal convolutions throughout.
        `"hybrid"` uses causal `convs1` and symmetric `convs2`.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_decoder_bigvgan"
    base_config_key = "bigvgan_config"

    conv_pre_kernel_size: int = 5
    conv_pre_stride: int = 1
    conv_pre_padding: int = 2
    resblock_causal_modes: list[str] | tuple[str, ...] = (
        "full_causal",
        "full_causal",
        "hybrid",
        "hybrid",
        "hybrid",
        "hybrid",
    )


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookDecoderConfig(Qwen2_5OmniToken2WavConfig):
    model_type = "qwen3_tts_tokenizer_single_codebook_decoder"
    sub_configs = {
        "dit_config": Qwen3TTSTokenizerSingleCodebookDiTConfig,
        "bigvgan_config": Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig,
    }

    def __post_init__(self, **kwargs):
        if self.dit_config is None:
            self.dit_config = Qwen3TTSTokenizerSingleCodebookDiTConfig()
        elif isinstance(self.dit_config, dict):
            self.dit_config = Qwen3TTSTokenizerSingleCodebookDiTConfig(**self.dit_config)

        if self.bigvgan_config is None:
            self.bigvgan_config = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig()
        elif isinstance(self.bigvgan_config, dict):
            self.bigvgan_config = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig(**self.bigvgan_config)

        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookEncoderConfig(Qwen2AudioEncoderConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    num_layers_before_quantizer (`int`, *optional*, defaults to 1):
        Number of encoder layers run before the sibling quantizer.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_encoder"
    base_config_key = "encoder_config"
    attribute_map = {
        "num_hidden_layers": "encoder_layers",
        "d_model": "hidden_size",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    encoder_layers: int = 1
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    hidden_size: int = 1024
    num_layers_before_quantizer: int = 1


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookQuantizerConfig(PreTrainedConfig):
    r"""
    hidden_size (`int`, *optional*, defaults to 1024):
        Encoder hidden size entering the quantizer.
    codebook_size (`int`, *optional*, defaults to 512):
        Number of vectors in the single codebook.
    codebook_dim (`int`, *optional*, defaults to 512):
        Dimension of each codebook vector.
    downsample_rate (`int`, *optional*, defaults to 2):
        Stride of the convolution applied before quantization.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_quantizer"
    base_config_key = "quantizer_config"

    hidden_size: int = 1024
    codebook_size: int = 512
    codebook_dim: int = 512
    downsample_rate: int = 2


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict`, *optional*):
        Configuration of the Whisper-family encoder.
    quantizer_config (`dict`, *optional*):
        Configuration of the sibling vector quantizer.
    decoder_config (`dict`, *optional*):
        Configuration of the DiT and BigVGAN decoder.
    input_sample_rate (`int`, *optional*, defaults to 24000):
        Sample rate of the input audio.
    output_sample_rate (`int`, *optional*, defaults to 24000):
        Sample rate of the decoded waveform.
    encode_downsample_rate (`int`, *optional*, defaults to 200):
        Frames of input audio represented by one code.
    decode_upsample_rate (`int`, *optional*, defaults to 200):
        Samples of output audio produced from one code.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook"
    sub_configs = {
        "encoder_config": Qwen3TTSTokenizerSingleCodebookEncoderConfig,
        "quantizer_config": Qwen3TTSTokenizerSingleCodebookQuantizerConfig,
        "decoder_config": Qwen3TTSTokenizerSingleCodebookDecoderConfig,
    }

    encoder_config: dict | PreTrainedConfig | None = None
    quantizer_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    encode_downsample_rate: int = 200
    decode_upsample_rate: int = 200

    def __post_init__(self, **kwargs):
        if self.encoder_config is None:
            self.encoder_config = Qwen3TTSTokenizerSingleCodebookEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = Qwen3TTSTokenizerSingleCodebookEncoderConfig(**self.encoder_config)

        if self.quantizer_config is None:
            self.quantizer_config = Qwen3TTSTokenizerSingleCodebookQuantizerConfig()
        elif isinstance(self.quantizer_config, dict):
            self.quantizer_config = Qwen3TTSTokenizerSingleCodebookQuantizerConfig(**self.quantizer_config)

        if self.decoder_config is None:
            self.decoder_config = Qwen3TTSTokenizerSingleCodebookDecoderConfig()
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = Qwen3TTSTokenizerSingleCodebookDecoderConfig(**self.decoder_config)

        super().__post_init__(**kwargs)


@auto_docstring
class Qwen3TTSTokenizerSingleCodebookPreTrainedModel(PreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    _supports_sdpa = True


@auto_docstring
@dataclass
class Qwen3TTSTokenizerSingleCodebookEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length)`):
        Discrete speech codes.
    audio_codes_mask (`torch.Tensor` of shape `(batch_size, codes_length)`, *optional*):
        Mask over valid codes. `1` is a real frame.
    """

    audio_codes: torch.LongTensor | None = None
    audio_codes_mask: torch.Tensor | None = None


@auto_docstring
@dataclass
class Qwen3TTSTokenizerSingleCodebookDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Decoded waveform.
    """

    audio_values: torch.FloatTensor | None = None


class Qwen3TTSTokenizerSingleCodebookAttention(Qwen2AudioAttention):
    pass


class Qwen3TTSTokenizerSingleCodebookEncoderLayer(Qwen2AudioEncoderLayer):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)
        self.self_attn = Qwen3TTSTokenizerSingleCodebookAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )


class Qwen3TTSTokenizerSingleCodebookEncoder(Qwen2AudioEncoder):
    config: Qwen3TTSTokenizerSingleCodebookEncoderConfig
    config_class = Qwen3TTSTokenizerSingleCodebookEncoderConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _no_split_modules = ["Qwen3TTSTokenizerSingleCodebookEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTokenizerSingleCodebookEncoderLayer,
        "attentions": Qwen3TTSTokenizerSingleCodebookAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookEncoderLayer(config)
                for _ in range(min(config.encoder_layers, config.num_layers_before_quantizer))
            ]
        )
        self.avg_pooler = nn.Identity()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        after_cnn = (input_lengths - 1) // 2 + 1
        return after_cnn, after_cnn

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features,
        attention_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        attention_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*):
            Mask of valid log-mel frames. Padding frames are ignored after the convolutional stem.
        """
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        seq_len = inputs_embeds.size(1)
        if seq_len > self.max_source_positions:
            raise ValueError(
                f"Encoder sequence length {seq_len} exceeds `max_source_positions` ({self.max_source_positions})."
            )
        hidden_states = inputs_embeds + self.embed_positions.weight[:seq_len]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_attention_mask = None
        if attention_mask is not None:
            after_cnn = self._get_feat_extract_output_lengths(attention_mask.long().sum(-1))[0]
            feature_mask = torch.arange(seq_len, device=hidden_states.device)[None, :] < after_cnn[:, None]
            encoder_attention_mask = create_bidirectional_mask(self.config, hidden_states, feature_mask.long())

        num_layers = self.config.num_layers_before_quantizer
        if num_layers < 1:
            raise ValueError("`num_layers_before_quantizer` must be at least 1.")
        for layer_idx, encoder_layer in enumerate(self.layers):
            if layer_idx >= num_layers:
                break
            hidden_states = encoder_layer(hidden_states, encoder_attention_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3TTSTokenizerSingleCodebookEuclideanCodebook(XcodecEuclideanCodebook):
    pass


class Qwen3TTSTokenizerSingleCodebookVectorQuantization(XcodecVectorQuantization):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookQuantizerConfig):
        nn.Module.__init__(self)
        requires_projection = config.codebook_dim != config.hidden_size
        self.project_in = nn.Linear(config.hidden_size, config.codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(config.codebook_dim, config.hidden_size) if requires_projection else nn.Identity()
        self.codebook = Qwen3TTSTokenizerSingleCodebookEuclideanCodebook(config)

    def encode(self, hidden_states):
        hidden_states = self.project_in(hidden_states)
        return self.codebook.encode(hidden_states)

    def decode(self, embed_ind):
        return self.project_out(self.codebook.decode(embed_ind))


class Qwen3TTSTokenizerSingleCodebookQuantizer(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookQuantizerConfig

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookQuantizerConfig):
        super().__init__(config)
        stride = config.downsample_rate
        if stride > 1:
            self.downsample = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=stride, stride=stride)
        else:
            self.downsample = nn.Identity()
        self.vq = Qwen3TTSTokenizerSingleCodebookVectorQuantization(config)
        self.post_init()

    def encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        hidden_states = self.downsample(hidden_states.transpose(1, 2)).transpose(1, 2)
        audio_codes = self.vq.encode(hidden_states)
        audio_codes_mask = None
        if attention_mask is not None:
            code_lengths = attention_mask.long().sum(-1)
            # CODEPATH: 25 Hz tokenizer uses downsample_rate=2; rate 1 would skip this integer divide
            if self.config.downsample_rate > 1:
                code_lengths = code_lengths // self.config.downsample_rate
            seq_len = audio_codes.size(1)
            audio_codes_mask = torch.arange(seq_len, device=audio_codes.device)[None, :] < code_lengths[:, None]
        return audio_codes, audio_codes_mask


class CausalConv1d(VoxtralRealtimeCausalConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = ""


class Qwen3TTSTokenizerSingleCodebookSnakeBeta(Qwen2_5OmniSnakeBeta):
    pass


class Qwen3TTSTokenizerSingleCodebookAntiAliasedActivation1d(Qwen2_5OmniAntiAliasedActivation1d):
    pass


class Qwen3TTSTokenizerSingleCodebookAMPBlock(Qwen2_5OmniAMPBlock):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), causal_mode="hybrid"):
        nn.Module.__init__(self)
        if causal_mode not in {"hybrid", "full_causal"}:
            raise ValueError(f"Unsupported causal_mode={causal_mode}. Use 'hybrid' or 'full_causal'.")

        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[0]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[1]),
                CausalConv1d(channels, channels, kernel_size, 1, dilation=dilation[2]),
            ]
        )
        if causal_mode == "hybrid":
            self.convs2 = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
                    )
                    for _ in range(3)
                ]
            )
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()
        else:
            self.convs2 = nn.ModuleList(
                [CausalConv1d(channels, channels, kernel_size, 1, dilation=1) for _ in range(3)]
            )
            self.pre_conv = nn.Conv1d(
                channels, channels, kernel_size, stride=1, padding=self._get_padding(kernel_size, 1)
            )
            self.pre_act = Qwen3TTSTokenizerSingleCodebookAntiAliasedActivation1d(
                activation=Qwen3TTSTokenizerSingleCodebookSnakeBeta(channels)
            )

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookAntiAliasedActivation1d(
                    activation=Qwen3TTSTokenizerSingleCodebookSnakeBeta(channels)
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.pre_act(self.pre_conv(hidden_states))
        return super().forward(hidden_states)


class Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel(Qwen2_5OmniToken2WavBigVGANModel):
    config: Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig):
        super().__init__(config)
        self.conv_pre = nn.Conv1d(
            config.mel_dim,
            config.upsample_initial_channel,
            config.conv_pre_kernel_size,
            config.conv_pre_stride,
            padding=config.conv_pre_padding,
        )
        self.resblocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookAMPBlock(
                    config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                    kernel_size,
                    dilation,
                    config.resblock_causal_modes[layer_idx],
                )
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )


class Qwen3TTSTokenizerSingleCodebookDiTRotaryEmbedding(CohereRotaryEmbedding):
    pass


class Qwen3TTSTokenizerSingleCodebookDiTAttention(DiTAttention):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )
        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_output = self.to_out[0](attention_weights.to(query.dtype))
        attention_output = self.to_out[1](attention_output)
        return attention_output


class Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(DiTDecoderLayer):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__(config, look_ahead_block=look_ahead_block, look_backward_block=look_backward_block)
        self.attn = Qwen3TTSTokenizerSingleCodebookDiTAttention(config)


class Qwen3TTSTokenizerSingleCodebookDecoderDiTModel(Qwen2_5OmniToken2WavDiTModel):
    config: Qwen3TTSTokenizerSingleCodebookDiTConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDiTConfig
    _no_split_modules = ["Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDiTConfig):
        super().__init__(config)
        self.rotary_embed = Qwen3TTSTokenizerSingleCodebookDiTRotaryEmbedding(config)
        self.transformer_blocks = nn.ModuleList(
            [
                Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(
                    config,
                    look_ahead_block=1
                    if i in config.look_ahead_layers
                    else 0,  # CODEPATH: Omni Token2Wav DiT look-ahead layers
                    look_backward_block=1
                    if i in config.look_backward_layers
                    else 0,  # CODEPATH: Omni Token2Wav DiT look-backward layers
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        noise_initialization = torch.randn(
            [quantized_code.shape[0], 30000, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
            device=quantized_code.device,
        )
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                return self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                    apply_cfg=False,
                )
            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        time_embedding = torch.linspace(0, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype)
        if sway_coefficient is not None:
            time_embedding = time_embedding + sway_coefficient * (
                torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding
            )

        values = initial_state.clone()
        for t0, t1 in zip(time_embedding[:-1], time_embedding[1:]):
            values = values + ode_function(t0, values) * (t1 - t0)
        return values.permute(0, 2, 1)


class Qwen3TTSTokenizerSingleCodebookDecoder(Qwen2_5OmniToken2WavModel):
    config: Qwen3TTSTokenizerSingleCodebookDecoderConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDecoderConfig
    _no_split_modules = [
        "Qwen3TTSTokenizerSingleCodebookDecoderDiTModel",
        "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel",
    ]

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDecoderConfig):
        PreTrainedModel.__init__(self, config)
        self.dit = Qwen3TTSTokenizerSingleCodebookDecoderDiTModel._from_config(
            config.dit_config, attn_implementation="sdpa"
        )
        self.bigvgan = Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation="sdpa"
        )
        self.post_init()

    @auto_docstring
    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        r"""
        code (`torch.LongTensor`):
            Discrete speech codes.
        conditioning (`torch.FloatTensor`):
            Speaker conditioning vector for the DiT sampler.
        reference_mel (`torch.FloatTensor`):
            Reference mel spectrogram for the DiT sampler.
        """
        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        return self.bigvgan(mel_spectrogram)


@auto_docstring(
    custom_intro="""
    Qwen3-TTS single-codebook tokenizer: Whisper-family encoder, sibling quantizer, and DiT/BigVGAN decoder.
    """
)
class Qwen3TTSTokenizerSingleCodebookModel(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookConfig):
        super().__init__(config)
        self.encoder = Qwen3TTSTokenizerSingleCodebookEncoder._from_config(config.encoder_config)
        self.quantizer = Qwen3TTSTokenizerSingleCodebookQuantizer._from_config(config.quantizer_config)
        self.decoder = Qwen3TTSTokenizerSingleCodebookDecoder._from_config(config.decoder_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Qwen3TTSTokenizerSingleCodebookEncoderOutput:
        encoder_outputs = self.encoder(input_features, attention_mask=input_features_mask)
        encoder_mask = None
        if input_features_mask is not None:
            after_cnn = self.encoder._get_feat_extract_output_lengths(input_features_mask.long().sum(-1))[0]
            seq_len = encoder_outputs.last_hidden_state.size(1)
            encoder_mask = torch.arange(seq_len, device=input_features.device)[None, :] < after_cnn[:, None]
        audio_codes, audio_codes_mask = self.quantizer.encode(
            encoder_outputs.last_hidden_state, attention_mask=encoder_mask
        )
        return Qwen3TTSTokenizerSingleCodebookEncoderOutput(audio_codes=audio_codes, audio_codes_mask=audio_codes_mask)

    @can_return_tuple
    @auto_docstring
    def decode(
        self,
        audio_codes: torch.LongTensor,
        xvectors: torch.FloatTensor,
        ref_mels: torch.FloatTensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
        **kwargs,
    ) -> Qwen3TTSTokenizerSingleCodebookDecoderOutput:
        audio_values = self.decoder(
            code=audio_codes,
            conditioning=xvectors,
            reference_mel=ref_mels,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        return Qwen3TTSTokenizerSingleCodebookDecoderOutput(audio_values=audio_values)


__all__ = [
    "Qwen3TTSTokenizerSingleCodebookConfig",
    "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig",
    "Qwen3TTSTokenizerSingleCodebookDecoderConfig",
    "Qwen3TTSTokenizerSingleCodebookDiTConfig",
    "Qwen3TTSTokenizerSingleCodebookEncoderConfig",
    "Qwen3TTSTokenizerSingleCodebookQuantizerConfig",
    "Qwen3TTSTokenizerSingleCodebookPreTrainedModel",
    "Qwen3TTSTokenizerSingleCodebookEncoder",
    "Qwen3TTSTokenizerSingleCodebookQuantizer",
    "Qwen3TTSTokenizerSingleCodebookDecoderDiTModel",
    "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel",
    "Qwen3TTSTokenizerSingleCodebookDecoder",
    "Qwen3TTSTokenizerSingleCodebookModel",
]
