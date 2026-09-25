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

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedAudioTokenizerBase, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, get_max_seqlen, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..cohere.modeling_cohere import CohereRotaryEmbedding, apply_rotary_pos_emb
from ..qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniToken2WavConfig,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import (
    DiTAttention,
    DiTCodecEmbedding,
    DiTDecoderLayer,
    DiTInputEmbedding,
    DiTTimestepEmbedding,
    Qwen2_5_OmniAdaLayerNormZero_Final,
    Qwen2_5OmniAMPBlock,
    Qwen2_5OmniAntiAliasedActivation1d,
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniAudioEncoder,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniDownSample1d,
    Qwen2_5OmniSnakeBeta,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
    Qwen2_5OmniToken2WavModel,
    Qwen2_5OmniUpSample1d,
    SinusoidsPositionEmbedding,
    chunk_and_pad_features,
    get_audio_cu_seqlens,
    get_valid_indices,
    kaiser_sinc_filter1d,
)
from ..voxtral_realtime.modeling_voxtral_realtime import VoxtralRealtimeCausalConv1d
from ..xcodec.modeling_xcodec import XcodecEuclideanCodebook, XcodecVectorQuantization


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookDiTConfig(Qwen2_5OmniDiTConfig):
    r"""
    ff_mult (`int`, *optional*, defaults to 2):
        The multiplier for the feedforward layer in each transformer block.
    emb_dim (`int`, *optional*, defaults to 512):
        The dimension of the codec embedding layer.
    block_size (`int`, *optional*, defaults to 24):
        Number of mel frames in each block of the block-causal attention mask.
    look_ahead_layers (`list[int]`, *optional*, defaults to `[10]`):
        Indices of the transformer layers that may attend to the next block.
    look_backward_layers (`list[int]`, *optional*, defaults to `[0, 20]`):
        Indices of the transformer layers that may attend to the previous block.
    repeats (`int`, *optional*, defaults to 2):
        Number of mel frames generated per speech code.
    num_embeds (`int`, *optional*, defaults to 8193):
        The number of unique embeddings in the codec.
    mel_dim (`int`, *optional*, defaults to 80):
        The dimension of the mel-spectrogram.
    enc_emb_dim (`int`, *optional*, defaults to 192):
        The dimension of the speaker embedding (`xvectors`) passed to the decoder.
    enc_dim (`int`, *optional*, defaults to 128):
        The output dimension of the reference-mel speaker encoder.
    enc_channels (`list[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
        A list of output channels for each TDNN/SERes2Net layer in the reference-mel speaker encoder.
    enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
        A list of kernel sizes for each layer in the reference-mel speaker encoder.
    enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
        A list of dilations for each layer in the reference-mel speaker encoder.
    enc_attention_channels (`int`, *optional*, defaults to 64):
        The number of attention channels in the SqueezeExcitationBlock.
    enc_res2net_scale (`int`, *optional*, defaults to 2):
        The scale of the Res2Net block in the reference-mel speaker encoder.
    enc_se_channels (`int`, *optional*, defaults to 64):
        The number of output channels after squeeze in the SqueezeExcitationBlock.
    """

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
    r"""
    dit_config (`dict`, *optional*):
        Configuration of the diffusion transformer that generates mel-spectrograms from speech codes.
    bigvgan_config (`dict`, *optional*):
        Configuration of the BigVGAN vocoder that turns mel-spectrograms into a waveform.
    """

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
class Qwen3TTSTokenizerSingleCodebookEncoderConfig(PreTrainedConfig):
    r"""
    encoder_layers (`int`, *optional*, defaults to 6):
        Number of transformer layers kept from the Whisper-style encoder. The original checkpoint has more
        layers, but only the layers that precede the quantizer take part in tokenization.
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    n_window (`int`, *optional*, defaults to 100):
        Number of post-convolution frames per attention window. Frames only attend within their own window.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_encoder"
    base_config_key = "encoder_config"
    attribute_map = {
        "d_model": "hidden_size",
        "num_hidden_layers": "encoder_layers",
        "num_attention_heads": "encoder_attention_heads",
        "intermediate_size": "encoder_ffn_dim",
    }

    num_mel_bins: int = 128
    encoder_layers: int = 6
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    hidden_size: int = 1280
    dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    activation_function: str = "gelu"
    activation_dropout: float | int = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 100


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookQuantizerConfig(PreTrainedConfig):
    r"""
    codebook_size (`int`, *optional*, defaults to 32768):
        Number of vectors in the single codebook.
    codebook_dim (`int`, *optional*, defaults to 1280):
        Dimension of each codebook vector. A projection is added when it differs from `hidden_size`.
    downsample_rate (`int`, *optional*, defaults to 2):
        Stride of the convolution applied before quantization.
    """

    model_type = "qwen3_tts_tokenizer_single_codebook_quantizer"
    base_config_key = "quantizer_config"

    hidden_size: int = 1280
    codebook_size: int = 32768
    codebook_dim: int = 1280
    downsample_rate: int = 2


@auto_docstring
@strict
class Qwen3TTSTokenizerSingleCodebookConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict`, *optional*):
        Configuration of the Whisper-style encoder.
    quantizer_config (`dict`, *optional*):
        Configuration of the vector quantizer.
    decoder_config (`dict`, *optional*):
        Configuration of the DiT and BigVGAN decoder.
    input_sample_rate (`int`, *optional*, defaults to 16000):
        Sample rate of the audio the encoder log-mel features are computed from.
    output_sample_rate (`int`, *optional*, defaults to 24000):
        Sample rate of the decoded waveform.
    encode_downsample_rate (`int`, *optional*, defaults to 640):
        Number of input audio samples represented by one speech code.
    decode_upsample_rate (`int`, *optional*, defaults to 960):
        Number of output audio samples produced from one speech code.
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
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    encode_downsample_rate: int = 640
    decode_upsample_rate: int = 960

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
        Decoded waveform. Samples past each item's own duration are zero.
    """

    audio_values: torch.FloatTensor | None = None


class Qwen3TTSTokenizerSingleCodebookSinusoidsPositionEmbedding(SinusoidsPositionEmbedding):
    pass


class Qwen3TTSTokenizerSingleCodebookAudioAttention(Qwen2_5OmniAudioAttention):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)


class Qwen3TTSTokenizerSingleCodebookAudioEncoderLayer(Qwen2_5OmniAudioEncoderLayer):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)


class Qwen3TTSTokenizerSingleCodebookEuclideanCodebook(XcodecEuclideanCodebook):
    pass


class Qwen3TTSTokenizerSingleCodebookUpSample1d(Qwen2_5OmniUpSample1d):
    pass


class Qwen3TTSTokenizerSingleCodebookDownSample1d(Qwen2_5OmniDownSample1d):
    pass


@auto_docstring
class Qwen3TTSTokenizerSingleCodebookPreTrainedModel(PreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    input_modalities = "audio"
    _supports_sdpa = True

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Qwen3TTSTokenizerSingleCodebookSinusoidsPositionEmbedding):
            init.copy_(module.positional_embedding, module.compute_default_singular_positional_embedding())
        elif isinstance(module, Qwen3TTSTokenizerSingleCodebookUpSample1d):
            filter_tensor = kaiser_sinc_filter1d(0.5 / module.ratio, 0.6 / module.ratio, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, Qwen3TTSTokenizerSingleCodebookDownSample1d):
            filter_tensor = kaiser_sinc_filter1d(module.cutoff, module.half_width, module.kernel_size)
            init.copy_(module.filter, filter_tensor)
        elif isinstance(module, Qwen3TTSTokenizerSingleCodebookEuclideanCodebook):
            init.copy_(module.inited, torch.Tensor([True]))
            init.zeros_(module.cluster_size)
            init.zeros_(module.embed)
            init.zeros_(module.embed_avg)


@auto_docstring(
    custom_intro="""
    Whisper-style encoder of the Qwen3-TTS single-codebook tokenizer. Log-mel frames are processed in windows of
    `2 * n_window` frames: each window gets its own convolutional stem and positional embeddings, and attention never
    crosses a window boundary. Only the layers that precede the quantizer are kept.
    """
)
class Qwen3TTSTokenizerSingleCodebookEncoder(Qwen2_5OmniAudioEncoder):
    config: Qwen3TTSTokenizerSingleCodebookEncoderConfig
    config_class = Qwen3TTSTokenizerSingleCodebookEncoderConfig
    _no_split_modules = ["Qwen3TTSTokenizerSingleCodebookAudioEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTokenizerSingleCodebookAudioEncoderLayer,
        "attentions": Qwen3TTSTokenizerSingleCodebookAudioAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookEncoderConfig):
        super().__init__(config)
        self.positional_embedding = Qwen3TTSTokenizerSingleCodebookSinusoidsPositionEmbedding(
            config.max_source_positions, config.hidden_size
        )
        del self.audio_bos_eos_token
        del self.ln_post
        del self.avg_pooler
        del self.proj

    def _freeze_parameters(self):
        raise AttributeError("Not needed for Qwen3TTSTokenizerSingleCodebook")

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        raise AttributeError("Not needed for Qwen3TTSTokenizerSingleCodebook")

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Number of encoder frames produced from `input_lengths` log-mel frames by the stride-2 convolution."""
        return (input_lengths - 1) // 2 + 1

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of valid log-mel frames, `1` for a real frame and `0` for padding.
        """
        batch_size, _, num_frames = input_features.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, num_frames, dtype=torch.long, device=input_features.device)
        feature_lens = attention_mask.long().sum(-1)
        # Pack the valid frames of every item into one `(num_mel_bins, total_frames)` sequence.
        packed_features = input_features.transpose(1, 2)[attention_mask.bool()].transpose(0, 1)

        padded_feature, chunk_lengths = chunk_and_pad_features(
            packed_features, feature_lens, self.n_window, kwargs=kwargs
        )
        valid_indices = get_valid_indices(chunk_lengths, kwargs=kwargs)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, kwargs=kwargs)
        max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs=kwargs)

        padded_feature = padded_feature.to(self.conv1.weight.dtype)
        padded_mask = (
            (torch.arange(padded_feature.shape[2], device=padded_feature.device) < chunk_lengths.unsqueeze(1))
            .unsqueeze(1)
            .long()
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)
        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = torch.index_select(padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, **kwargs)[0]

        # Unpack to `(batch_size, max_frames_after_cnn, hidden_size)` for the quantizer.
        output_lengths = self._get_feat_extract_output_lengths(feature_lens)
        positions = torch.arange(self._get_feat_extract_output_lengths(num_frames), device=hidden_states.device)
        valid_positions = positions[None, :] < output_lengths[:, None]
        offsets = nn.functional.pad(output_lengths[:-1].cumsum(0), (1, 0))
        flat_indices = (offsets[:, None] + positions[None, :]).clamp(max=hidden_states.shape[0] - 1)
        hidden_states = hidden_states[flat_indices] * valid_positions.unsqueeze(-1).to(hidden_states.dtype)
        return BaseModelOutput(last_hidden_state=hidden_states)


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


@auto_docstring(
    custom_intro="""
    Single-codebook vector quantizer of the Qwen3-TTS single-codebook tokenizer. Encoder frames are downsampled by
    `downsample_rate` with a strided convolution and mapped to their nearest codebook entry.
    """
)
class Qwen3TTSTokenizerSingleCodebookQuantizer(Qwen3TTSTokenizerSingleCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerSingleCodebookQuantizerConfig

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookQuantizerConfig):
        super().__init__(config)
        # CODEPATH: Qwen3-TTS-Tokenizer-25Hz uses downsample_rate=2; rate 1 keeps the encoder frame rate as is
        if config.downsample_rate > 1:
            self.downsample = nn.Conv1d(
                config.hidden_size,
                config.hidden_size,
                kernel_size=config.downsample_rate,
                stride=config.downsample_rate,
            )
        else:
            self.downsample = nn.Identity()
        self.vq = Qwen3TTSTokenizerSingleCodebookVectorQuantization(config)
        self.post_init()

    @auto_docstring
    def encode(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        r"""
        hidden_states (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`):
            Encoder output.
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of valid encoder frames, `1` for a real frame and `0` for padding.
        """
        hidden_states = self.downsample(hidden_states.transpose(1, 2)).transpose(1, 2)
        audio_codes = self.vq.encode(hidden_states)
        audio_codes_mask = None
        if attention_mask is not None:
            code_lengths = attention_mask.long().sum(-1) // self.config.downsample_rate
            audio_codes_mask = (
                torch.arange(audio_codes.shape[1], device=audio_codes.device)[None, :] < code_lengths[:, None]
            )
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
        # The Qwen3-TTS vocoder chains the three convolution pairs and adds the output of every pair to the
        # block input, instead of restarting each pair from the running residual as BigVGAN does.
        residual = hidden_states
        hidden_states = self.pre_act(self.pre_conv(hidden_states))
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            hidden_states = conv1(act1(hidden_states))
            hidden_states = conv2(act2(hidden_states))
            residual = residual + hidden_states
        return residual


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

    @auto_docstring
    def forward(self, mel_spectrogram: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        mel_spectrogram (`torch.FloatTensor` of shape `(batch_size, mel_dim, num_frames)`):
            Log-mel spectrogram generated by the diffusion transformer.
        """
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            hidden_representation = residual_output / self.num_residual_blocks

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze(1)


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

        # Interleaved rotary embedding applied to every head.
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

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


class Qwen3TTSTokenizerSingleCodebookAdaLayerNormZeroFinal(Qwen2_5_OmniAdaLayerNormZero_Final):
    pass


class Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(DiTDecoderLayer):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__(config, look_ahead_block=look_ahead_block, look_backward_block=look_backward_block)
        self.attn = Qwen3TTSTokenizerSingleCodebookDiTAttention(config)


class Qwen3TTSTokenizerSingleCodebookDecoderDiTModel(Qwen2_5OmniToken2WavDiTModel):
    config: Qwen3TTSTokenizerSingleCodebookDiTConfig
    config_class = Qwen3TTSTokenizerSingleCodebookDiTConfig
    _no_split_modules = ["Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookDiTConfig):
        PreTrainedModel.__init__(self, config)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)
        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = DiTInputEmbedding(config)
        self.rotary_embed = Qwen3TTSTokenizerSingleCodebookDiTRotaryEmbedding(config)
        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads
        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # CODEPATH: Omni Token2Wav DiT look-ahead and look-backward layer sets
            self.transformer_blocks.append(
                Qwen3TTSTokenizerSingleCodebookDiTDecoderLayer(
                    config,
                    look_ahead_block=int(i in config.look_ahead_layers),
                    look_backward_block=int(i in config.look_backward_layers),
                )
            )
        self.norm_out = Qwen3TTSTokenizerSingleCodebookAdaLayerNormZeroFinal(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)
        self.post_init()

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
        batch_size = quantized_code.shape[0]
        maximum_duration = quantized_code.shape[1] * self.repeats
        if maximum_duration > self.config.max_position_embeddings:
            raise ValueError(
                f"Requested mel length ({maximum_duration}) exceeds `dit_config.max_position_embeddings` "
                f"({self.config.max_position_embeddings}). Provide shorter `quantized_code`."
            )

        initial_state = torch.randn(
            [batch_size, maximum_duration, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
            device=quantized_code.device,
        )
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

        # Euler integration, as in the original 25 Hz tokenizer.
        values = initial_state
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
        # The block-causal attention mask of the DiT is boolean, which eager attention does not accept.
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
        code (`torch.LongTensor` of shape `(batch_size, codes_length)`):
            Discrete speech codes.
        conditioning (`torch.FloatTensor` of shape `(batch_size, enc_emb_dim)`):
            Speaker embedding conditioning the diffusion transformer.
        reference_mel (`torch.FloatTensor` of shape `(batch_size, num_frames, mel_dim)`):
            Reference mel spectrogram conditioning the diffusion transformer.
        num_steps (`int`, *optional*, defaults to 10):
            Number of Euler steps of the diffusion sampler.
        guidance_scale (`float`, *optional*, defaults to 0.5):
            Classifier-free guidance scale. `0` disables guidance.
        sway_coefficient (`float`, *optional*, defaults to -1.0):
            Sway-sampling coefficient that skews the diffusion time steps. `None` keeps them uniform.
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
    Qwen3-TTS single-codebook tokenizer: a windowed Whisper-style encoder, a single-codebook vector quantizer, and a
    DiT plus BigVGAN decoder. `encode` maps log-mel features to one code per 640 input samples, `decode` maps codes
    back to a 24 kHz waveform.
    """
)
class Qwen3TTSTokenizerSingleCodebookModel(
    Qwen3TTSTokenizerSingleCodebookPreTrainedModel, PreTrainedAudioTokenizerBase
):
    def __init__(self, config: Qwen3TTSTokenizerSingleCodebookConfig):
        super().__init__(config)
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.encode_downsample_rate = config.encode_downsample_rate
        self.decode_upsample_rate = config.decode_upsample_rate

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
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`):
            Whisper-style log-mel features computed by [`Qwen3TTSTokenizerSingleCodebookFeatureExtractor`].
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of valid log-mel frames, `1` for a real frame and `0` for padding.
        """
        encoder_outputs = self.encoder(input_features, attention_mask=input_features_mask)
        encoder_mask = None
        if input_features_mask is not None:
            output_lengths = self.encoder._get_feat_extract_output_lengths(input_features_mask.long().sum(-1))
            seq_len = encoder_outputs.last_hidden_state.shape[1]
            encoder_mask = torch.arange(seq_len, device=input_features.device)[None, :] < output_lengths[:, None]
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
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length)`):
            Discrete speech codes. Pad shorter items with `-1`; the decoded waveform is zero past their duration.
        xvectors (`torch.FloatTensor` of shape `(batch_size, enc_emb_dim)`):
            Speaker embedding of the target voice. The original tokenizer computes it with an external CAM++ speaker
            model, so it is an input of this model rather than an output of the feature extractor.
        ref_mels (`torch.FloatTensor` of shape `(batch_size, num_frames, mel_dim)`):
            Reference mel spectrogram of the target voice, computed by [`Qwen3TTSTokenizerSingleCodebookFeatureExtractor`].
        num_steps (`int`, *optional*, defaults to 10):
            Number of Euler steps of the diffusion sampler.
        guidance_scale (`float`, *optional*, defaults to 0.5):
            Classifier-free guidance scale. `0` disables guidance.
        sway_coefficient (`float`, *optional*, defaults to -1.0):
            Sway-sampling coefficient that skews the diffusion time steps. `None` keeps them uniform.
        """
        audio_lengths = (audio_codes > -1).sum(1) * self.decode_upsample_rate
        audio_codes = torch.clamp(audio_codes, min=0)
        audio_values = self.decoder(
            code=audio_codes,
            conditioning=xvectors,
            reference_mel=ref_mels,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        audio_values_mask = (
            torch.arange(audio_values.shape[-1], device=audio_values.device)[None, :] < audio_lengths[:, None]
        )
        audio_values = audio_values * audio_values_mask
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
