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
"""PyTorch Qwen3TTS Multi-Codebook Tokenizer model."""

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedAudioTokenizerBase
from ...utils import auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..dac.modeling_dac import DacDecoderOutput
from ..encodec.modeling_encodec import EncodecOutput
from ..mimi.modeling_mimi import (
    MimiEncoderOutput,
    MimiEuclideanCodebook,
    MimiModel,
    MimiPreTrainedModel,
    MimiResidualVectorQuantizer,
    MimiSplitResidualVectorQuantizer,
    MimiVectorQuantization,
)
from ..qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniSnakeBeta
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
)


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class Qwen3TTSTokenizerMultiCodebookQuantizerConfig(PreTrainedConfig):
    r"""
    codebook_dim (`int`, *optional*, defaults to 256):
        Dimension of each codebook vector.
    frame_rate (`int`, *optional*, defaults to 0):
        Frame rate used by the vector quantizers.
    num_quantizers (`int`, *optional*, defaults to 16):
        Total number of residual vector quantizers.
    num_semantic_quantizers (`int`, *optional*, defaults to 1):
        Number of quantizers assigned to semantic codes.
    vector_quantization_hidden_dimension (`int`, *optional*, defaults to 256):
        Dimension used within the vector quantizers.
    """

    model_type = "qwen3_tts_tokenizer_multi_codebook_quantizer"
    base_config_key = "quantizer_config"

    codebook_size: int = 2048
    codebook_dim: int = 256
    frame_rate: int = 0
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    vector_quantization_hidden_dimension: int = 256
    hidden_size: int = 512


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerMultiCodebookCode2WavConfig(PreTrainedConfig):
    r"""
    num_quantizers (`int`, *optional*, defaults to 16):
        Number of residual vector quantizers used in the vocoder for fine-grained audio reconstruction.
    upsample_rates (`Tuple[int]`, *optional*, defaults to `(8, 5, 4, 3)`):
        Rate at which features are upsampled in the final waveform synthesis stage.
    upsampling_ratios (`Tuple[int]`, *optional*, defaults to `(2, 2)`):
        Ratios used in transposed convolutional layers to progressively upsample feature maps to waveform.
    decoder_dim (`int`, *optional*, defaults to 1536):
        Final dimensionality of the decoder's output before waveform generation.
    num_semantic_quantizers (`int`, *optional*, defaults to 1):
        Number of semantic quantizer layers.
    semantic_codebook_size (`int`, *optional*, defaults to 4096):
        Size of the semantic codebook.
    latent_dim (`int`, *optional*, defaults to 1024):
        Latent dimension used between pre-conv and transformer.
    vector_quantization_hidden_dimension (`int`, *optional*, defaults to 512):
        Hidden dimension for the vector quantization projection.
    use_causal_conv (`bool`, *optional*, defaults to `True`):
        Whether to use causal convolutions in the decoder.
    trim_right_ratio (`float`, *optional*, defaults to 1.0):
        Ratio for trimming the right side of transposed convolution output.
    """

    model_type = "qwen3_tts_tokenizer_multi_codebook_code2wav"
    codebook_size: int = 2048
    hidden_size: int = 512
    max_position_embeddings: int = 8000
    rope_parameters: dict | None = None
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 1024
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    num_quantizers: int = 16
    upsample_rates: list[int] | tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: list[int] | tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    head_dim: int = 64
    codebook_dim: int = 512
    num_semantic_quantizers: int = 1
    semantic_codebook_size: int = 4096
    latent_dim: int = 1024
    vector_quantization_hidden_dimension: int = 512
    use_causal_conv: bool = True
    trim_right_ratio: float = 1.0
    @property
    def layer_types(self):
        return ["sliding_attention"] * self.num_hidden_layers


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-Tokenizer-12Hz")
@strict
class Qwen3TTSTokenizerMultiCodebookConfig(PreTrainedConfig):
    r"""
    encoder_config (`dict`, *optional*):
        Configuration for the Mimi-based encoder sub-model.
    quantizer_config (`dict`, *optional*):
        Configuration for the split residual vector quantizer.
    decoder_config (`dict`, *optional*):
        Configuration for the Code2Wav decoder sub-model.
    input_sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate, in hertz (Hz), of the encoder's input audio waveform.
    output_sampling_rate (`int`, *optional*, defaults to 24000):
        Sampling rate, in hertz (Hz), of the decoder's output audio waveform.
    """

    model_type = "qwen3_tts_tokenizer_multi_codebook"
    sub_configs = {
        "encoder_config": AutoConfig,
        "quantizer_config": Qwen3TTSTokenizerMultiCodebookQuantizerConfig,
        "decoder_config": AutoConfig,
    }

    encoder_config: dict | PreTrainedConfig | None = None
    quantizer_config: dict | PreTrainedConfig | None = None
    decoder_config: dict | PreTrainedConfig | None = None
    input_sampling_rate: int | None = 24000
    output_sampling_rate: int | None = 24000

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config["model_type"] = self.encoder_config.get("model_type", "mimi")
            self.encoder_config["num_quantizers"] = self.encoder_config.get("num_quantizers", 16)
            self.encoder_config = CONFIG_MAPPING[self.encoder_config["model_type"]](**self.encoder_config)
        elif self.encoder_config is None:
            logger.info("encoder_config is None. Initializing V2 encoder with default values.")
            self.encoder_config = CONFIG_MAPPING["mimi"](num_quantizers=16)

        if isinstance(self.decoder_config, dict):
            self.decoder_config["model_type"] = self.decoder_config.get(
                "model_type", "qwen3_tts_tokenizer_multi_codebook_code2wav"
            )
            self.decoder_config = CONFIG_MAPPING[self.decoder_config["model_type"]](**self.decoder_config)
        elif self.decoder_config is None:
            logger.info("decoder_config is None. Initializing V2 decoder with default values.")
            self.decoder_config = CONFIG_MAPPING["qwen3_tts_tokenizer_multi_codebook_code2wav"]()

        if self.quantizer_config is None:
            self.quantizer_config = Qwen3TTSTokenizerMultiCodebookQuantizerConfig(
                codebook_size=self.decoder_config.codebook_size,
                codebook_dim=self.decoder_config.codebook_dim // 2,
                num_quantizers=self.decoder_config.num_quantizers,
                num_semantic_quantizers=self.decoder_config.num_semantic_quantizers,
                vector_quantization_hidden_dimension=self.decoder_config.codebook_dim // 2,
                hidden_size=self.decoder_config.codebook_dim,
            )
        elif isinstance(self.quantizer_config, dict):
            self.quantizer_config = Qwen3TTSTokenizerMultiCodebookQuantizerConfig(**self.quantizer_config)

        super().__post_init__(**kwargs)

class Qwen3TTSTokenizerMultiCodebookCausalConvNet(Qwen3OmniMoeCausalConvNet):
    pass


class Qwen3TTSTokenizerMultiCodebookCausalTransConvNet(Qwen3OmniMoeCausalTransConvNet):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(in_channels, out_channels, kernel_size, stride)
        pad = kernel_size - stride
        self.left_pad = 0
        self.right_pad = int(pad)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerMultiCodebookConvNeXtBlock(Qwen3OmniMoeConvNeXtBlock):
    pass


class Qwen3TTSTokenizerMultiCodebookEncoderOutput(MimiEncoderOutput):
    pass


class Qwen3TTSTokenizerMultiCodebookDecoderOutput(DacDecoderOutput):
    pass


class Qwen3TTSTokenizerMultiCodebookOutput(EncodecOutput):
    pass



class Qwen3TTSTokenizerMultiCodebookPreTrainedModel(MimiPreTrainedModel):
    base_model_prefix = "model"
    _can_compile_fullgraph = False


@auto_docstring
class Qwen3TTSTokenizerMultiCodebookCode2WavPreTrainedModel(Qwen3TTSTokenizerMultiCodebookPreTrainedModel):
    config_class = Qwen3TTSTokenizerMultiCodebookCode2WavConfig
    _no_split_modules = ["Qwen3OmniMoeCode2WavTransformerLayer", "Qwen3TTSTokenizerMultiCodebookDecoderBlock"]


#  Decoder block


class Qwen3TTSTokenizerMultiCodebookDecoderBlock(Qwen3OmniMoeCode2WavDecoderBlock):
    pass


class Qwen3TTSTokenizerMultiCodebookSnakeBeta(Qwen2_5OmniSnakeBeta):
    pass


#  VQ / RVQ classes


class Qwen3TTSTokenizerMultiCodebookEuclideanCodebook(MimiEuclideanCodebook):
    pass


class Qwen3TTSTokenizerMultiCodebookVectorQuantization(MimiVectorQuantization):
    pass


class Qwen3TTSTokenizerMultiCodebookResidualVectorQuantizer(MimiResidualVectorQuantizer):
    pass


class Qwen3TTSTokenizerMultiCodebookSplitResidualVectorQuantizer(MimiSplitResidualVectorQuantizer):
    pass


class Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel(Qwen3OmniMoeCode2WavTransformerModel):
    pass


#  Decoder


class Qwen3TTSTokenizerMultiCodebookDecoder(Qwen3TTSTokenizerMultiCodebookCode2WavPreTrainedModel):
    config_class = Qwen3TTSTokenizerMultiCodebookCode2WavConfig

    def __init__(self, config: config_class):
        super().__init__(config)
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        self.pre_transformer = Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel(config)
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

        self.pre_conv = Qwen3TTSTokenizerMultiCodebookCausalConvNet(
            config.codebook_dim, config.latent_dim, kernel_size=3
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerMultiCodebookCausalTransConvNet(
                            config.latent_dim, config.latent_dim, factor, factor
                        ),
                        Qwen3TTSTokenizerMultiCodebookConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerMultiCodebookCausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerMultiCodebookDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            Qwen3TTSTokenizerMultiCodebookSnakeBeta(output_dim),
            Qwen3TTSTokenizerMultiCodebookCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
        self.post_init()

    @auto_docstring
    def forward(self, quantized_representation, **kwargs):
        r"""
        quantized_representation (`torch.FloatTensor` of shape `(batch_size, codebook_dim, sequence_length)`):
            Quantized continuous representation to decode into waveform values.
        """
        hidden = self.pre_conv(quantized_representation).transpose(1, 2)
        hidden = self.input_proj(hidden)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = self.output_proj(hidden)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, quantized_representation, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < quantized_representation.shape[-1]:
            end_index = min(start_index + chunk_size, quantized_representation.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            hidden_states_chunk = quantized_representation[..., start_index - context_size : end_index]
            wav_chunk = self(hidden_states_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


#  Encoder (Mimi-based, encoder-only)


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerMultiCodebook encoder model, based on MimiModel but only using the encoder path.
    """
)
class Qwen3TTSTokenizerMultiCodebookEncoderModel(MimiModel):
    def __init__(self, config):
        super().__init__(config)
        # Encoder-only model: waveform reconstruction is handled by the separate multi-codebook
        # decoder, so Mimi's decode stack is dropped. `upsample` is cleared through `setattr`
        # because Mimi assigns it twice — once as `None`, then conditionally as a module — and a
        # plain `self.upsample = None` here is folded into the first of those by the modular
        # converter, leaving the module to be rebuilt afterwards. It has no weights in the
        # checkpoint, so keeping it would ship a randomly initialized parameter.
        self.decoder = None
        self.decoder_transformer = None
        setattr(self, "upsample", None)


#  Top-level Model


@auto_docstring
class Qwen3TTSTokenizerMultiCodebookModel(Qwen3TTSTokenizerMultiCodebookPreTrainedModel, PreTrainedAudioTokenizerBase):
    config_class = Qwen3TTSTokenizerMultiCodebookConfig
    main_input_name = "input_values"

    def __init__(self, config: Qwen3TTSTokenizerMultiCodebookConfig):
        super().__init__(config)
        self.config = config

        self.input_sampling_rate = config.input_sampling_rate
        self.output_sampling_rate = config.output_sampling_rate

        self.encoder = Qwen3TTSTokenizerMultiCodebookEncoderModel(self.config.encoder_config)
        self.quantizer = Qwen3TTSTokenizerMultiCodebookSplitResidualVectorQuantizer(self.config.quantizer_config)
        self.decoder = Qwen3TTSTokenizerMultiCodebookDecoder(self.config.decoder_config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Padding mask used to pad `input_values`.
        """

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = self.encoder.encode(
            input_values=input_values.unsqueeze(1),
            num_quantizers=self.config.encoder_config.num_quantizers,
            return_dict=True,
        )
        audio_codes = encoded_frames.audio_codes
        audio_codes = [
            code[..., : -(-mask.sum() // self.encoder.config.frame_size)].transpose(0, 1)
            for code, mask in zip(audio_codes, padding_mask)
        ]

        return Qwen3TTSTokenizerMultiCodebookEncoderOutput(audio_codes=audio_codes)

    @can_return_tuple
    @auto_docstring
    def decode(
        self,
        audio_codes: torch.Tensor,
    ):
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, codes_length, num_quantizers)`):
            Discrete code indices computed using `model.encode`.
        """
        quantized_representation = self.quantizer.decode(audio_codes.transpose(1, 2))
        audio_values = self.decoder.chunked_decode(quantized_representation).squeeze(1)

        return Qwen3TTSTokenizerMultiCodebookDecoderOutput(audio_values=audio_values)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Input audio waveform.
        padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Padding mask used to pad `input_values`.
        """
        length = input_values.shape[-1]
        encoder_outputs = self.encode(input_values, padding_mask=padding_mask, return_dict=True)
        audio_codes = pad_sequence(encoder_outputs.audio_codes, batch_first=True, padding_value=-1)

        decoder_outputs = self.decode(audio_codes.clamp(min=0), return_dict=True)
        audio_values = decoder_outputs.audio_values[..., :length]

        return Qwen3TTSTokenizerMultiCodebookOutput(
            audio_values=audio_values,
            audio_codes=audio_codes,
        )


__all__ = [
    "Qwen3TTSTokenizerMultiCodebookConfig",
    "Qwen3TTSTokenizerMultiCodebookCode2WavConfig",
    "Qwen3TTSTokenizerMultiCodebookModel",
    "Qwen3TTSTokenizerMultiCodebookPreTrainedModel",
]
