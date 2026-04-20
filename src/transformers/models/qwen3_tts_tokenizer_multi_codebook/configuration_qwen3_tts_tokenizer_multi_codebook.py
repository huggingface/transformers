# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging
from ..mimi.configuration_mimi import MimiConfig
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig


logger = logging.get_logger(__name__)


@strict
class Qwen3TTSTokenizerMultiCodebookDecoderConfig(Qwen3OmniMoeCode2WavConfig):
    r"""
    Configuration class for the Qwen3-TTS V2 tokenizer decoder (Code2Wav).

    Args:
        hidden_size (`int`, *optional*, defaults to 512):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for GQA.
        head_dim (`int`, *optional*):
            Attention head dimension. Defaults to `hidden_size // num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 1024):
            MLP intermediate dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon for RMS norm.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size for attention.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE configuration.
        layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
            Initial scale for layer scale residual.
        codebook_dim (`int`, *optional*, defaults to 512):
            Dimension of quantizer codebook vectors.
        num_quantizers (`int`, *optional*, defaults to 16):
            Number of residual vector quantizers.
        codebook_size (`int`, *optional*, defaults to 2048):
            Size of each codebook.
        latent_dim (`int`, *optional*, defaults to 1024):
            Latent dimension used between pre-conv and transformer.
        decoder_dim (`int`, *optional*, defaults to 1536):
            Initial dimension for the convolutional decoder stack.
        upsample_rates (`list[int]`, *optional*, defaults to `[8, 5, 4, 3]`):
            Upsampling rates for the BigVGAN-style decoder.
        upsampling_ratios (`list[int]`, *optional*, defaults to `[2, 2]`):
            Upsampling ratios for the pre-transformer upsample blocks.
    """

    model_type = "qwen3_tts_tokenizer_multi_codebook_code2wav"

    def __init__(
        self,
        hidden_size: int | None = 512,
        num_hidden_layers: int | None = 8,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        intermediate_size: int | None = 1024,
        hidden_act: str | None = "silu",
        rms_norm_eps: float | None = 1e-5,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        sliding_window: int | None = 4096,
        max_position_embeddings: int | None = 32768,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        layer_scale_initial_scale: float | None = 0.01,
        codebook_dim: int | None = 512,
        num_quantizers: int | None = 16,
        codebook_size: int | None = 2048,
        latent_dim: int | None = 1024,
        decoder_dim: int | None = 1536,
        upsample_rates: list[int] | None = None,
        upsampling_ratios: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_type": "default", "rope_theta": 10000.0}
        )
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.upsample_rates = list(upsample_rates) if upsample_rates is not None else [8, 5, 4, 3]
        self.upsampling_ratios = list(upsampling_ratios) if upsampling_ratios is not None else [2, 2]


@strict
class Qwen3TTSTokenizerMultiCodebookConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V2 tokenizer (encoder + decoder).

    Args:
        encoder_config (`dict`, *optional*):
            Configuration for the Mimi-based encoder sub-model.
        decoder_config (`dict`, *optional*):
            Configuration for the Code2Wav decoder sub-model.
        encoder_valid_num_quantizers (`int`, *optional*, defaults to 4):
            Number of quantizer layers the encoder actually uses.
        input_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the input audio.
        output_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the decoded output audio.
        decode_upsample_rate (`int`, *optional*, defaults to 16):
            Upsampling rate applied during decoding.
        encode_downsample_rate (`int`, *optional*, defaults to 16):
            Downsampling rate applied during encoding.
    """

    model_type = "qwen3_tts_tokenizer_multi_codebook"
    sub_configs = {
        "encoder_config": MimiConfig,
        "decoder_config": Qwen3TTSTokenizerMultiCodebookDecoderConfig,
    }

    def __init__(
        self,
        encoder_config: dict | None = None,
        decoder_config: dict | None = None,
        encoder_valid_num_quantizers: int | None = 4,
        input_sample_rate: int | None = 24000,
        output_sample_rate: int | None = 24000,
        decode_upsample_rate: int | None = 16,
        encode_downsample_rate: int | None = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing V2 encoder with default values.")
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing V2 decoder with default values.")

        self.encoder_config = MimiConfig(**encoder_config) if isinstance(encoder_config, dict) else encoder_config
        self.decoder_config = (
            Qwen3TTSTokenizerMultiCodebookDecoderConfig(**decoder_config)
            if isinstance(decoder_config, dict)
            else decoder_config
        )

        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate


__all__ = ["Qwen3TTSTokenizerMultiCodebookConfig", "Qwen3TTSTokenizerMultiCodebookCode2WavConfig"]
