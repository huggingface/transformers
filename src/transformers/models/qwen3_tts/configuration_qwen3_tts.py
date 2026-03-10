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
"""Qwen3-TTS configuration classes."""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..mimi.configuration_mimi import MimiConfig


logger = logging.get_logger(__name__)


class Qwen3TTSSpeakerEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSSpeakerEncoder`].
    It is used to instantiate a Qwen3-TTS speaker encoder model according to the specified arguments,
    defining the model architecture. The architecture is based on the ECAPA-TDNN model.

    Args:
        mel_dim (`int`, *optional*, defaults to 128):
            The dimension of the input mel-spectrogram.
        enc_dim (`int`, *optional*, defaults to 1024):
            The dimension of the final speaker embedding.
        enc_channels (`list[int]`, *optional*, defaults to `[512, 512, 512, 512, 1536]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder, corresponding to `enc_channels`.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder, corresponding to `enc_channels`.
        enc_attention_channels (`int`, *optional*, defaults to 128):
            The number of attention channels in the `AttentiveStatisticsPooling` layer.
        enc_res2net_scale (`int`, *optional*, defaults to 8):
            The scale of the `Res2NetBlock` in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 128):
            The number of channels in the squeeze part of the `SqueezeExcitationBlock`.
        sample_rate (`int`, *optional*, defaults to 24000):
            The sample rate of the audio.
    """

    model_type = "qwen3_tts_speaker_encoder"

    def __init__(
        self,
        mel_dim: int | None = 128,
        enc_dim: int | None = 1024,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int | None = 128,
        enc_res2net_scale: int | None = 8,
        enc_se_channels: int | None = 128,
        sample_rate: int | None = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.mel_dim = mel_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels if enc_channels is not None else [512, 512, 512, 512, 1536]
        self.enc_kernel_sizes = enc_kernel_sizes if enc_kernel_sizes is not None else [5, 3, 3, 3, 1]
        self.enc_dilations = enc_dilations if enc_dilations is not None else [1, 2, 3, 4, 1]
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.sample_rate = sample_rate


class Qwen3TTSTalkerCodePredictorConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerCodePredictorModel`].
    It is used to instantiate a Qwen3-TTS code predictor model according to the specified arguments,
    defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the Qwen3-TTS code predictor model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            The number of key_value heads for Grouped Query Attention (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embedding weights with output projection weights.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention.
        layer_types (`list[str]`, *optional*):
            List of attention layer types for each hidden layer. Defaults to alternating between `"full_attention"`
            and `"sliding_attention"` based on `max_window_layers`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups (codebooks).
        pad_token_id (`int`, *optional*):
            Padding token ID.
    """

    model_type = "qwen3_tts_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 2048,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 3072,
        num_hidden_layers: int | None = 5,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        num_code_groups: int | None = 32,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_type": "default", "rope_theta": 500000.0}
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.num_code_groups = num_code_groups


class Qwen3TTSTalkerConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerModel`].
    It is used to instantiate a Qwen3-TTS talker model according to the specified arguments,
    defining the model architecture.

    Args:
        code_predictor_config (`Qwen3TTSTalkerCodePredictorConfig`, *optional*):
            Configuration for the code predictor sub-model.
        vocab_size (`int`, *optional*, defaults to 3072):
            Vocabulary size of the Qwen3-TTS talker model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            The number of key_value heads for Grouped Query Attention (GQA).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embedding weights with output projection weights.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups (codebooks).
        text_hidden_size (`int`, *optional*, defaults to 2048):
            The dimension of the text embedding in the talker.
        codec_eos_token_id (`int`, *optional*, defaults to 4198):
            The end-of-sequence token ID for codec tokens.
        codec_think_id (`int`, *optional*, defaults to 4202):
            Token ID used to signal thinking mode in codec generation.
        codec_nothink_id (`int`, *optional*, defaults to 4203):
            Token ID used to signal non-thinking mode in codec generation.
        codec_think_bos_id (`int`, *optional*, defaults to 4204):
            Beginning-of-sequence token ID for codec thinking mode.
        codec_think_eos_id (`int`, *optional*, defaults to 4205):
            End-of-sequence token ID for codec thinking mode.
        codec_pad_id (`int`, *optional*, defaults to 4196):
            The padding token ID for codec tokens.
        codec_bos_id (`int`, *optional*, defaults to 4197):
            The beginning-of-sequence token ID for codec tokens.
        spk_id (`int`, *optional*):
            Speaker ID for built-in voice presets.
        spk_is_dialect (`bool`, *optional*):
            Whether the speaker uses a dialect variant.
        codec_language_id (`int`, *optional*):
            Language ID for codec generation.
        text_vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the text tokenizer.
        pad_token_id (`int`, *optional*):
            Padding token ID.
    """

    model_type = "qwen3_tts_talker"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}

    def __init__(
        self,
        code_predictor_config: dict | None = None,
        vocab_size: int | None = 3072,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 20,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 2,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        attention_dropout: float | None = 0.0,
        num_code_groups: int | None = 32,
        text_hidden_size: int | None = 2048,
        codec_eos_token_id: int | None = 4198,
        codec_think_id: int | None = 4202,
        codec_nothink_id: int | None = 4203,
        codec_think_bos_id: int | None = 4204,
        codec_think_eos_id: int | None = 4205,
        codec_pad_id: int | None = 4196,
        codec_bos_id: int | None = 4197,
        spk_id: int | None = None,
        spk_is_dialect: bool | None = None,
        codec_language_id: int | None = None,
        text_vocab_size: int | None = 152064,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.text_vocab_size = text_vocab_size

        # Build sub-config BEFORE super().__init__() so that the _attn_implementation
        # setter (triggered by super()) can propagate to it via sub_configs.
        if code_predictor_config is None:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(code_predictor_config, Qwen3TTSTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**code_predictor_config)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = (
            rope_parameters if rope_parameters is not None else {"rope_type": "default", "rope_theta": 500000.0}
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_think_id = codec_think_id
        self.codec_language_id = codec_language_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.spk_id = spk_id
        self.spk_is_dialect = spk_is_dialect


@auto_docstring(checkpoint="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
class Qwen3TTSConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSForConditionalGeneration`].
    It is used to instantiate a Qwen3-TTS model according to the specified arguments, defining the model architecture.

    Args:
        talker_config (`Qwen3TTSTalkerConfig`, *optional*):
            Configuration for the talker sub-model (text-to-acoustic backbone).
        speaker_encoder_config (`Qwen3TTSSpeakerEncoderConfig`, *optional*):
            Configuration for the speaker encoder sub-model (extracts speaker embeddings).
        tokenizer_type (`str`, *optional*):
            Type of audio tokenizer to use (e.g., "12hz", "25hz").
        tts_model_size (`str`, *optional*):
            Size of the TTS model.
        tts_model_type (`str`, *optional*):
            Type of TTS model.
        im_start_token_id (`int`, *optional*, defaults to 151644):
            The beginning-of-image token ID (used as special marker in input).
        im_end_token_id (`int`, *optional*, defaults to 151645):
            The end-of-image token ID (used as special marker in input).
        tts_pad_token_id (`int`, *optional*, defaults to 151671):
            The padding token ID for TTS generation.
        tts_bos_token_id (`int`, *optional*, defaults to 151672):
            The beginning-of-sequence token ID for TTS generation.
        tts_eos_token_id (`int`, *optional*, defaults to 151673):
            The end-of-sequence token ID for TTS generation.
    """

    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
        "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig,
    }

    def __init__(
        self,
        talker_config: dict | None = None,
        speaker_encoder_config: dict | None = None,
        tokenizer_type: str | None = None,
        tts_model_size: str | None = None,
        tts_model_type: str | None = None,
        im_start_token_id: int | None = 151644,
        im_end_token_id: int | None = 151645,
        tts_pad_token_id: int | None = 151671,
        tts_bos_token_id: int | None = 151672,
        tts_eos_token_id: int | None = 151673,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")
        if speaker_encoder_config is None:
            speaker_encoder_config = {}
            logger.info("speaker_encoder_config is None. Initializing speaker encoder with default values")

        self.talker_config = Qwen3TTSTalkerConfig(**talker_config)
        self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**speaker_encoder_config)

        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type

        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id


class Qwen3TTSTokenizerV2Code2WavConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V2 tokenizer decoder (Code2Wav).

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for GQA.
        head_dim (`int`, *optional*):
            Attention head dimension. Defaults to `hidden_size // num_attention_heads`.
        intermediate_size (`int`, *optional*, defaults to 2048):
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
        codebook_dim (`int`, *optional*, defaults to 256):
            Dimension of quantizer codebook vectors.
        num_quantizers (`int`, *optional*, defaults to 4):
            Number of residual vector quantizers.
        codebook_size (`int`, *optional*, defaults to 2048):
            Size of each codebook.
        latent_dim (`int`, *optional*, defaults to 512):
            Latent dimension used between pre-conv and transformer.
        decoder_dim (`int`, *optional*, defaults to 512):
            Initial dimension for the convolutional decoder stack.
        upsample_rates (`list[int]`, *optional*, defaults to `[5, 4, 2, 2]`):
            Upsampling rates for the BigVGAN-style decoder.
        upsampling_ratios (`list[int]`, *optional*, defaults to `[8, 6]`):
            Upsampling ratios for the pre-transformer upsample blocks.
    """

    model_type = "qwen3_tts_tokenizer_v2_code2wav"

    def __init__(
        self,
        hidden_size: int | None = 1024,
        num_hidden_layers: int | None = 4,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        intermediate_size: int | None = 2048,
        hidden_act: str | None = "silu",
        rms_norm_eps: float | None = 1e-5,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        sliding_window: int | None = 4096,
        max_position_embeddings: int | None = 32768,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        layer_scale_initial_scale: float | None = 0.01,
        codebook_dim: int | None = 256,
        num_quantizers: int | None = 4,
        codebook_size: int | None = 2048,
        latent_dim: int | None = 512,
        decoder_dim: int | None = 512,
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
        self.upsample_rates = list(upsample_rates) if upsample_rates is not None else [5, 4, 2, 2]
        self.upsampling_ratios = list(upsampling_ratios) if upsampling_ratios is not None else [8, 6]
        self.layer_types = ["sliding_attention" for _ in range(self.num_hidden_layers)]


class Qwen3TTSTokenizerV2Config(PreTrainedConfig):
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

    model_type = "qwen3_tts_tokenizer_v2"
    sub_configs = {
        "encoder_config": MimiConfig,
        "decoder_config": Qwen3TTSTokenizerV2Code2WavConfig,
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
            Qwen3TTSTokenizerV2Code2WavConfig(**decoder_config) if isinstance(decoder_config, dict) else decoder_config
        )

        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate


class Qwen3TTSDiTConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V1 DiT decoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the DiT model.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of transformer blocks.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        ff_mult (`int`, *optional*, defaults to 2):
            Feedforward layer multiplier.
        emb_dim (`int`, *optional*, defaults to 512):
            Codec embedding dimension.
        head_dim (`int`, *optional*, defaults to 64):
            Attention head dimension.
        repeats (`int`, *optional*, defaults to 2):
            Number of times codec embeddings are repeated.
        num_embeds (`int`, *optional*, defaults to 8193):
            Number of unique codec embeddings.
        mel_dim (`int`, *optional*, defaults to 80):
            Mel-spectrogram dimension.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout probability.
        block_size (`int`, *optional*, defaults to 24):
            Block size for block-diagonal attention mask.
        look_ahead_layers (`list[int]`, *optional*, defaults to `[10]`):
            Layer indices that use look-ahead attention.
        look_backward_layers (`list[int]`, *optional*, defaults to `[0, 20]`):
            Layer indices that use look-backward attention.
        enc_emb_dim (`int`, *optional*, defaults to 192):
            Speaker embedding dimension.
        enc_dim (`int`, *optional*, defaults to 128):
            Encoder output dimension.
        enc_channels (`list[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
            Encoder channel sizes.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            Encoder kernel sizes.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            Encoder dilations.
        enc_attention_channels (`int`, *optional*, defaults to 64):
            Encoder attention channels.
        enc_res2net_scale (`int`, *optional*, defaults to 2):
            Encoder Res2Net scale.
        enc_se_channels (`int`, *optional*, defaults to 64):
            Encoder SE channels.
        rope_parameters (`RopeParameters`, *optional*):
            RoPE configuration.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length.
    """

    model_type = "qwen3_tts_dit"

    def __init__(
        self,
        hidden_size: int | None = 1024,
        num_hidden_layers: int | None = 22,
        num_attention_heads: int | None = 16,
        ff_mult: int | None = 2,
        emb_dim: int | None = 512,
        head_dim: int | None = 64,
        repeats: int | None = 2,
        num_embeds: int | None = 8193,
        mel_dim: int | None = 80,
        dropout: float | None = 0.1,
        block_size: int | None = 24,
        look_ahead_layers: list[int] | None = None,
        look_backward_layers: list[int] | None = None,
        enc_emb_dim: int | None = 192,
        enc_dim: int | None = 128,
        enc_channels: list[int] | None = None,
        enc_kernel_sizes: list[int] | None = None,
        enc_dilations: list[int] | None = None,
        enc_attention_channels: int | None = 64,
        enc_res2net_scale: int | None = 2,
        enc_se_channels: int | None = 64,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        max_position_embeddings: int | None = 32768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ff_mult = ff_mult
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.repeats = repeats
        self.num_embeds = num_embeds
        self.mel_dim = mel_dim
        self.dropout = dropout
        self.block_size = block_size
        self.look_ahead_layers = look_ahead_layers if look_ahead_layers is not None else [10]
        self.look_backward_layers = look_backward_layers if look_backward_layers is not None else [0, 20]
        self.enc_emb_dim = enc_emb_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels if enc_channels is not None else [256, 256, 256, 256, 768]
        self.enc_kernel_sizes = enc_kernel_sizes if enc_kernel_sizes is not None else [5, 3, 3, 3, 1]
        self.enc_dilations = enc_dilations if enc_dilations is not None else [1, 2, 3, 4, 1]
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.rope_parameters = rope_parameters
        self.max_position_embeddings = max_position_embeddings


class Qwen3TTSTokenizerV1DecoderBigVGANConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V1 BigVGAN vocoder.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            Mel-spectrogram input dimension.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            Initial channel count for the upsampling stack.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            Kernel sizes for each residual block.
        resblock_dilation_sizes (`list[list[int]]`, *optional*):
            Dilation sizes for each residual block.
        upsample_rates (`list[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            Upsampling rates for each layer.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            Kernel sizes for each upsampling layer.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder_bigvgan"

    def __init__(
        self,
        mel_dim: int | None = 80,
        upsample_initial_channel: int | None = 1536,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mel_dim = mel_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes if resblock_kernel_sizes is not None else [3, 7, 11]
        self.resblock_dilation_sizes = (
            resblock_dilation_sizes if resblock_dilation_sizes is not None else [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        )
        self.upsample_rates = upsample_rates if upsample_rates is not None else [5, 3, 2, 2, 2, 2]
        self.upsample_kernel_sizes = (
            upsample_kernel_sizes if upsample_kernel_sizes is not None else [11, 7, 4, 4, 4, 4]
        )


class Qwen3TTSTokenizerV1DecoderConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V1 decoder (DiT + BigVGAN).

    Args:
        dit_config (`dict`, *optional*):
            Configuration for the DiT sub-model.
        bigvgan_config (`dict`, *optional*):
            Configuration for the BigVGAN sub-model.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder"
    sub_configs = {
        "dit_config": Qwen3TTSDiTConfig,
        "bigvgan_config": Qwen3TTSTokenizerV1DecoderBigVGANConfig,
    }

    def __init__(
        self,
        dit_config: dict | None = None,
        bigvgan_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if dit_config is None:
            dit_config = {}
            logger.info("dit_config is None. Initializing DiT with default values.")
        if bigvgan_config is None:
            bigvgan_config = {}
            logger.info("bigvgan_config is None. Initializing BigVGAN with default values.")

        self.dit_config = Qwen3TTSDiTConfig(**dit_config) if isinstance(dit_config, dict) else dit_config
        self.bigvgan_config = (
            Qwen3TTSTokenizerV1DecoderBigVGANConfig(**bigvgan_config)
            if isinstance(bigvgan_config, dict)
            else bigvgan_config
        )


class Qwen3TTSTokenizerV1EncoderConfig(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V1 Whisper-based VQ encoder.

    Args:
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel filterbanks.
        n_ctx (`int`, *optional*, defaults to 1500):
            Maximum context length.
        n_state (`int`, *optional*, defaults to 1024):
            Hidden state dimension.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads.
        n_layer (`int`, *optional*, defaults to 24):
            Number of transformer layers.
        n_window (`int`, *optional*, defaults to 128):
            Window size for windowed attention.
        output_dim (`int`, *optional*, defaults to 512):
            VQ output dimension.
        grad_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
        enable_mp (`bool`, *optional*, defaults to `False`):
            Whether to enable mixed precision.
        audio_sequence_parallel (`bool`, *optional*, defaults to `False`):
            Whether to use sequence parallelism for audio.
        audio_vq_type (`str`, *optional*, defaults to `"residual"`):
            Type of vector quantization.
        audio_vq_layers (`int`, *optional*, defaults to 1):
            Number of VQ layers.
        audio_vq_codebook_size (`int`, *optional*, defaults to 512):
            Codebook size.
        audio_vq_codebook_dim (`int`, *optional*, defaults to 512):
            Codebook vector dimension.
        audio_vq_pe (`str`, *optional*, defaults to `"rope"`):
            Position encoding type for VQ.
        audio_vq_ds_rate (`int`, *optional*, defaults to 2):
            Downsampling rate inside the VQ module.
    """

    model_type = "qwen3_tts_tokenizer_v1_encoder"

    def __init__(
        self,
        n_mels: int | None = 128,
        n_ctx: int | None = 1500,
        n_state: int | None = 1024,
        n_head: int | None = 16,
        n_layer: int | None = 24,
        n_window: int | None = 128,
        output_dim: int | None = 512,
        grad_checkpointing: bool | None = False,
        enable_mp: bool | None = False,
        audio_sequence_parallel: bool | None = False,
        audio_vq_type: str | None = "residual",
        audio_vq_layers: int | None = 1,
        audio_vq_codebook_size: int | None = 512,
        audio_vq_codebook_dim: int | None = 512,
        audio_vq_pe: str | None = "rope",
        audio_vq_ds_rate: int | None = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.n_state = n_state
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_window = n_window
        self.output_dim = output_dim
        self.grad_checkpointing = grad_checkpointing
        self.enable_mp = enable_mp
        self.audio_sequence_parallel = audio_sequence_parallel
        self.audio_vq_type = audio_vq_type
        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_codebook_dim = audio_vq_codebook_dim
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_ds_rate = audio_vq_ds_rate


class Qwen3TTSTokenizerV1Config(PreTrainedConfig):
    r"""
    Configuration class for the Qwen3-TTS V1 tokenizer (Whisper VQ encoder + DiT/BigVGAN decoder).

    Args:
        encoder_config (`dict`, *optional*):
            Configuration for the Whisper-based VQ encoder.
        decoder_config (`dict`, *optional*):
            Configuration for the DiT+BigVGAN decoder.
        input_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the input audio.
        output_sample_rate (`int`, *optional*, defaults to 24000):
            Sample rate of the decoded output audio.
        decode_upsample_rate (`int`, *optional*, defaults to 200):
            Upsampling rate applied during decoding.
        encode_downsample_rate (`int`, *optional*, defaults to 200):
            Downsampling rate applied during encoding.
    """

    model_type = "qwen3_tts_tokenizer_v1"
    sub_configs = {
        "encoder_config": Qwen3TTSTokenizerV1EncoderConfig,
        "decoder_config": Qwen3TTSTokenizerV1DecoderConfig,
    }

    def __init__(
        self,
        encoder_config: dict | None = None,
        decoder_config: dict | None = None,
        input_sample_rate: int | None = 24000,
        output_sample_rate: int | None = 24000,
        decode_upsample_rate: int | None = 200,
        encode_downsample_rate: int | None = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing V1 encoder with default values.")
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing V1 decoder with default values.")

        self.encoder_config = (
            Qwen3TTSTokenizerV1EncoderConfig(**encoder_config) if isinstance(encoder_config, dict) else encoder_config
        )
        self.decoder_config = (
            Qwen3TTSTokenizerV1DecoderConfig(**decoder_config) if isinstance(decoder_config, dict) else decoder_config
        )

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate


__all__ = [
    "MimiConfig",
    "Qwen3TTSConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
    "Qwen3TTSTokenizerV2Code2WavConfig",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSDiTConfig",
    "Qwen3TTSTokenizerV1DecoderBigVGANConfig",
    "Qwen3TTSTokenizerV1DecoderConfig",
    "Qwen3TTSTokenizerV1EncoderConfig",
    "Qwen3TTSTokenizerV1Config",
]
