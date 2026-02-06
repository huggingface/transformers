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
from ...utils import logging

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
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups (codebooks).
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
        self.rope_parameters = rope_parameters
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
        code_predictor_config (`Qwen3TTSTalkerCodePredictorConfig`, *optional*):
            Configuration for the code predictor sub-model.
        codec_eos_token_id (`int`, *optional*, defaults to 4198):
            The end-of-sequence token ID for codec tokens.
        codec_pad_id (`int`, *optional*, defaults to 4196):
            The padding token ID for codec tokens.
        codec_bos_id (`int`, *optional*, defaults to 4197):
            The beginning-of-sequence token ID for codec tokens.
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
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if code_predictor_config is None:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(code_predictor_config, Qwen3TTSTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**code_predictor_config)

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


__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSSpeakerEncoderConfig",
    "Qwen3TTSTalkerCodePredictorConfig",
]
