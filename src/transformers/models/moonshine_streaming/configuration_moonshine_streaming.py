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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ..auto import CONFIG_MAPPING


class MoonshineStreamingEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineStreamingEncoder`]. It is used to
    instantiate a Moonshine Streaming encoder according to the specified arguments, defining the encoder architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Moonshine Streaming tiny model.
    e.g. [UsefulSensors/moonshine-streaming-tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 320):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1280):
            Dimension of the MLP representations.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        sample_rate (`int`, *optional*, defaults to 16000):
            The sample rate of the audio input in Hz.
        frame_ms (`float`, *optional*, defaults to 5.0):
            The frame duration in milliseconds for audio processing.
        sliding_windows (`list[tuple[int, int]]`, *optional*, defaults to `[(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)]`):
            List of sliding window configurations for each encoder layer. Each tuple contains (window_size, shift).
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads.

    ```python
    >>> from transformers import MoonshineStreamingEncoder, MoonshineStreamingEncoderConfig

    >>> # Initializing a Moonshine Streaming encoder configuration
    >>> configuration = MoonshineStreamingEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = MoonshineStreamingEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine_streaming_encoder"

    def __init__(
        self,
        hidden_size: int | None = 320,
        intermediate_size: int | None = 1280,
        hidden_act: str | None = "gelu",
        num_hidden_layers: int | None = 6,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 8,
        max_position_embeddings: int | None = 4096,
        attention_dropout: float | None = 0.0,
        attention_bias: bool | None = False,
        sample_rate: int = 16000,
        frame_ms: float = 5.0,
        sliding_windows: list[tuple[int, int]] = [(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)],
        head_dim: int | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.sliding_windows = [list(window) for window in sliding_windows]

        super().__init__(**kwargs)


class MoonshineStreamingConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineStreamingModel`]. It is used to
    instantiate a Moonshine Streaming model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Moonshine
    Streaming tiny model.
    e.g. [UsefulSensors/moonshine-streaming-tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
            encoder_config (`MoonshineStreamingEncoderConfig`, *optional*):
                Configuration of the encoder. If not provided, a default `MoonshineStreamingEncoderConfig` will be
                instantiated.
            vocab_size (`int`, *optional*, defaults to 32768):
                Vocabulary size of the Moonshine Streaming decoder model. Defines the number of different tokens that can
                be represented by the `inputs_ids` passed when calling [`MoonshineStreamingModel`].
            hidden_size (`int`, *optional*, defaults to 320):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 1280):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 6):
                Number of hidden layers in the Transformer decoder.
            num_attention_heads (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the Transformer decoder.
            hidden_act (`str`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 4096):
                The maximum sequence length that this model might ever be used with.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*, defaults to 0):
                Padding token id.
            bos_token_id (`int`, *optional*, defaults to 1):
                Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 2):
                End of stream token id.
            rope_parameters (`RopeParameters` or `dict`, *optional*, defaults to `{'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 0.8}`):
                Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
                a value for `rope_theta`, `rope_type`, and optionally `partial_rotary_factor` for partial RoPE application.
            attention_bias (`bool`, *optional*, defaults to `False`):
                Whether to use a bias in the query, key, value and output projection layers during self-attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            decoder_start_token_id (`int`, *optional*):
                The decoder start token id. If not specified, it will default to `bos_token_id`.
            head_dim (`int`, *optional*):
                The attention head dimension. If None, it will default to hidden_size // num_attention_heads.
            pad_head_dim_to_multiple_of (`int`, *optional*):
                If set, the head dimension will be padded to a multiple of this value.
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether to tie weight embeddings
            is_encoder_decoder (`bool`, *optional*, defaults to `True`):
                Whether the model is used as an encoder/decoder or not.

    ```python
    >>> from transformers import MoonshineStreamingModel, MoonshineStreamingConfig

    >>> # Initializing a Moonshine Streaming configuration
    >>> configuration = MoonshineStreamingConfig()

    >>> # Initializing a model from the configuration
    >>> model = MoonshineStreamingModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine_streaming"
    sub_configs = {"encoder_config": MoonshineStreamingEncoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        encoder_config: MoonshineStreamingEncoderConfig = None,
        vocab_size: int = 32768,
        hidden_size: int | None = 320,
        intermediate_size: int | None = 1280,
        num_hidden_layers: int | None = 6,
        num_attention_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int = 4096,
        use_cache: bool | None = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.8,
        },
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        decoder_start_token_id: int | None = None,
        head_dim: int | None = None,
        pad_head_dim_to_multiple_of: int | None = None,
        tie_word_embeddings: bool = False,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = encoder_config.get("model_type", "moonshine_streaming_encoder")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["moonshine_streaming_encoder"]()

        self.encoder_config = encoder_config

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rope_parameters = rope_parameters
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


__all__ = ["MoonshineStreamingConfig", "MoonshineStreamingEncoderConfig"]
