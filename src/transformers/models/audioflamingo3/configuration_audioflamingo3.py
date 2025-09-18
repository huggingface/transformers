# coding=utf-8
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

"""AudioFlamingo3 model configuration"""

from typing import Any, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class AudioFlamingo3EncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AudioFlamingo3Encoder`]. It is used to instantiate an
    AudioFlamingo3 audio encoder according to the specified arguments, defining the model architecture. Instantiating an
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the AudioFlamingo3
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `AudioFlamingo3Processor` class.
        encoder_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        encoder_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556)
            for more details.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(d_model).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        avg_pool_kernel_size (`int`, *optional*, defaults to 2):
            Kernel size for the average pooling layer applied after the transformer encoder.
        avg_pool_stride (`int`, *optional*, defaults to 2):
            Stride for the average pooling layer applied after the transformer encoder.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3EncoderConfig, AudioFlamingo3Encoder

    >>> # Initializing an AudioFlamingo3EncoderConfig
    >>> configuration = AudioFlamingo3EncoderConfig()

    >>> # Initializing an AudioFlamingo3Encoder (with random weights)
    >>> model = AudioFlamingo3Encoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        encoder_layerdrop: float = 0.0,
        activation_function: str = "gelu",
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        scale_embedding: bool = False,
        max_source_positions: int = 1500,
        avg_pool_kernel_size: int = 2,
        avg_pool_stride: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.avg_pool_kernel_size = avg_pool_kernel_size
        self.avg_pool_stride = avg_pool_stride


class AudioFlamingo3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AudioFlamingo3ForConditionalGeneration`]. It is used to instantiate an
    AudioFlamingo3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AudioFlamingo3.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AudioFlamingo3EncoderConfig, dict]`, *optional*, defaults to `AudioFlamingo3EncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151669):
            The audio token index to encode the audio prompt.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Config, AudioFlamingo3EncoderConfig, Qwen2Config

    >>> # Initializing an AudioFlamingo3Encoder config
    >>> audio_config = AudioFlamingo3EncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing an AudioFlamingo3 configuration
    >>> configuration = AudioFlamingo3Config(audio_config, text_config)

    >>> # Initializing a model from the audioflamingo3 style configuration
    >>> model = AudioFlamingo3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3"
    sub_configs = {
        "audio_config": AudioFlamingo3EncoderConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        audio_config: Optional[Union[AudioFlamingo3EncoderConfig, dict[str, Any]]] = None,
        text_config: Optional[Union[AutoConfig, dict[str, Any]]] = None,
        audio_token_id: int = 151669,
        **kwargs,
    ):
        self.audio_token_id = audio_token_id

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "audioflamingo3_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["audioflamingo3_encoder"]()

        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
