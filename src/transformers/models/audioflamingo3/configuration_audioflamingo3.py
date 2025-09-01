# coding=utf-8
# Copyright 2024 NVIDIA CORPORATION and The HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class AudioFlamingo3EncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioFlamingo3Encoder`]. It is used to instantiate an
    AudioFlamingo3 audio encoder according to the specified arguments, defining the model architecture. Instantiating a
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
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        d_model (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3EncoderConfig, AudioFlamingo3Encoder

    >>> # Initializing a AudioFlamingo3EncoderConfig
    >>> configuration = AudioFlamingo3EncoderConfig()

    >>> # Initializing a AudioFlamingo3Encoder (with random weights)
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
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions


class AudioFlamingo3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AudioFlamingo3ForConditionalGeneration`]. It is used to instantiate an
    AudioFlamingo3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AudioFlamingo3 architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        llm_cfg (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        sound_tower_cfg (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio backbone.
        sound_mm_projector_cfg (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the multi-modal projector.

    Example:

    ```python
    >>> from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Config, AudioFlamingo3EncoderConfig

    >>> # Initializing a AudioFlamingo3Encoder config
    >>> audio_config = AudioFlamingo3EncoderConfig()

    >>> # Initializing a AudioFlamingo3 configuration
    >>> configuration = AudioFlamingo3Config(sound_tower_cfg=audio_config)

    >>> # Initializing a model from the audioflamingo3 style configuration
    >>> model = AudioFlamingo3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "audioflamingo3"
    sub_configs = {"llm_cfg": AutoConfig, "sound_tower_cfg": AutoConfig}

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        **kwargs,
    ) -> None:

        if isinstance(sound_tower_cfg, dict):
            sound_tower_cfg["model_type"] = sound_tower_cfg.get("model_type", "audioflamingo3_encoder")
            sound_tower_cfg = CONFIG_MAPPING[sound_tower_cfg["model_type"]](**sound_tower_cfg)
        elif sound_tower_cfg is None:
            sound_tower_cfg = CONFIG_MAPPING["audioflamingo3_encoder"](
                d_model=1280,
                encoder_attention_heads=20,
                encoder_ffn_dim=5120,
                encoder_layerdrop=0.0,
                encoder_layers=32,
                num_mel_bins=128,
                max_source_positions=1500,
                scale_embedding=False,
                activation_function="gelu",
            )

        self.sound_tower_cfg = sound_tower_cfg

        if isinstance(llm_cfg, dict):
            llm_cfg["model_type"] = llm_cfg.get("model_type", "qwen2")
            llm_cfg = CONFIG_MAPPING[llm_cfg["model_type"]](**llm_cfg)
        elif llm_cfg is None:
            llm_cfg = CONFIG_MAPPING["qwen2"]()

        self.llm_cfg = llm_cfg

        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        super().__init__(**kwargs)


__all__ = ["AudioFlamingo3Config", "AudioFlamingo3EncoderConfig"]
