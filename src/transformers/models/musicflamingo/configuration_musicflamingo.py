# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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

from ...utils import logging
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3Config, AudioFlamingo3EncoderConfig
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


class MusicFlamingoEncoderConfig(AudioFlamingo3EncoderConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicFlamingoEncoder`].

    e.g. [nvidia/music-flamingo-hf](https://huggingface.co/nvidia/music-flamingo-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `MusicFlamingoProcessor` class.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of encoder layers.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556)
            for more details.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(hidden_size).
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    Example:

    ```python
    >>> from transformers import MusicFlamingoEncoderConfig, MusicFlamingoEncoder

    >>> # Initializing an MusicFlamingoEncoderConfig
    >>> configuration = MusicFlamingoEncoderConfig()

    >>> # Initializing an MusicFlamingoEncoder (with random weights)
    >>> model = MusicFlamingoEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "musicflamingo_encoder"


class MusicFlamingoConfig(AudioFlamingo3Config):
    r"""
    This is the configuration class to store the configuration of a [`MusicFlamingoForConditionalGeneration`].

    e.g. [nvidia/music-flamingo-hf](https://huggingface.co/nvidia/music-flamingo-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[MusicFlamingoEncoderConfig, dict]`, *optional*, defaults to `MusicFlamingoEncoderConfig`):
            The config object or dictionary of the audio backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        audio_token_id (`int`, *optional*, defaults to 151669):
            The audio token index to encode the audio prompt.
        audio_bos_token_id (`int`, *optional*):
            The beginning-of-audio token index used to mark the start of audio spans.
        audio_eos_token_id (`int`, *optional*):
            The end-of-audio token index used to mark the end of audio spans.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            Activation function used in the projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to include bias terms in the projector.

    Example:

    ```python
    >>> from transformers import MusicFlamingoForConditionalGeneration, MusicFlamingoConfig, MusicFlamingoEncoderConfig, Qwen2Config

    >>> # Initializing an MusicFlamingoEncoder config
    >>> audio_config = MusicFlamingoEncoderConfig()

    >>> # Initializing a Qwen2 config
    >>> text_config = Qwen2Config()

    >>> # Initializing an MusicFlamingo configuration
    >>> configuration = MusicFlamingoConfig(audio_config, text_config)

    >>> # Initializing a model from the musicflamingo style configuration
    >>> model = MusicFlamingoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "musicflamingo"
    sub_configs = {
        "audio_config": MusicFlamingoEncoderConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151669,
        audio_bos_token_id=None,
        audio_eos_token_id=None,
        projector_hidden_act="gelu",
        projector_bias=True,
        **kwargs,
    ):
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "musicflamingo_encoder")
        elif audio_config is None:
            audio_config = {"model_type": "musicflamingo_encoder"}

        super().__init__(
            audio_config=audio_config,
            text_config=text_config,
            audio_token_id=audio_token_id,
            projector_hidden_act=projector_hidden_act,
            projector_bias=projector_bias,
            **kwargs,
        )


__all__ = ["MusicFlamingoConfig", "MusicFlamingoEncoderConfig"]
