# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class VoxtralEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxtralEncoder`]. It is used to instantiate a
    Voxtral audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the Voxtral
    architecture.

    e.g. [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51866):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by dividing by sqrt(hidden_size) if True.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu",
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `VoxtralProcessor` class.
        max_source_positions (`int`, *optional*, defaults to 1500):
            The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import VoxtralEncoderConfig, VoxtralEncoder

    >>> # Initializing a VoxtralEncoderConfig
    >>> configuration = VoxtralEncoderConfig()

    >>> # Initializing a VoxtralEncoder (with random weights)
    >>> model = VoxtralEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "voxtral_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    def __init__(
        self,
        vocab_size=51866,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        scale_embedding=False,
        activation_function="gelu",
        num_mel_bins=128,
        max_source_positions=1500,
        initializer_range=0.02,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(hidden_size) if True
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.initializer_range = initializer_range

        # TODO: @eustlb, we do not use dropout and layerdrop, yet we need to hardcode them
        # to be able to use Whisper with modular (here actually from Qwen2-Audio and copied from).
        # After a future Whisper refactor, we should remove this.
        self.dropout = 0.0
        self.layerdrop = 0.0
        self.activation_dropout = 0.0

        self.attention_dropout = attention_dropout


class VoxtralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VoxtralForConditionalGeneration`]. It is used to instantiate an
    Voxtral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Voxtral-Mini-3B.

    e.g. [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        audio_token_id (`int`, *optional*):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function (function or string) in the multi-modal projector.

    ```python
    >>> from transformers import VoxtralForConditionalGeneration, VoxtralConfig

    >>> # Initializing a Voxtral configuration
    >>> configuration = VoxtralConfig(audio_token_id=24, projector_hidden_act="gelu")

    >>> # Initializing a 3B model with random weights
    >>> model = VoxtralForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "voxtral"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 131072,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 100000000.0,
        "head_dim": 128,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=None,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"] if "model_type" in audio_config else "voxtral_encoder"
            )
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["voxtral_encoder"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.audio_token_id = audio_token_id
        self.projector_hidden_act = projector_hidden_act

        super().__init__(**kwargs)


__all__ = ["VoxtralEncoderConfig", "VoxtralConfig"]
