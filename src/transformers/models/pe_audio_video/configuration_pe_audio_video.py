# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class PeAudioVideoEncoderConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PeAudioVideoEncoderModel`]. It is used to instantiate a
    PeAudioVideoEncoder model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of pe-av-large.
    e.g. [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        audio_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the audio encoder component.
        video_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the video encoder component.
        hidden_size (`int`, *optional*, defaults to 1792):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4800):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 14):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder.
        max_position_embeddings (`int`, *optional*, defaults to 10000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        rope_parameters (`RopeParameters` or `dict`, *optional*, defaults to `{"rope_theta": 20000}`):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import PeAudioVideoEncoder, PeAudioVideoEncoderConfig

    >>> # Initializing a PeAudioVideoEncoder style configuration
    >>> configuration = PeAudioVideoEncoderConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioVideoEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_audio_video_encoder"
    base_config_key = "audio_video_config"
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    def __init__(
        self,
        audio_config: Optional[Union[dict, PreTrainedConfig]] = None,
        video_config: Optional[Union[dict, PreTrainedConfig]] = None,
        hidden_size: Optional[int] = 1792,
        intermediate_size: Optional[int] = 4800,
        num_hidden_layers: Optional[int] = 6,
        num_attention_heads: Optional[int] = 14,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = 128,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 10000,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        rope_parameters: Optional[Union[RopeParameters, dict]] = {"rope_theta": 20000},
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "pe_audio_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["pe_audio_encoder"]()

        if isinstance(video_config, dict):
            video_config["model_type"] = video_config.get("model_type", "pe_video_encoder")
            video_config = CONFIG_MAPPING[video_config["model_type"]](**video_config)
        elif video_config is None:
            video_config = CONFIG_MAPPING["pe_video_encoder"]()

        self.audio_config = audio_config
        self.video_config = video_config

        super().__init__(**kwargs)


class PeAudioVideoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PeAudioVideoModel`]. It is used to instantiate a
    PeAudioVideoModel model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of pe-av-large.
    e.g. [facebook/pe-av-large](https://huggingface.co/facebook/pe-av-large)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the text model component.
        audio_video_config (`dict` or `PreTrainedConfig`, *optional*):
            Configuration for the audio-video encoder component.

    ```python
    >>> from transformers import PeAudioVideoModel, PeAudioVideoConfig

    >>> # Initializing a PeAudioVideoModel style configuration
    >>> configuration = PeAudioVideoConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_audio_video"
    sub_configs = {"text_config": AutoConfig, "audio_video_config": PeAudioVideoEncoderConfig}

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    def __init__(
        self,
        text_config=None,
        audio_video_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(audio_video_config, dict):
            audio_video_config = PeAudioVideoEncoderConfig(**audio_video_config)
        elif audio_video_config is None:
            audio_video_config = PeAudioVideoEncoderConfig()

        self.text_config = text_config
        self.audio_video_config = audio_video_config

        super().__init__(**kwargs)

    @property
    def audio_config(self):
        return CONFIG_MAPPING["pe_audio"](
            text_config=self.text_config,
            audio_config=self.audio_video_config.audio_config,
        )

    @property
    def video_config(self):
        return CONFIG_MAPPING["pe_video"](
            text_config=self.text_config,
            video_config=self.audio_video_config.video_config,
        )


__all__ = ["PeAudioVideoEncoderConfig", "PeAudioVideoConfig"]
