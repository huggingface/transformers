# Copyright 2025 the HuggingFace Team. All rights reserved.
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
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ..auto import CONFIG_MAPPING, AutoConfig


class GlmAsrEncoderConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`GlmAsrEncoder`]. It is used to instantiate a
    glmasr audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the glmasr
    architecture.

    e.g. [zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1280):
            Dimensionality of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5120):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        max_position_embeddings (`int`, *optional*, defaults to 1500):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of mel features used per input features. Should correspond to the value used in the
            `GlmAsrProcessor` class.

    ```python
    >>> from transformers import GlmAsrEncoderConfig, GlmAsrEncoder

    >>> # Initializing a GlmAsrEncoderConfig
    >>> configuration = GlmAsrEncoderConfig()

    >>> # Initializing a GlmAsrEncoder (with random weights)
    >>> model = GlmAsrEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glmasr_encoder"

    def __init__(
        self,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        num_key_value_heads=None,
        hidden_act="gelu",
        max_position_embeddings=1500,
        initializer_range=0.02,
        rope_parameters=None,
        attention_dropout=0.0,
        num_mel_bins=128,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.num_mel_bins = num_mel_bins

        kwargs.setdefault("partial_rotary_factor", 0.5)
        super().__init__(**kwargs)


class GlmAsrConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmAsrForConditionalGeneration`]. It is used to instantiate an
    glmasr model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the glmasr-Mini-3B.

    e.g. [zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        audio_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the audio encoder.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text model.
        audio_token_id (`int`, *optional*, defaults to 59260):
            The audio token index to encode the audio prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function (function or string) in the multi-modal projector.

    ```python
    >>> from transformers import GlmAsrForConditionalGeneration, GlmAsrConfig

    >>> # Initializing a glmasr configuration
    >>> configuration = GlmAsrConfig()

    >>> # Initializing a GLM-ASR-Nano-2512 model with random weights
    >>> model = GlmAsrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glmasr"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 59264,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "eos_token_id": [59246, 59253, 59255],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=59260,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "glmasr_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["glmasr_encoder"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
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


__all__ = ["GlmAsrEncoderConfig", "GlmAsrConfig"]
