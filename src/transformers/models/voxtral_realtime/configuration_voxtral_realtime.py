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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..mistral.configuration_mistral import MistralConfig


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
@strict
class VoxtralRealtimeTextConfig(MistralConfig):
    model_type = "voxtral_realtime_text"


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
@strict
class VoxtralRealtimeEncoderConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import VoxtralRealtimeEncoderConfig, VoxtralRealtimeEncoder

    >>> # Initializing a VoxtralRealtimeEncoderConfig
    >>> configuration = VoxtralRealtimeEncoderConfig()

    >>> # Initializing a VoxtralRealtimeEncoder (with random weights)
    >>> model = VoxtralRealtimeEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "voxtral_realtime_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
        "num_key_value_heads": "num_attention_heads",
    }

    vocab_size: int = 131072
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    activation_function: str = "gelu"
    num_mel_bins: int = 128
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0
    hidden_act: str = "silu"
    max_position_embeddings: int = 1500
    rms_norm_eps: float = 1e-05
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int = 750
    head_dim: int = 64

    def __post_init__(self, **kwargs):
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
@strict
class VoxtralRealtimeConfig(PreTrainedConfig):
    r"""
    audio_length_per_tok (`int`, *optional*, defaults to 8):
        The number of audio frames corresponding to each text token.
    default_num_delay_tokens (`int`, *optional*, defaults to 6):
        The default number of delay tokens used for streaming.
    downsample_factor (`int`, *optional*, defaults to 4):
        The downsampling factor applied to audio features before projection.

    ```python
    >>> from transformers import VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeConfig

    >>> # Initializing a VoxtralRealtime configuration
    >>> configuration = VoxtralRealtimeConfig()

    >>> # Initializing a model with random weights
    >>> model = VoxtralRealtimeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "voxtral_realtime"
    sub_configs = {"text_config": AutoConfig, "audio_config": AutoConfig}

    _default_text_config_kwargs = {
        "vocab_size": 131072,
        "hidden_size": 3072,
        "intermediate_size": 9216,
        "num_hidden_layers": 26,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-05,
        "use_cache": True,
        "rope_theta": 1000000.0,
        "head_dim": 128,
        "tie_word_embeddings": True,
        "sliding_window": 8192,
    }

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    projector_hidden_act: str = "gelu"
    audio_length_per_tok: int = 8
    default_num_delay_tokens: int = 6
    downsample_factor: int = 4

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "voxtral_realtime_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["voxtral_realtime_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "voxtral_realtime_text")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["voxtral_realtime_text"](**self._default_text_config_kwargs)

        self.hidden_size = self.text_config.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["VoxtralRealtimeEncoderConfig", "VoxtralRealtimeConfig", "VoxtralRealtimeTextConfig"]
