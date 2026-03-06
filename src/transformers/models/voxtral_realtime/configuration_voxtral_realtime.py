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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..mistral.configuration_mistral import MistralConfig


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
class VoxtralRealtimeTextConfig(MistralConfig):
    model_type = "voxtral_realtime_text"


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
class VoxtralRealtimeEncoderConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
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
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=32,
        activation_function="gelu",
        num_mel_bins=128,
        initializer_range=0.02,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=1500,
        rms_norm_eps=1e-05,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        sliding_window=750,
        head_dim=64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.initializer_range = initializer_range
        self.num_key_value_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters
        self.hidden_act = hidden_act
        self.sliding_window = sliding_window
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout

        super().__init__(**kwargs)


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-4B-Realtime-2602")
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
    ```"""

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

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        projector_hidden_act="gelu",
        audio_length_per_tok=8,
        default_num_delay_tokens=6,
        downsample_factor=4,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "voxtral_realtime_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["voxtral_realtime_encoder"]()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "voxtral_realtime_text")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["voxtral_realtime_text"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.hidden_size = text_config.hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.audio_length_per_tok = audio_length_per_tok
        self.default_num_delay_tokens = default_num_delay_tokens
        self.downsample_factor = downsample_factor

        super().__init__(**kwargs)


__all__ = ["VoxtralRealtimeEncoderConfig", "VoxtralRealtimeConfig", "VoxtralRealtimeTextConfig"]
