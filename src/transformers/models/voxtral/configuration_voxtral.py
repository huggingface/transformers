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
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-3B-2507")
@strict
class VoxtralEncoderConfig(PreTrainedConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

    ```python
    >>> from transformers import VoxtralEncoderConfig, VoxtralEncoder

    >>> # Initializing a VoxtralEncoderConfig
    >>> configuration = VoxtralEncoderConfig()

    >>> # Initializing a VoxtralEncoder (with random weights)
    >>> model = VoxtralEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "voxtral_encoder"

    attribute_map = {
        "d_model": "hidden_size",
        "encoder_layers": "num_hidden_layers",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_ffn_dim": "intermediate_size",
        "encoder_layerdrop": "layerdrop",
    }

    vocab_size: int = 51866
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    scale_embedding: bool = False
    activation_function: str = "gelu"
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    initializer_range: float = 0.02
    attention_dropout: float | int = 0.0

    # TODO: @eustlb, we do not use dropout and layerdrop, yet we need to hardcode them
    # to be able to use Whisper with modular (here actually from Qwen2-Audio and copied from).
    # After a future Whisper refactor, we should remove this.
    dropout: float | int = 0.0
    layerdrop: float | int = 0.0
    activation_dropout: float | int = 0.0


@auto_docstring(checkpoint="mistralai/Voxtral-Mini-3B-2507")
@strict
class VoxtralConfig(PreTrainedConfig):
    r"""
    Example:

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

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int | None = None
    projector_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "voxtral_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["voxtral_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)

        self.hidden_size = self.text_config.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["VoxtralEncoderConfig", "VoxtralConfig"]
