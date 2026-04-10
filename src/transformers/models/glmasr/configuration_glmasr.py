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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="zai-org/GLM-ASR-Nano-2512")
@strict
class GlmAsrEncoderConfig(PreTrainedConfig):
    r"""
    Example:

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

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_key_value_heads: int | None = None
    hidden_act: str = "gelu"
    max_position_embeddings: int = 1500
    initializer_range: float = 0.02
    rope_parameters: dict | None = None
    attention_dropout: float | int = 0.0
    num_mel_bins: int = 128

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        kwargs.setdefault("partial_rotary_factor", 0.5)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="zai-org/GLM-ASR-Nano-2512")
@strict
class GlmAsrConfig(PreTrainedConfig):
    r"""
    Example:

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

    audio_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 59260
    projector_hidden_act: str = "gelu"

    def __post_init__(self, **kwargs):
        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "glmasr_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["glmasr_encoder"]()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)

        super().__post_init__(**kwargs)


__all__ = ["GlmAsrEncoderConfig", "GlmAsrConfig"]
