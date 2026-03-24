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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper import TimmWrapperConfig


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeVideoEncoderConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PeAudioEncoder, PeAudioEncoderConfig

    >>> # Initializing a PeAudioEncoder style configuration
    >>> configuration = PeAudioEncoderConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_video_encoder"
    sub_configs = {"vision_config": TimmWrapperConfig}
    base_config_key = "audio_video_config"

    _default_vision_config_kwargs = {
        "architecture": "vit_pe_core_large_patch14_336",
        "do_pooling": True,
        "num_classes": 1024,
        "global_pool": "map",
        "initializer_range": 0.02,
    }

    vision_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 1792
    intermediate_size: int = 4800
    num_hidden_layers: int = 6
    num_attention_heads: int = 14
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 10000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 20000}

        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "timm_wrapper")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]].from_dict(
                {**self._default_vision_config_kwargs, **self.vision_config}
            )
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["timm_wrapper"].from_dict(self._default_vision_config_kwargs)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeVideoConfig(PretrainedConfig):
    r"""
    video_config (`dict` or `PreTrainedConfig`, *optional*):
        Configuration for the video encoder component.

    ```python
    >>> from transformers import PeVideoModel, PeVideoConfig

    >>> # Initializing a PeVideoModel style configuration
    >>> configuration = PeVideoConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeVideoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_video"
    sub_configs = {"text_config": AutoConfig, "video_config": PeVideoEncoderConfig}
    base_config_key = "audio_video_config"

    _default_text_config_kwargs = {
        "model_type": "modernbert",
        "hidden_size": 1024,
        "intermediate_size": 2624,
        "num_hidden_layers": 22,
        "num_attention_heads": 16,
    }

    text_config: dict | PreTrainedConfig | None = None
    video_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "modernbert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(self.video_config, dict):
            self.video_config = PeVideoEncoderConfig(**self.video_config)
        elif self.video_config is None:
            self.video_config = PeVideoEncoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["PeVideoEncoderConfig", "PeVideoConfig"]
