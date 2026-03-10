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

from ...configuration_utils import PreTrainedConfig, PretrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig
from ..timm_wrapper import TimmWrapperConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/pe-av-large")
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

    def __init__(
        self,
        vision_config: dict | PreTrainedConfig | None = None,
        hidden_size: int | None = 1792,
        intermediate_size: int | None = 4800,
        num_hidden_layers: int | None = 6,
        num_attention_heads: int | None = 14,
        num_key_value_heads: int | None = None,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 10000,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        rope_parameters: RopeParameters | dict | None = {"rope_theta": 20000},
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
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

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "timm_wrapper")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]].from_dict(
                {**self._default_vision_config_kwargs, **vision_config}
            )
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["timm_wrapper"].from_dict(self._default_vision_config_kwargs)

        self.vision_config = vision_config

        super().__init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
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

    def __init__(
        self,
        text_config=None,
        video_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(video_config, dict):
            video_config = PeVideoEncoderConfig(**video_config)
        elif video_config is None:
            video_config = PeVideoEncoderConfig()

        self.text_config = text_config
        self.video_config = video_config

        super().__init__(**kwargs)


__all__ = ["PeVideoEncoderConfig", "PeVideoConfig"]
