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


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioVideoEncoderConfig(PreTrainedConfig):
    r"""
    video_config (`Union[PreTrainedConfig, dict]`, *optional*):
        Configuration for the video encoder. If a dictionary is provided, it is used to instantiate
        [`~transformers.PeVideoEncoderConfig`].

    ```python
    >>> from transformers import PeAudioVideoEncoder, PeAudioVideoEncoderConfig

    >>> # Initializing a PeAudioVideoEncoder style configuration
    >>> configuration = PeAudioVideoEncoderConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioVideoEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "pe_audio_video_encoder"
    base_config_key = "audio_video_config"
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    audio_config: dict | PreTrainedConfig | None = None
    video_config: dict | PreTrainedConfig | None = None
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

        if isinstance(self.audio_config, dict):
            self.audio_config["model_type"] = self.audio_config.get("model_type", "pe_audio_encoder")
            self.audio_config = CONFIG_MAPPING[self.audio_config["model_type"]](**self.audio_config)
        elif self.audio_config is None:
            self.audio_config = CONFIG_MAPPING["pe_audio_encoder"]()

        if isinstance(self.video_config, dict):
            self.video_config["model_type"] = self.video_config.get("model_type", "pe_video_encoder")
            self.video_config = CONFIG_MAPPING[self.video_config["model_type"]](**self.video_config)
        elif self.video_config is None:
            self.video_config = CONFIG_MAPPING["pe_video_encoder"]()

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 20000}
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
@strict
class PeAudioVideoConfig(PretrainedConfig):
    r"""
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

    text_config: dict | PreTrainedConfig | None = None
    audio_video_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "modernbert")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](
                **{**self._default_text_config_kwargs, **self.text_config}
            )
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(self.audio_video_config, dict):
            self.audio_video_config = PeAudioVideoEncoderConfig(**self.audio_video_config)
        elif self.audio_video_config is None:
            self.audio_video_config = PeAudioVideoEncoderConfig()

        super().__post_init__(**kwargs)

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
