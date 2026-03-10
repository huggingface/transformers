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


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/pe-av-large")
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
    ```"""

    model_type = "pe_audio_video_encoder"
    base_config_key = "audio_video_config"
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    def __init__(
        self,
        audio_config: dict | PreTrainedConfig | None = None,
        video_config: dict | PreTrainedConfig | None = None,
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


@auto_docstring(checkpoint="facebook/pe-av-large")
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
