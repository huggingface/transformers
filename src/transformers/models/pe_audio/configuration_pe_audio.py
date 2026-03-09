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
class PeAudioEncoderConfig(PreTrainedConfig):
    r"""
    dac_config (`Union[PreTrainedConfig, dict]`, *optional*):
        Configuration for the DAC audio encoder used to tokenize the raw audio inputs. If a dictionary is passed, it
        will be used to instantiate a [`~transformers.DacConfig`] with default DAC hyperparameters.

    ```python
    >>> from transformers import PeAudioEncoder, PeAudioEncoderConfig

    >>> # Initializing a PeAudioEncoder style configuration
    >>> configuration = PeAudioEncoderConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_audio_encoder"
    sub_configs = {"dac_config": AutoConfig}
    base_config_key = "audio_video_config"

    _default_dac_config_kwargs = {
        "downsampling_ratios": [2, 8, 10, 12],
        "encoder_hidden_size": 64,
        "codebook_dim": 128,
    }

    def __init__(
        self,
        dac_config: dict | PreTrainedConfig | None = None,
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

        if isinstance(dac_config, dict):
            dac_config["model_type"] = dac_config.get("model_type", "dac")
            dac_config = CONFIG_MAPPING[dac_config["model_type"]](**{**self._default_dac_config_kwargs, **dac_config})
        elif dac_config is None:
            dac_config = CONFIG_MAPPING["dac"](**self._default_dac_config_kwargs)

        self.dac_config = dac_config

        super().__init__(**kwargs)


@auto_docstring(checkpoint="facebook/pe-av-large")
class PeAudioConfig(PretrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PeAudioModel, PeAudioConfig

    >>> # Initializing a PeAudioModel style configuration
    >>> configuration = PeAudioConfig()

    >>> # Initializing a model from the pe-av-large style configuration
    >>> model = PeAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_config": PeAudioEncoderConfig}
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
        audio_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](**self._default_text_config_kwargs)

        if isinstance(audio_config, dict):
            audio_config = PeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = PeAudioEncoderConfig()

        self.text_config = text_config
        self.audio_config = audio_config

        super().__init__(**kwargs)


__all__ = ["PeAudioEncoderConfig", "PeAudioConfig"]
