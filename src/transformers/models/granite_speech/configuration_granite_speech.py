# Copyright 2025 The HuggingFace Inc. team.
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
"""Config class for Granite Speech."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.3-2b")
@strict
class GraniteSpeechEncoderConfig(PreTrainedConfig):
    r"""
    feedforward_mult (`int`, *optional*, defaults to 4):
        Multiplier for the up/down projections in the encoder's feedforward layers;
        The projections will have intermediate dim of size `hidden_dim * feedforward_mult`.
    output_dim (`int`, *optional*, defaults to 42):
        Intermediate dimension of the feedforward projections in the conformer
        to be added to every other encoder block's output.
    context_size (`int`, *optional*, defaults to 200):
        Context size to be used in conformer attention.
    max_pos_emb (`int`, *optional*, defaults to 512):
        Max pos embeds to be used in attention (shaw's relative positional encoding).
    conv_expansion_factor (`int`, *optional*, defaults to 2):
        Intermediate dimension to be used in conformer convolutions.

    Example:

    ```python
    >>> from transformers import GraniteSpeechEncoderConfig, GraniteSpeechCTCEncoder

    >>> # Initializing a GraniteSpeechEncoderConfig
    >>> configuration = GraniteSpeechEncoderConfig()

    >>> # Initializing a GraniteSpeechCTCEncoder (with random weights)
    >>> model = GraniteSpeechCTCEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech_encoder"

    input_dim: int = 160
    num_layers: int = 10
    hidden_dim: int = 1024
    feedforward_mult: int = 4
    num_heads: int = 8
    dim_head: int = 128
    output_dim: int = 42
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float | int = 0.1
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.3-2b")
@strict
class GraniteSpeechConfig(PreTrainedConfig):
    r"""
    projector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Blip2QFormerConfig`):
        The config object or dictionary of the audio projector.
    has_lora_adapter (`bool`, *optional*, defaults to `True`):
        Indicates whether or not the model has a lora adapter that should only
        be activate when processing audio inputs.
    downsample_rate (`int`, *optional*, defaults to 5):
        Downsample rate for the audio feature extractor.
    window_size (`int`, *optional*, defaults to 15):
        Window size for the audio feature projector.

    Example:

    ```python
    >>> from transformers import GraniteSpeechConfig, GraniteSpeechForConditionalGeneration

    >>> # Initializing a GraniteSpeechConfig
    >>> configuration = GraniteSpeechConfig()

    >>> # Initializing a GraniteSpeechForConditionalGeneration (with random weights)
    >>> model = GraniteSpeechForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_speech"
    attribute_map = {
        "audio_token_id": "audio_token_index",
    }
    sub_configs = {
        "text_config": AutoConfig,
        "encoder_config": GraniteSpeechEncoderConfig,
        "projector_config": AutoConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None
    projector_config: dict | PreTrainedConfig | None = None
    audio_token_index: int = 49155
    initializer_range: float = 0.02
    has_lora_adapter: bool = True
    downsample_rate: int = 5
    window_size: int = 15

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "granite")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["granite"]()

        if isinstance(self.projector_config, dict):
            self.projector_config["model_type"] = self.projector_config.get("model_type", "blip_2_qformer")
            self.projector_config = CONFIG_MAPPING[self.projector_config["model_type"]](**self.projector_config)
        elif self.projector_config is None:
            self.projector_config = CONFIG_MAPPING["blip_2_qformer"]()

        if not isinstance(self.encoder_config, GraniteSpeechEncoderConfig):
            self.encoder_config = {} if self.encoder_config is None else self.encoder_config
            self.encoder_config = GraniteSpeechEncoderConfig(**self.encoder_config)

        super().__post_init__(**kwargs)


__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechConfig"]
