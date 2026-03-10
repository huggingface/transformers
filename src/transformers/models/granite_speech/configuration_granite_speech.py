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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
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

    def __init__(
        self,
        input_dim=160,
        num_layers=10,
        hidden_dim=1024,
        feedforward_mult=4,
        num_heads=8,
        dim_head=128,
        output_dim=42,
        context_size=200,
        max_pos_emb=512,
        dropout=0.1,
        conv_kernel_size=15,
        conv_expansion_factor=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feedforward_mult = feedforward_mult
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.output_dim = output_dim
        self.context_size = context_size
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.conv_expansion_factor = conv_expansion_factor
        self.max_pos_emb = max_pos_emb


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
class GraniteSpeechConfig(PreTrainedConfig):
    r"""
    has_lora_adapter (`bool`, *optional*, defaults to `True`):
        Indicates whether or not the model has a lora adapter that should only
        be activate when processing audio inputs.
    downsample_rate (`int`, *optional*, defaults to 5):
        Downsample rate for the audio feature extractor.
    projector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Blip2QFormerConfig`):
        The config object or dictionary of the audio projector.
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

    def __init__(
        self,
        text_config=None,
        encoder_config=None,
        projector_config=None,
        audio_token_index=49155,
        initializer_range=0.02,
        has_lora_adapter=True,
        downsample_rate=5,
        window_size=15,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "granite")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["granite"]()

        if isinstance(projector_config, dict):
            projector_config["model_type"] = projector_config.get("model_type", "blip_2_qformer")
            projector_config = CONFIG_MAPPING[projector_config["model_type"]](**projector_config)
        elif projector_config is None:
            projector_config = CONFIG_MAPPING["blip_2_qformer"]()

        if not isinstance(encoder_config, GraniteSpeechEncoderConfig):
            encoder_config = {} if encoder_config is None else encoder_config
            encoder_config = GraniteSpeechEncoderConfig(**encoder_config)

        self.text_config = text_config
        self.encoder_config = encoder_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.initializer_range = initializer_range
        self.has_lora_adapter = has_lora_adapter
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        super().__init__(**kwargs)


__all__ = ["GraniteSpeechEncoderConfig", "GraniteSpeechConfig"]
