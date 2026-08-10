# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Musicgen Melody model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="facebook/musicgen-melody")
@strict
class MusicgenMelodyDecoderConfig(PreTrainedConfig):
    r"""
    audio_channels (`int`, *optional*, defaults to 1):
        Number of audio channels used by the model (either mono or stereo). Stereo models generate a separate
        audio stream for the left/right output channels. Mono models generate a single audio stream output.
    """

    model_type = "musicgen_melody_decoder"
    base_config_key = "decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 2048
    max_position_embeddings: int = 2048
    num_hidden_layers: int = 24
    ffn_dim: int = 4096
    num_attention_heads: int = 16
    layerdrop: float | int = 0.0
    use_cache: bool = True
    activation_function: str = "gelu"
    hidden_size: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    initializer_factor: float = 0.02
    scale_embedding: bool = False
    num_codebooks: int = 4
    audio_channels: int = 1
    pad_token_id: int | None = 2048
    bos_token_id: int | None = 2048
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    is_decoder: bool = False
    add_cross_attention: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {self.audio_channels} channels.")


@auto_docstring(checkpoint="facebook/musicgen-melody")
@strict
class MusicgenMelodyConfig(PreTrainedConfig):
    r"""
    text_encoder (`Union[dict, `PretrainedConfig`]`):
        An instance of a configuration object that defines the text encoder config.
    audio_encoder (`Union[dict, `PretrainedConfig`]`):
        An instance of a configuration object that defines the audio encoder config.
    decoder (`Union[dict, `PretrainedConfig`]`):
        An instance of a configuration object that defines the decoder config.
    num_chroma (`int`, *optional*, defaults to 12):
        Number of chroma bins to use.
    chroma_length (`int`, *optional*, defaults to 235):
        Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig(
    ...     text_encoder=text_encoder_config, audio_encoder=audio_encoder_config, decoder=decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")
    >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)
    ```"""

    model_type = "musicgen_melody"
    sub_configs = {
        "text_encoder": AutoConfig,
        "audio_encoder": AutoConfig,
        "decoder": MusicgenMelodyDecoderConfig,
    }
    has_no_defaults_at_init = True

    text_encoder: dict | PreTrainedConfig = None
    audio_encoder: dict | PreTrainedConfig = None
    decoder: dict | PreTrainedConfig = None
    num_chroma: int = 12
    chroma_length: int = 235
    initializer_factor: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.text_encoder, dict):
            text_encoder_model_type = self.text_encoder.pop("model_type")
            self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **self.text_encoder)
        elif self.text_encoder is None:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because text_encoder is not passed"
            )

        if isinstance(self.audio_encoder, dict):
            audio_encoder_model_type = self.audio_encoder.pop("model_type")
            self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder)
        elif self.audio_encoder is None:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because audio_encoder is not passed"
            )

        if isinstance(self.decoder, dict):
            self.decoder = MusicgenMelodyDecoderConfig(**self.decoder)
        elif self.decoder is None:
            self.decoder = MusicgenMelodyDecoderConfig()

        self.is_encoder_decoder = True
        super().__post_init__(**kwargs)

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate


__all__ = ["MusicgenMelodyConfig", "MusicgenMelodyDecoderConfig"]
