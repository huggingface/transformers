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
"""Moshi model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiDepthConfig(PreTrainedConfig):
    r"""
    input_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the input hidden states. Used to connect the main decoder to the depth decoder.
    audio_vocab_size (`int`, *optional*, defaults to 2048):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the Moshi models.
    ffn_dim (`int`, *optional*, defaults to 5632):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the depth decoder block. Must be even.

    Example:

    ```python
    >>> from transformers import (
    ...     MoshiDepthConfig,
    ...     MoshiDepthDecoder,
    ... )

    >>> configuration = MoshiDepthConfig()

    >>> # Initializing a MoshiDepthDecoder (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiDepthDecoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moshi_depth"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 1024
    input_size: int = 4096
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    audio_vocab_size: int = 2048
    max_position_embeddings: int = 9
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int = 8
    attention_dropout: float | int = 0.0
    ffn_dim: int = 5632
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 8
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={self.ffn_dim}` must be even.")


@auto_docstring(checkpoint="kmhf/hf-moshiko")
@strict
class MoshiConfig(PreTrainedConfig):
    r"""
    audio_vocab_size (`int`, *optional*):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the Moshi models.
    ffn_dim (`int`, *optional*, defaults to 22528):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.
    audio_encoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the audio encoder.
    depth_decoder_config (`PreTrainedConfig | dict`, *optional*):
        Configuration for the depth decoder.

    Example:

    ```python
    >>> from transformers import (
    ...     MoshiConfig,
    ...     MoshiForConditionalGeneration,
    ... )

    >>> configuration = MoshiConfig()

    >>> # Initializing a MoshiForConditionalGeneration (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("kmhf/hf-moshiko")

    >>> # loading model and config from pretrained folder
    >>> moshi_config = MoshiConfig.from_pretrained("kmhf/hf-moshiko")
    >>> model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko", config=moshi_config)
    ```"""

    model_type = "moshi"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"audio_encoder_config": AutoConfig, "depth_decoder_config": MoshiDepthConfig}

    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    audio_vocab_size: int | None = None
    max_position_embeddings: int = 3000
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int = 3000
    attention_dropout: float | int = 0.0
    ffn_dim: int = 22528
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 8
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    depth_decoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = (
            self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads
        )
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

        if isinstance(self.audio_encoder_config, dict):
            audio_encoder_model_type = self.audio_encoder_config.pop("model_type", "mimi")
            self.audio_encoder_config = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder_config)
        elif self.audio_encoder_config is None:
            self.audio_encoder_config = AutoConfig.for_model("mimi")

        self.audio_vocab_size = (
            self.audio_encoder_config.codebook_size if self.audio_vocab_size is None else self.audio_vocab_size
        )

        if isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config.update(
                {
                    "audio_vocab_size": self.audio_vocab_size,
                    "input_size": self.hidden_size,
                    "vocab_size": self.vocab_size,
                    "num_codebooks": self.num_codebooks,
                }
            )
            self.depth_decoder_config = MoshiDepthConfig(**self.depth_decoder_config)
        elif self.depth_decoder_config is None:
            self.depth_decoder_config = MoshiDepthConfig()
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={self.ffn_dim}` must be even.")

        if self.num_codebooks > self.audio_encoder_config.num_codebooks:
            raise ValueError(
                f"`num_codebooks={self.num_codebooks}` is greater than the maximum number of codebooks that the audio encoder can deal with ({self.audio_encoder_config.num_codebooks}). Please lower it."
            )

    @property
    def sampling_rate(self):
        return self.audio_encoder_config.sampling_rate

    @classmethod
    def from_audio_encoder_config(
        cls,
        audio_encoder_config: PreTrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MoshiConfig`] (or a derived class) from an audio encoder configuration.

        Returns:
            [`MoshiConfig`]: An instance of a configuration object
        """

        return cls(
            audio_encoder_config=audio_encoder_config.to_dict(),
            **kwargs,
        )


__all__ = ["MoshiConfig", "MoshiDepthConfig"]
