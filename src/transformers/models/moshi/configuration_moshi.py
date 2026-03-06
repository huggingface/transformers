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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kmhf/hf-moshiko")
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

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        input_size=4096,
        num_hidden_layers=6,
        num_attention_heads=16,
        num_key_value_heads=None,
        audio_vocab_size=2048,
        max_position_embeddings=9,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=8,
        attention_dropout=0.0,
        ffn_dim=5632,
        rms_norm_eps=1e-8,
        num_codebooks=8,
        tie_word_embeddings=False,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        if ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={ffn_dim}` must be even.")
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_codebooks = num_codebooks
        self.audio_vocab_size = audio_vocab_size

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


@auto_docstring(checkpoint="kmhf/hf-moshiko")
class MoshiConfig(PreTrainedConfig):
    """
    audio_vocab_size (`int`, *optional*):
        Vocabulary size of the audio part of model. Defines the number of different tokens that can be
        represented by the `audio_codes` passed when calling the Moshi models.
    ffn_dim (`int`, *optional*, defaults to 22528):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.

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

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        audio_vocab_size: int | None = None,
        max_position_embeddings: int | None = 3000,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        hidden_act: str | None = "silu",
        head_dim: int | None = None,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        sliding_window: int | None = 3000,
        attention_dropout: float | None = 0.0,
        ffn_dim: int | None = 22528,
        rms_norm_eps: int | None = 1e-8,
        num_codebooks: int | None = 8,
        tie_word_embeddings: bool | None = False,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        if ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={ffn_dim}` must be even.")
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_codebooks = num_codebooks
        self.rope_parameters = rope_parameters

        audio_encoder_config = kwargs.pop("audio_encoder_config", {})
        audio_encoder_model_type = audio_encoder_config.pop("model_type", "mimi")

        self.audio_encoder_config = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)

        if self.num_codebooks > self.audio_encoder_config.num_codebooks:
            raise ValueError(
                f"`num_codebooks={num_codebooks}` is greater than the maximum number of codebooks that the audio encoder can deal with ({self.audio_encoder_config.num_codebooks}). Please lower it."
            )

        self.audio_vocab_size = (
            self.audio_encoder_config.codebook_size if audio_vocab_size is None else audio_vocab_size
        )

        depth_decoder_config = kwargs.pop("depth_decoder_config", {})
        depth_decoder_config.update(
            {
                "audio_vocab_size": self.audio_vocab_size,
                "input_size": hidden_size,
                "vocab_size": vocab_size,
                "num_codebooks": num_codebooks,
            }
        )

        self.depth_decoder_config = MoshiDepthConfig(**depth_decoder_config)

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

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
