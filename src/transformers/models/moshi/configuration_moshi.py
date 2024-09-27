# coding=utf-8
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class MoshiConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoshiModel`]. It is used to instantiate a
    Moshi model according to the specified arguments, defining the audio encoder, Moshi depth decoder and Moshi decoder
    configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MoshiDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MoshiDecoder`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the layers and the pooler layer of the main decoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the main decoder block.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 3000):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        sliding_window (`int`, *optional*, defaults to 3000):
            Sliding window attention window size. If not specified, will default to `3000`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_dim (`int`, *optional*, defaults to 22528):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.
        num_codebooks (`int`, *optional*, defaults to 8):
            The number of audio codebooks for each audio channels.
        rms_norm_eps (`float`, *optional*, defaults to 1e-8):
            The epsilon used by the rms normalization layers.
        depth_hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer of the depth decoder.
        depth_num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of depth decoder layers.
        depth_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the depth decoder block.
        depth_max_position_embeddings (`int`, *optional*, defaults to 8):
            The maximum sequence length that the depth decoder model might ever be used with. Typically, set this to the
            number of codebooks.
        depth_ffn_dim (`int`, *optional*, defaults to 5632):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the depth decoder block. Must be even.
        depth_head_dim (`int`, *optional*, defaults to `depth_hidden_size // depth_num_attention_heads`):
            The attention head dimension of the depth encoder layers.
        depth_num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention in the depth decoder.
            If it is not specified, will default to `depth_num_key_value_heads`.
        depth_sliding_window (`int`, *optional*, defaults to 8):
            Sliding window attention window size. If not specified, will default to `8`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether input and output word embeddings should be tied.
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.


    Example:

    ```python # TODO(YL): update
    >>> from transformers import (
    ...     MoshiConfig,
    ...     EncodecConfig,
    ...     MoshiForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> audio_encoder_config = EncodecConfig()

    >>> configuration = MoshiConfig.from_sub_models_config(
    ...     audio_encoder_config
    ... )

    >>> # Initializing a MoshiForConditionalGeneration (with random weights) from the kmhf/hf-moshiko style configuration
    >>> model = MoshiForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("moshi-model")

    >>> # loading model and config from pretrained folder
    >>> moshi_config = MoshiConfig.from_pretrained("moshi-model")
    >>> model = MoshiForConditionalGeneration.from_pretrained("moshi-model", config=moshi_config)
    ```"""

    model_type = "moshi"
    is_composition = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        audio_vocab_size=None,  # TODO
        max_position_embeddings=3000,
        rope_theta=10000.0,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=3000,
        attention_dropout=0.0,
        ffn_dim=22528,
        rms_norm_eps=1e-8,
        num_codebooks=8,
        depth_hidden_size=1024,
        depth_num_hidden_layers=6,
        depth_max_position_embeddings=8,
        depth_num_attention_heads=16,
        depth_ffn_dim=5632,
        depth_head_dim=None,
        depth_num_key_value_heads=None,
        depth_sliding_window=8,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
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

        self.depth_hidden_size = depth_hidden_size
        self.depth_num_hidden_layers = depth_num_hidden_layers
        self.depth_max_position_embeddings = depth_max_position_embeddings
        self.depth_num_attention_heads = depth_num_attention_heads
        if depth_ffn_dim % 2 == 1:
            raise ValueError(f"`depth_ffn_dim={depth_ffn_dim}` must be even.")
        self.depth_ffn_dim = depth_ffn_dim
        self.depth_head_dim = depth_head_dim or depth_hidden_size // depth_num_attention_heads
        self.depth_num_key_value_heads = (
            depth_num_key_value_heads if depth_num_key_value_heads is not None else depth_num_attention_heads
        )
        self.depth_sliding_window = depth_sliding_window

        audio_encoder_config = kwargs.pop("audio_encoder", {})
        audio_encoder_model_type = audio_encoder_config.pop("model_type", "mimi")

        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)

        if self.num_codebooks > self.audio_encoder.num_codebooks:
            raise ValueError(
                f"`num_codebooks={num_codebooks}` is greater than the maximum number of codebooks that the audio encoder can deal with ({self.audio_encoder.num_codebooks}). Please lower it."
            )

        self.audio_vocab_size = self.audio_encoder.codebook_size if audio_vocab_size is None else audio_vocab_size

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate

    @classmethod
    def from_audio_encoder_config(
        cls,
        audio_encoder_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MoshiConfig`] (or a derived class) from an audio encoder configuration.

        Returns:
            [`MoshiConfig`]: An instance of a configuration object
        """

        return cls(
            audio_encoder=audio_encoder_config.to_dict(),
            **kwargs,
        )
