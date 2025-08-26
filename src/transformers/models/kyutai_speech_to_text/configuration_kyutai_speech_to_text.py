# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.s

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class KyutaiSpeechToTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`KyutaiSpeechToTextForConditionalGeneration`].
    It is used to instantiate a Kyutai Speech-to-Text model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    2.6b-en model.

    e.g. [kyutai/stt-2.6b-en-trfs](https://huggingface.co/kyutai/stt-2.6b-en-trfs)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        codebook_vocab_size (`int`, *optional*, defaults to 2049):
            Vocabulary size of the codebook. Defines the number of different audio tokens that can be represented by each codebook.
        vocab_size (`int`, *optional*, defaults to 4001):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the layers and the pooler layer of the main decoder.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the main decoder block.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 750):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        rope_theta (`float`, *optional*, defaults to 100000.0):
            The base period of the RoPE embeddings.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        head_dim (`int`, *optional*, defaults to `hidden_size // num_attention_heads`):
            The attention head dimension.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        sliding_window (`int`, *optional*, defaults to 375):
            Sliding window attention window size. If not specified, will default to `3000`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_dim (`int`, *optional*, defaults to 11264):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the main decoder block. Must be even.
        rms_norm_eps (`float`, *optional*, defaults to 1e-08):
            The epsilon used by the rms normalization layers.
        num_codebooks (`int`, *optional*, defaults to 32):
            The number of audio codebooks for each audio channels.
        audio_bos_token_id (`int`, *optional*, defaults to 2048):
            Beginning of stream token id for codebook tokens.
        audio_pad_token_id (`int`, *optional*, defaults to 69569):
            Padding token id for codebook tokens.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        pad_token_id (`int`, *optional*, defaults to 3):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 48000):
            Beginning of stream token id for text tokens.
        codec_config (`PretrainedConfig`, *optional*):
            Configuration for the codec.
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:
                - **audio_encoder_config** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **depth__config** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the depth decoder config.


    Example:
    ```python
    >>> from transformers import KyutaiSpeechToTextConfig, KyutaiSpeechToTextForConditionalGeneration

    >>> # Initializing a KyutaiSpeechToTextConfig
    >>> configuration = KyutaiSpeechToTextConfig()

    >>> # Initializing a model
    >>> model = KyutaiSpeechToTextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kyutai_speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"codec_config": AutoConfig}

    def __init__(
        self,
        codebook_vocab_size=2049,
        vocab_size=4001,
        hidden_size=2048,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=None,
        max_position_embeddings=750,
        rope_theta=100000.0,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=375,
        attention_dropout=0.0,
        ffn_dim=11264,
        rms_norm_eps=1e-8,
        num_codebooks=32,
        audio_bos_token_id=2048,
        audio_pad_token_id=69569,
        tie_word_embeddings=False,
        pad_token_id=3,
        bos_token_id=48000,
        codec_config=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

        if codec_config is None:
            self.codec_config = AutoConfig.for_model("mimi")
            logger.info("codec_config is None, using default audio encoder config.")
        elif isinstance(codec_config, dict):
            self.codec_config = AutoConfig.for_model(**codec_config)
        elif isinstance(codec_config, PretrainedConfig):
            self.codec_config = codec_config

        self.num_codebooks = num_codebooks
        self.frame_size = self.codec_config.frame_size

        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.codebook_vocab_size = codebook_vocab_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        if ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={ffn_dim}` must be even.")
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sliding_window = sliding_window


__all__ = ["KyutaiSpeechToTextConfig"]
