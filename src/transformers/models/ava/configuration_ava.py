# -*- coding: utf-8 -*-
# Copyright 2025 Nika Kudukhashvili <nikakuduxashvili0@gmail.com>. All rights reserved.
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

from ...configuration_utils import PretrainedConfig

class AvaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`AvaModel`]. It is used to instantiate an AVA model
    according to the specified arguments, defining the model architecture. 

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the AVA model. Defines the number of different tokens that can be represented.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        kv_heads (`int`, *optional*):
            Number of key/value heads (for Grouped Query Attention). Defaults to num_attention_heads.
        head_dim (`int`, *optional*):
            The dimension of each attention head. Defaults to hidden_size // num_attention_heads.
    """

    model_type = "ava"
    PREDEFINED_MODELS = {
        # Tiny models (Edge devices, IoT, offline agents, chatbots)
        '100m': {
            'hidden_size': 768,
            'intermediate_size': 3072,
            'num_hidden_layers': 6,
            'num_attention_heads': 12,
            'max_position_embeddings': 2048,
            'head_dim': 64,
            'kv_heads': 4
        },
        '500m': {
            'hidden_size': 1024,
            'intermediate_size': 4096,
            'num_hidden_layers': 8,
            'num_attention_heads': 16,
            'max_position_embeddings': 2048,
            'head_dim': 64,
            'kv_heads': 4
        },
        # Small models (Mobile apps, personal assistants, summarization)
        '1b': {
            'hidden_size': 1280,
            'intermediate_size': 5120,
            'num_hidden_layers': 12,
            'num_attention_heads': 16,
            'max_position_embeddings': 4096,
            'head_dim': 80,
            'kv_heads': 8
        },
        '3b': {
            'hidden_size': 1600,
            'intermediate_size': 6400,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'max_position_embeddings': 4096,
            'head_dim': 100,
            'kv_heads': 8
        },
        # Medium models (Coding, reasoning, multi-turn chat, translation)
        '7b': {
            'hidden_size': 4096,
            'intermediate_size': 11008,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'max_position_embeddings': 8192,
            'head_dim': 128,
            'kv_heads': 8
        },
        '13b': {
            'hidden_size': 5120,
            'intermediate_size': 13824,
            'num_hidden_layers': 40,
            'num_attention_heads': 40,
            'max_position_embeddings': 8192,
            'head_dim': 128,
            'kv_heads': 8
        },
        # Large models (Research, enterprise-level applications)
        '30b': {
            'hidden_size': 6656,
            'intermediate_size': 17920,
            'num_hidden_layers': 60,
            'num_attention_heads': 52,
            'max_position_embeddings': 8192,
            'head_dim': 128,
            'kv_heads': 8
        },
        '65b': {
            'hidden_size': 8192,
            'intermediate_size': 22016,
            'num_hidden_layers': 80,
            'num_attention_heads': 64,
            'max_position_embeddings': 8192,
            'head_dim': 128,
            'kv_heads': 8
        },
        # Massive models (AGI research, cutting-edge LLMs)
        '100b': {
            'hidden_size': 12288,
            'intermediate_size': 33024,
            'num_hidden_layers': 96,
            'num_attention_heads': 96,
            'max_position_embeddings': 16384,
            'head_dim': 128,
            'kv_heads': 8
        }
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_dropout=0.0,
        kv_heads=None,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.kv_heads = kv_heads if kv_heads is not None else num_attention_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        # Validate parameters
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration parameters"""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")

        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")

        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads, got {self.hidden_size} and {self.num_attention_heads}"
            )

        if self.kv_heads <= 0:
            raise ValueError(f"kv_heads must be positive, got {self.kv_heads}")

        if self.num_attention_heads % self.kv_heads != 0:
            raise ValueError(
                f"num_attention_heads must be divisible by kv_heads, got {self.num_attention_heads} and {self.kv_heads}"
            )

    @classmethod
    def from_predefined(cls, model_size="7b", **kwargs):
        """
        Instantiate a config from a predefined model architecture.

        Args:
            model_size (`str`): 
                One of the predefined model sizes (e.g., '100m', '500m', '1b', '3b', '7b', '13b', '30b', '65b', '100b')
            **kwargs:
                Additional arguments to override the predefined config

        Returns:
            AvaConfig: The configuration object
        """
        if model_size not in cls.PREDEFINED_MODELS:
            raise ValueError(
                f"Unknown model size '{model_size}'. Available sizes: {list(cls.PREDEFINED_MODELS.keys())}"
            )

        config_dict = cls.PREDEFINED_MODELS[model_size].copy()
        config_dict.update(kwargs)

        return cls(**config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a config from a pretrained model name or path.

        This method handles both:
        - Actual pretrained model paths (files/directories)
        - Predefined model shortcuts (e.g., "ava/7b")
        """
        if isinstance(pretrained_model_name_or_path, str):
            if pretrained_model_name_or_path.startswith("ava/"):
                model_size = pretrained_model_name_or_path.split("/")[-1]
                return cls.from_predefined(model_size, **kwargs)
 
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)