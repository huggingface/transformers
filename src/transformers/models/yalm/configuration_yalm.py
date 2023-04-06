# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" YaLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

YALM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class YalmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YalmModel`]. It is used to instantiate an YaLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YaLM-100B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        padded_vocab_size (`int`, *optional*, defaults to 128000):
            Vocabulary size of the YaLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`YalmModel`]
        embedding_size (`int`, *optional*, defaults to 2048):
            Token embeding dimension
        hidden_size (`int`, *optional*, defaults to 10240):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27308):
            Dimension of the MLP representations.
        num_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer in the Transformer encoder.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to True):
            Whether to scale attention output by inverse layer depth
        activation_type (`str` or `function`, *optional*, defaults to `"geglu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layernorm_epsilon (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import YalmModel, YalmConfig

    >>> # Initializing a YaLM yalm-100b style configuration
    >>> configuration = YalmConfig()

    >>> # Initializing a model from the yalm-100b style configuration
    >>> model = YalmModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "yalm"

    def __init__(
        self,
        padded_vocab_size=128000,
        embedding_size=2048,
        hidden_size=10240,
        intermediate_size=27308,
        num_layers=80,
        num_attention_heads=128,
        scale_attn_by_inverse_layer_idx=True,
        activation_type="silu",
        max_position_embeddings=1024,
        apply_residual_connection_post_layernorm=False,
        initializer_range=0.02,
        layernorm_epsilon=1e-5,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.padded_vocab_size = padded_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.activation_type = activation_type
        self.max_position_embeddings = max_position_embeddings
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
