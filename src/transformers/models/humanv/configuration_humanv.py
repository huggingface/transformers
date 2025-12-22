# coding=utf-8
# Copyright 2025 The HumanV Team. All rights reserved.
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
"""HumanV model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    nilla-story [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the HumanV model.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads.
        head_dim (`int`, *optional*, defaults to 32):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`dict`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. Must include `rope_theta`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP projections.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention implementation. Supported: `"eager"`, `"sdpa"`.
        layer_types (`list[str]`, *optional*):
            Attention pattern for each layer. Supported: `"full_attention"`, `"sliding_attention"`.

        use_sparse_attention (`bool`, *optional*, defaults to `False`):
            If `True`, the default `layer_types` will be set to sliding attention for all layers.
        sparse_attention_impl (`str`, *optional*, defaults to `"local_global_block"`):
            Sparse implementation. Supported: `"local_global_block"`, `"masked"`.
        sparse_attention_window (`int`, *optional*, defaults to 256):
            Token window size for masked sparse fallback.
        sparse_block_size (`int`, *optional*, defaults to 64):
            Block size for block-sparse attention.
        sparse_local_num_blocks (`int`, *optional*, defaults to 4):
            Number of local previous blocks (including current) attended by each query block.
        sparse_global_num_blocks (`int`, *optional*, defaults to 2):
            Number of global landmark blocks attended by each query block (includes block 0).
        sparse_global_block_stride (`int`, *optional*, defaults to 4):
            Stride (in blocks) to choose landmark blocks from the past.

    ```python
    >>> from transformers import HumanVConfig, HumanVModel
    >>> configuration = HumanVConfig()
    >>> model = HumanVModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "humanv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sparse_attention: bool = False,
        sparse_attention_impl: str = "local_global_block",
        sparse_attention_window: int = 256,
        sparse_block_size: int = 64,
        sparse_local_num_blocks: int = 4,
        sparse_global_num_blocks: int = 2,
        sparse_global_block_stride: int = 4,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        rope_theta = kwargs.get("rope_theta", 10000.0)
        if rope_parameters is None:
            rope_parameters = {"rope_theta": rope_theta}
        elif "rope_theta" not in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = rope_theta

        if layer_types is None:
            if use_sparse_attention:
                layer_types = ["sliding_attention"] * num_hidden_layers
            else:
                layer_types = ["full_attention"] * num_hidden_layers

        layer_type_validation(layer_types, num_hidden_layers)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.attn_implementation = attn_implementation
        self.rope_parameters = rope_parameters
        self.layer_types = layer_types

        self.use_sparse_attention = use_sparse_attention
        self.sparse_attention_impl = sparse_attention_impl
        self.sparse_attention_window = sparse_attention_window
        self.sparse_block_size = sparse_block_size
        self.sparse_local_num_blocks = sparse_local_num_blocks
        self.sparse_global_num_blocks = sparse_global_num_blocks
        self.sparse_global_block_stride = sparse_global_block_stride

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HumanVConfig"]
