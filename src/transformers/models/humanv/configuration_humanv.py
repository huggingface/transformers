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

from __future__ import annotations

from typing import Any, Optional

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture.

    A pretrained checkpoint is available at: [nebularesearchtrain/nilla-story](https://huggingface.co/nebularesearchtrain/nilla-story)

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the MLP intermediate representations.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to `num_attention_heads`):
            Number of key/value heads.
        head_dim (`int`, *optional*, defaults to 32):
            Dimension of each attention head.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Initializer range.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            RMSNorm epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether input and output embeddings should be tied.
        rope_parameters (`dict`, *optional*):
            RoPE parameters. If not provided, `rope_theta` will be used.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP projections.
        attn_implementation (`str`, *optional*, defaults to `"eager"`):
            Attention implementation hint.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type specification.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding window attention.
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
        num_key_value_heads: Optional[int] = None,
        head_dim: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_parameters: Optional[dict[str, Any]] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        attn_implementation: str = "eager",
        layer_types: Optional[list[str]] = None,
        use_sliding_window: bool = False,
        **kwargs,
    ):
        rope_theta = kwargs.get("rope_theta", 10000.0)
        if rope_parameters is None:
            rope_parameters = {"rope_theta": rope_theta}
        elif "rope_theta" not in rope_parameters:
            rope_parameters = dict(rope_parameters)
            rope_parameters["rope_theta"] = rope_theta

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers

        layer_type_validation(layer_types, num_hidden_layers)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.attn_implementation = attn_implementation
        self.layer_types = layer_types
        self.use_sliding_window = use_sliding_window
        self.rope_parameters = rope_parameters

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["HumanVConfig"]
