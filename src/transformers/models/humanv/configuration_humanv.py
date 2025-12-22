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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class HumanVConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HumanVModel`]. It is used to instantiate a
    HumanV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    nilla-story [humanvprojectceo/nilla-story](https://huggingface.co/humanvprojectceo/nilla-story).

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
            Number of key/value heads. If `num_key_value_heads=num_attention_heads`, the model uses MHA.
            If `num_key_value_heads=1`, the model uses MQA. Otherwise, it uses GQA.
        head_dim (`int`, *optional*, defaults to 32):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the MLP projections.

        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration. To enable YaRN, pass a dictionary like:
            `{"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 2048}`.
        qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply QK-Norm (L2-normalize Q and K vectors) in attention for improved stability.
        qk_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon used for QK-Norm.

    ```python
    >>> from transformers import HumanVConfig, HumanVForCausalLM
    >>> config = HumanVConfig(
    ...     qk_norm=True,
    ...     rope_scaling={
    ...         "rope_type": "yarn",
    ...         "factor": 8.0,
    ...         "beta_fast": 32.0,
    ...         "beta_slow": 1.0,
    ...         "original_max_position_embeddings": 2048,
    ...     },
    ... )
    >>> model = HumanVForCausalLM(config)
    ```"""

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
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HumanVConfig"]
