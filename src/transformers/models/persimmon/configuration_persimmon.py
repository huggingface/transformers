# coding=utf-8
# Copyright 2023 Adept AI and the HuggingFace Inc. team. All rights reserved.
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
"""Persimmon model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)


class PersimmonConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PersimmonModel`]. It is used to instantiate an
    Persimmon model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 262144):
            Vocabulary size of the Persimmon model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`PersimmonModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        qk_layernorm (`bool`, *optional*, default to `True`):
            Whether or not to normalize the Queries and Keys after projecting the hidden states
        hidden_dropout (`float`, *optional*, default to 0.0):
            The dropout ratio after applying the MLP to the hidden states.
        attention_dropout (`float`, *optional*, default to 0.0):
            The dropout ratio after computing the attention scores.
        partial_rotary_factor (`float`, *optional*, default to 0.5):
            Percentage of the query and keys which will have rotary embedding.

        Example:

    ```python
    >>> from transformers import PersimmonModel, PersimmonConfig

    >>> # Initializing a Persimmon persimmon-7b style configuration
    >>> configuration = PersimmonConfig()
    ```"""

    model_type = "persimmon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: Optional[int] = 262144,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 16384,
        num_hidden_layers: Optional[int] = 36,
        num_attention_heads: Optional[int] = 64,
        hidden_act: Optional[str] = "relu2",
        max_position_embeddings: Optional[int] = 16384,
        initializer_range: Optional[float] = 0.02,
        layer_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        qk_layernorm: Optional[bool] = True,
        hidden_dropout: Optional[float] = 0.0,
        attention_dropout: Optional[float] = 0.0,
        partial_rotary_factor: Optional[float] = 0.5,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.qk_layernorm = qk_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 25000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["PersimmonConfig"]
