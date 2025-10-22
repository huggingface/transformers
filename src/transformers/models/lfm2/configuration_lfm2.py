# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params


class Lfm2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm2Model`]. It is used to instantiate a LFM2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LFM2-1.2B model.
    e.g. [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Lfm2Model`]
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with. Lfm2 1 supports up to 2048 tokens,
            Lfm2 2 up to 4096, CodeLfm2 up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the conv layers.
        conv_L_cache (`int`, *optional*, defaults to 3):
            L_cache dim in the conv layers.
        block_multiple_of (`int`, *optional*, defaults to 256):
            Multiple for the `intermediate_size`.
        block_ffn_dim_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier for the `intermediate_size`.
        block_auto_adjust_ff_dim (`bool`, *optional*, defaults to `True`):
            Whether to adjust the dim of the `intermediate_size`.
        full_attn_idxs (`Optional`, *optional*):
            Index of the layers which use attention.
        layer_types (`Optional`, *optional*):
            Type of each layers.

    ```python
    >>> from transformers import Lfm2Model, Lfm2Config

    >>> # Initializing a LFM2 model
    >>> configuration = Lfm2Config()

    >>> # Initializing a model from the LFM2-1.2B style configuration
    >>> model = Lfm2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: Optional[int] = 65536,
        hidden_size: Optional[int] = 2560,
        intermediate_size: Optional[int] = 12288,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 8,
        max_position_embeddings: Optional[int] = 128_000,
        initializer_range: Optional[float] = 0.02,
        norm_eps: Optional[float] = 0.00001,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        conv_bias: Optional[bool] = False,
        conv_L_cache: Optional[int] = 3,
        block_multiple_of: Optional[int] = 256,
        block_ffn_dim_multiplier: Optional[float] = 1.0,
        block_auto_adjust_ff_dim: Optional[bool] = True,
        full_attn_idxs: Optional[list[int]] = None,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range

        # attn operator config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # custom operator config
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache

        # MLP config
        self.intermediate_size = kwargs.get("block_ff_dim", intermediate_size)  # to fit original config keys
        self.block_multiple_of = block_multiple_of
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        self.layer_types = layer_types
        if self.layer_types is None:
            full_attn_idxs = full_attn_idxs if full_attn_idxs is not None else list(range(num_hidden_layers))
            self.layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(num_hidden_layers)]

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("theta", kwargs.get("rope_theta", 1000000.0))
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        tie_word_embeddings = kwargs.get("tie_embedding", tie_word_embeddings)  # to fit original config keys
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Lfm2Config"]
