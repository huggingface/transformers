# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
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
"""nGPT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class NGPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NGPTModel`]. It is used to instantiate an nGPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the nGPT-8B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the nGPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`NGPTModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.014):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1000000):
            The base period of the RoPE embeddings.
        partial_rotary_factor (`float`, *optional*, defaults to 1.0): Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj and down_proj layers in the MLP layers.
        sqk_init_value (`float`, *optional*, defaults to 1.0):
            nGPT: The initial value for the self-attention sqk parameter.
        attn_alpha_init_value (`float`, *optional*, defaults to 0.05):
            nGPT: The initial value for the self-attention interpolation parameter.
        suv_init_value (`float`, *optional*, defaults to 1.0):
            nGPT: The initial value for the mlp suv parameter.
        suv_init_scaling (`float`, *optional*, defaults to 1.0):
            nGPT: The scaling factor for the mlp suv parameter.
        mlp_alpha_init_value (`float`, *optional*, defaults to 0.05):
            nGPT: The initial value for the mlp interpolation parameter.
        sz_init_value (`float`, *optional*, defaults to 1.0):
            nGPT: The initial value for the lm-head sz parameter.

    ```python
    >>> from transformers import NGPTModel, NGPTConfig

    >>> # Initializing a NGPT-8B style configuration
    >>> configuration = NGPTConfig()

    >>> # Initializing a model from the ngpt-8b style configuration
    >>> model = NGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ngpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `NGPTModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.sqk": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.suv": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=32,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,
        hidden_act="relu2",
        max_position_embeddings=8192,
        initializer_range=0.014,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1000000,
        partial_rotary_factor=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        sqk_init_value=1.0,
        attn_alpha_init_value=0.05,
        suv_init_value=1.0,
        suv_init_scaling=1.0,
        mlp_alpha_init_value=0.05,
        sz_init_value=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        # nGPT specific parameters
        self.sqk_init_value = sqk_init_value
        self.attn_alpha_init_value = attn_alpha_init_value
        self.suv_init_value = suv_init_value
        self.suv_init_scaling = suv_init_scaling
        self.mlp_alpha_init_value = mlp_alpha_init_value
        self.sz_init_value = sz_init_value

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["NGPTConfig"]
