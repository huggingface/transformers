# coding=utf-8
# Copyright 2024 The OpenBMB team and the HuggingFace Inc. team. All rights reserved.
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
""" MiniCPM3 model configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)

MINICPM3_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class MiniCPM3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniCPM3Model`]. It is used to instantiate a MiniCPM3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MiniCPM3-4B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MiniCPM3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiniCPM3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        qk_nope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that don't use rotary position embeddings.
        qk_rope_head_dim (`int`, *optional*, defaults to 32):
            Dimension of the query/key heads that use rotary position embeddings.
        q_lora_rank (`int`, *optional*, defaults to 768):
            Rank of the LoRA matrices for query projections.
        kv_lora_rank (`int`, *optional*, defaults to 256):
            Rank of the LoRA matrices for key and value projections.
        v_head_dim (`int`, *optional*, defaults to None):
            Dimension of the value heads. If None, defaults to qk_nope_head_dim.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. MiniCPM 1 supports up to 2048 tokens,
            MiniCPM 2 up to 4096, CodeMiniCPM up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports multiple scaling
            strategies: linear, dynamic, and longrope. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. For longrope, additional parameters like "short_factor",
            "long_factor", and "original_max_position_embeddings" are required. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalMiniCPM/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        scale_emb (`float`, *optional*, defaults to 1):
            Scaling factor for embeddings.
        dim_model_base (`float`, *optional*, defaults to 1):
            Base dimension for model scaling.
        scale_depth (`float`, *optional*, defaults to 1):
            Depth scaling factor for residual connections.

    ```python
    >>> from transformers import MiniCPM3Model, MiniCPM3Config

    >>> # Initializing a MiniCPM3 style configuration
    >>> configuration = MiniCPM3Config()

    >>> # Initializing a model from the MiniCPM3 style configuration
    >>> model = MiniCPM3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "minicpm3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=768,
        kv_lora_rank=256,
        v_head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        scale_emb=1,
        dim_model_base=None,
        scale_depth=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = hidden_size // num_attention_heads if v_head_dim is None else v_head_dim
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = hidden_size if dim_model_base is None else dim_model_base
        self.scale_depth = math.sqrt(self.num_hidden_layers) if scale_depth is None else scale_depth

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None:
            if "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]
            if "factor" not in self.rope_scaling:
                original_max_position_embeddings = self.rope_scaling["original_max_position_embeddings"]
                factor = max_position_embeddings / original_max_position_embeddings
                self.rope_scaling["factor"] = factor

        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MiniCPM3Config"]
