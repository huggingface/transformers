# coding=utf-8
# Copyright 2024 Convai Innovations and The HuggingFace Inc. team. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""ConvaiCausalLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CONVAICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-causal-lm": "https://huggingface.co/convaiinnovations/hindi-causal-lm/resolve/main/config.json",
}


class ConvaiCausalLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvaiCausalLMModel`]. It is used to instantiate a
    ConvaiCausalLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [convaiinnovations/hindi-causal-lm](https://huggingface.co/convaiinnovations/hindi-causal-lm) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the ConvaiCausalLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ConvaiCausalLMModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden states.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by averaging the weights of the heads in the group. For more details visit
            [this blog post](https://huggingface.co/blog/GQA). If not specified, will default to `num_attention_heads`.
        head_dim (`int`, *optional*):
            The attention head dimension. If not specified, will default to `hidden_size // num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder MLP layers.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers. Matches Llama's default for easier inheritance.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id. Matches the default from the training script.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id. Matches the default from the training script.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id. Matches the default from the training script.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight matrices of input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for RoPE embeddings. Currently supports `linear` and `dynamic` scaling. For dynamic scaling, pass {'type': 'dynamic', 'factor': scaling_factor}. For linear scaling, pass {'type': 'linear', 'factor': scaling_factor}. Unused if null.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether or not the model uses bias in attention projections. Matches Llama's default.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether or not the model uses bias in the MLP layers. Matches Llama's default.

        Example:
        ```python
        >>> from transformers import ConvaiCausalLMModel, ConvaiCausalLMConfig

        >>> # Initializing a ConvaiCausalLM convaiinnovations/hindi-causal-lm style configuration
        >>> configuration = ConvaiCausalLMConfig()

        >>> # Initializing a model from the convaiinnovations/hindi-causal-lm style configuration
        >>> model = ConvaiCausalLMModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "convaicausallm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=None,  # Will be calculated if None
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5,  # Using RMSNorm like Llama
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,  # Matches Llama default
        attention_dropout=0.0,
        mlp_bias=False,  # Matches Llama default
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # GQA setup
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # Head dim calculation
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim
        # Ensure hidden_size is consistent
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_attention_heads})."
                f" If you specified `head_dim`, make sure `head_dim * num_attention_heads == hidden_size`."
            )

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias  # Store mlp_bias

        # RoPE parameters
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()  # Validate RoPE scaling config if present

        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
