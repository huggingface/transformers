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
from ...modeling_rope_utils import RopeParameters


class Lfm2MoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm2MoeModel`]. It is used to instantiate a LFM2 Moe
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LFM2-8B-A1B model.
    e.g. [LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Lfm2Model`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 7168):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1792):
            Intermediate size of the routed expert.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
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
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the conv layers.
        conv_L_cache (`int`, *optional*, defaults to 3):
            L_cache dim in the conv layers.
        num_dense_layers (`int`, *optional*, defaults to 2):
            Number of dense Lfm2MoeMLP layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 32):
            Number of routed experts.
        use_expert_bias (`bool`, *optional*, defaults to `True`):
            Whether to use the expert bias on the routing weights.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts in MoE models.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        layer_types (`Optional`, *optional*):
            Type of each layers.

    ```python
    >>> from transformers import Lfm2MoeModel, Lfm2MoeConfig

    >>> # Initializing a LFM2 Moe model
    >>> configuration = Lfm2MoeConfig()

    >>> # Initializing a model from the LFM2-8B-A1B style configuration
    >>> model = Lfm2MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lfm2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 7168,
        moe_intermediate_size: int = 1792,
        num_hidden_layers: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_parameters: RopeParameters = None,
        max_position_embeddings: int = 128_000,
        use_cache: bool = True,
        norm_eps: float = 0.00001,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        conv_bias: bool = False,
        conv_L_cache: int = 3,
        num_dense_layers: int = 2,
        num_experts_per_tok: int = 4,
        num_experts: int = 32,
        use_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        layer_types: Optional[list[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.norm_eps = norm_eps

        # attn operator config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # custom operator config
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache

        # moe config
        self.num_dense_layers = num_dense_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.use_expert_bias = use_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.layer_types = layer_types

        self.rope_parameters = rope_parameters
        tie_word_embeddings = kwargs.get("tie_embedding", tie_word_embeddings)  # to fit original config keys
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Lfm2MoeConfig"]
