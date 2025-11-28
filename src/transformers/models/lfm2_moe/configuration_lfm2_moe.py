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
            vocab_size (`int`, *optional*, defaults to 65536): <fill_docstring>
            hidden_size (`int`, *optional*, defaults to 2048): <fill_docstring>
            intermediate_size (`int`, *optional*, defaults to 7168): <fill_docstring>
            moe_intermediate_size (`int`, *optional*, defaults to 1792): <fill_docstring>
            num_hidden_layers (`int`, *optional*, defaults to 32): <fill_docstring>
            pad_token_id (`int`, *optional*, defaults to 0): <fill_docstring>
            bos_token_id (`int`, *optional*, defaults to 1): <fill_docstring>
            eos_token_id (`int`, *optional*, defaults to 2): <fill_docstring>
            tie_word_embeddings (`bool`, *optional*, defaults to `True`): <fill_docstring>
            rope_parameters (`RopeParameters`, *optional*): <fill_docstring>
            max_position_embeddings (`int`, *optional*, defaults to 128000): <fill_docstring>
            use_cache (`bool`, *optional*, defaults to `True`): <fill_docstring>
            norm_eps (`float`, *optional*, defaults to 1e-05): <fill_docstring>
            num_attention_heads (`int`, *optional*, defaults to 32): <fill_docstring>
            num_key_value_heads (`int`, *optional*, defaults to 8): <fill_docstring>
            conv_bias (`bool`, *optional*, defaults to `False`): <fill_docstring>
            conv_L_cache (`int`, *optional*, defaults to 3): <fill_docstring>
            num_dense_layers (`int`, *optional*, defaults to 2): <fill_docstring>
            num_experts_per_tok (`int`, *optional*, defaults to 4): <fill_docstring>
            num_experts (`int`, *optional*, defaults to 32): <fill_docstring>
            use_expert_bias (`bool`, *optional*, defaults to `True`): <fill_docstring>
            routed_scaling_factor (`float`, *optional*, defaults to 1.0): <fill_docstring>
            norm_topk_prob (`bool`, *optional*, defaults to `True`): <fill_docstring>
            layer_types (`Optional`, *optional*): <fill_docstring>
            hidden_act (`str`, *optional*, defaults to `"silu"`): <fill_docstring>
            initializer_range (`float`, *optional*, defaults to 0.02): <fill_docstring>

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
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
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
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

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
