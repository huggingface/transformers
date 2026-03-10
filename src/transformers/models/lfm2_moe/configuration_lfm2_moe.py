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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-8B-A1B")
class Lfm2MoeConfig(PreTrainedConfig):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 3):
        L_cache dim in the conv layers.
    num_dense_layers (`int`, *optional*, defaults to 2):
        Number of dense Lfm2MoeMLP layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
    use_expert_bias (`bool`, *optional*, defaults to `True`):
        Whether to use the expert bias on the routing weights.

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
        initializer_range: float = 0.02,
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
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
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
        self.initializer_range = initializer_range

        self.rope_parameters = rope_parameters
        tie_word_embeddings = kwargs.get("tie_embedding", tie_word_embeddings)  # to fit original config keys
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["Lfm2MoeConfig"]
