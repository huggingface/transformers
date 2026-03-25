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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-8B-A1B")
@strict
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
    ```
    """

    model_type = "lfm2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 65536
    hidden_size: int = 2048
    intermediate_size: int = 7168
    moe_intermediate_size: int = 1792
    num_hidden_layers: int = 32
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    use_cache: bool = True
    norm_eps: float = 0.00001
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    conv_bias: bool = False
    conv_L_cache: int = 3
    num_dense_layers: int = 2
    num_experts_per_tok: int = 4
    num_experts: int = 32
    use_expert_bias: bool = True
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2MoeConfig"]
