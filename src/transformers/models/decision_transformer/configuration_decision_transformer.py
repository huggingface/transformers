# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Decision Transformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="")
@strict
class DecisionTransformerConfig(PreTrainedConfig):
    r"""
    state_dim (`int`, *optional*, defaults to 17):
        The state size for the RL environment
    act_dim (`int`, *optional*, defaults to 4):
        The size of the output action space
    max_ep_len (`int`, *optional*, defaults to 4096):
        The maximum length of an episode in the environment
    action_tanh (`bool`, *optional*, defaults to True):
        Whether to use a tanh activation on action prediction
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

    >>> # Initializing a DecisionTransformer configuration
    >>> configuration = DecisionTransformerConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DecisionTransformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "decision_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    state_dim: int = 17
    act_dim: int = 4
    hidden_size: int = 128
    max_ep_len: int = 4096
    action_tanh: bool = True
    vocab_size: int = 1
    n_positions: int = 1024
    n_layer: int = 3
    n_head: int = 1
    n_inner: int | None = None
    activation_function: str = "relu"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False


__all__ = ["DecisionTransformerConfig"]
