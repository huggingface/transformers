# Copyright 2021 The HuggingFace Inc. team.
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
"""OpenAI ImageGPT configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openai/imagegpt-small")
@strict
class ImageGPTConfig(PreTrainedConfig):
    r"""
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import ImageGPTConfig, ImageGPTModel

    >>> # Initializing a ImageGPT configuration
    >>> configuration = ImageGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ImageGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "imagegpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 512 + 1  # add one for start of sentence (sos) token
    n_positions: int = 32 * 32
    n_embd: int = 512
    n_layer: int = 24
    n_head: int = 8
    n_inner: int | None = None
    activation_function: str = "quick_gelu"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = False
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


__all__ = ["ImageGPTConfig"]
