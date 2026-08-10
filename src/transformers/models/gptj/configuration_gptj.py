# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
"""GPT-J model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="EleutherAI/gpt-j-6B")
@strict
class GPTJConfig(PreTrainedConfig):
    r"""
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of dimensions in the embedding that Rotary Position Embedding is applied to.

    Example:

    ```python
    >>> from transformers import GPTJModel, GPTJConfig

    >>> # Initializing a GPT-J 6B configuration
    >>> configuration = GPTJConfig()

    >>> # Initializing a model from the configuration
    >>> model = GPTJModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gptj"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50400
    n_positions: int = 2048
    n_embd: int = 4096
    n_layer: int = 28
    n_head: int = 16
    rotary_dim: int = 64
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float | int = 0.0
    embd_pdrop: float | int = 0.0
    attn_pdrop: float | int = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False


__all__ = ["GPTJConfig"]
