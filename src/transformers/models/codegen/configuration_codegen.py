# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
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
"""CodeGen model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Salesforce/codegen-2B-mono")
@strict
class CodeGenConfig(PreTrainedConfig):
    r"""
    n_ctx (`int`, *optional*, defaults to 2048):
        This attribute is used in `CodeGenModel.__init__` without any real effect.
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of dimensions in the embedding that Rotary Position Embedding is applied to.
    n_inner (`int`, *optional*):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd

    Example:

    ```python
    >>> from transformers import CodeGenConfig, CodeGenModel

    >>> # Initializing a CodeGen 6B configuration
    >>> configuration = CodeGenConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = CodeGenModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "codegen"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50400
    n_positions: int = 2048
    n_ctx: int = 2048
    n_embd: int = 4096
    n_layer: int = 28
    n_head: int = 16
    rotary_dim: int = 64
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    tie_word_embeddings: bool = False


__all__ = ["CodeGenConfig"]
