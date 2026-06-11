# Copyright 2022 The Metaseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""OPT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/opt-350m")
@strict
class OPTConfig(PreTrainedConfig):
    r"""
    do_layer_norm_before (`bool`, *optional*, defaults to `True`):
        Whether to perform layer normalization before the attention block.
    word_embed_proj_dim (`int`, *optional*):
        `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
        `hidden_size`.
    enable_bias (`bool`, *optional*, defaults to `True`):
        Whether or not if the linear layers in the attention blocks should use the bias term.
    layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
        Whether or not if the layer norms should have learnable parameters.

    Example:

    ```python
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    >>> configuration = OPTConfig()

    >>> # Initializing a model (with random weights) from the facebook/opt-large style configuration
    >>> model = OPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50272
    hidden_size: int = 768
    num_hidden_layers: int = 12
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    do_layer_norm_before: bool = True
    _remove_final_layer_norm: bool = False
    word_embed_proj_dim: int | None = None
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    num_attention_heads: int = 12
    activation_function: str = "relu"
    layerdrop: float | int = 0.0
    init_std: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 2
    enable_bias: bool = True
    layer_norm_elementwise_affine: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.word_embed_proj_dim = (
            self.word_embed_proj_dim if self.word_embed_proj_dim is not None else self.hidden_size
        )
        super().__post_init__(**kwargs)


__all__ = ["OPTConfig"]
