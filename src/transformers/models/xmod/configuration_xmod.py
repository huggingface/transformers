# Copyright 2023 The Meta AI Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""X-MOD configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/xmod-base")
@strict
class XmodConfig(PreTrainedConfig):
    r"""
    pre_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization before each block.
    adapter_reduction_factor (`int` or `float`, *optional*, defaults to 2):
        The factor by which the dimensionality of the adapter is reduced relative to `hidden_size`.
    adapter_layer_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply a new layer normalization before the adapter modules (shared across all adapters).
    adapter_reuse_layer_norm (`bool`, *optional*, defaults to `True`):
        Whether to reuse the second layer normalization and apply it before the adapter modules as well.
    ln_before_adapter (`bool`, *optional*, defaults to `True`):
        Whether to apply the layer normalization before the residual connection around the adapter module.
    languages (`Iterable[str]`, *optional*, defaults to `["en_XX"]`):
        An iterable of language codes for which adapter modules should be initialized.
    default_language (`str`, *optional*):
        Language code of a default language. It will be assumed that the input is in this language if no language
        codes are explicitly passed to the forward method.

    Examples:

    ```python
    >>> from transformers import XmodConfig, XmodModel

    >>> # Initializing an X-MOD facebook/xmod-base style configuration
    >>> configuration = XmodConfig()

    >>> # Initializing a model (with random weights) from the facebook/xmod-base style configuration
    >>> model = XmodModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xmod"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    pre_norm: bool = False
    adapter_reduction_factor: int = 2
    adapter_layer_norm: bool = False
    adapter_reuse_layer_norm: bool = True
    ln_before_adapter: bool = True
    languages: list[str] | tuple[str, ...] = ("en_XX",)
    default_language: str | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XmodConfig"]
