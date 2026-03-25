# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""Longformer configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/longformer-base-4096")
@strict
class LongformerConfig(PreTrainedConfig):
    r"""
    attention_window (`int` or `list[int]`, *optional*, defaults to 512):
        Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
        different window size for each layer, use a `list[int]` where `len(attention_window) == num_hidden_layers`.
    onnx_export (`bool`, *optional*, defaults to False):
        Whether to create a model so that it is ONNX exportable or not.

    Example:

    ```python
    >>> from transformers import LongformerConfig, LongformerModel

    >>> # Initializing a Longformer configuration
    >>> configuration = LongformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = LongformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "longformer"

    attention_window: list[int] | int = 512
    sep_token_id: int | None = 2
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
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
    onnx_export: bool = False
    tie_word_embeddings: bool = True


__all__ = ["LongformerConfig"]
