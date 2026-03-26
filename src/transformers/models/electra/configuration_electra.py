# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""ELECTRA model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/electra-small-discriminator")
@strict
class ElectraConfig(PreTrainedConfig):
    r"""
    summary_type (`str`, *optional*, defaults to `"first"`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        Pass `"gelu"` for a gelu activation to the output, any other value will result in no activation.
    summary_last_dropout (`float`, *optional*, defaults to 0.0):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.
        The dropout ratio to be used after the projection and activation.

    Examples:

    ```python
    >>> from transformers import ElectraConfig, ElectraModel

    >>> # Initializing a ELECTRA electra-base-uncased style configuration
    >>> configuration = ElectraConfig()

    >>> # Initializing a model (with random weights) from the electra-base-uncased style configuration
    >>> model = ElectraModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "electra"

    vocab_size: int = 30522
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    summary_type: str = "first"
    summary_use_proj: bool = True
    summary_activation: str = "gelu"
    summary_last_dropout: float | int = 0.1
    pad_token_id: int | None = 0
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["ElectraConfig"]
