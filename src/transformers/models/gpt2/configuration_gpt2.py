# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""OpenAI GPT-2 configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openai-community/gpt2")
@strict
class GPT2Config(PreTrainedConfig):
    r"""
    summary_type (`string`, *optional*, defaults to `"cls_index"`):
        Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in for the multiple choice head in
        [`GPT2DoubleHeadsModel`].
        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].
        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`].
        The dropout ratio to be used after the projection and activation.
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["GPT2Config"]
