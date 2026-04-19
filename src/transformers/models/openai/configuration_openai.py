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
"""OpenAI GPT configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openai-community/openai-gpt")
@strict
class OpenAIGPTConfig(PreTrainedConfig):
    r"""
    afn (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers
    summary_type (`str`, *optional*, defaults to `"cls_index"`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Has to be one of the following options:
            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing sequence summary, used in the models [`OpenAIGPTDoubleHeadsModel`] and
        [`OpenAIGPTDoubleHeadsModel`].
        The dropout ratio to be used after the projection and activation.

    Examples:

    ```python
    >>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

    >>> # Initializing a GPT configuration
    >>> configuration = OpenAIGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = OpenAIGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "openai-gpt"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 40478
    n_positions: int = 512
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    afn: str = "gelu"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    summary_type: str = "cls_index"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["OpenAIGPTConfig"]
