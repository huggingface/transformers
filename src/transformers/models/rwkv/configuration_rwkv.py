# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""RWKV configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="RWKV/rwkv-4-169m-pile")
@strict
class RwkvConfig(PreTrainedConfig):
    r"""
    context_length (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model can be used with in a single forward (using it in RNN mode
        lets use any sequence length).
    attention_hidden_size (`int`, *optional*):
        Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.
    rescale_every (`int`, *optional*, defaults to 6):
        At inference, the hidden states (and weights of the corresponding output layers) are divided by 2 every
        `rescale_every` layer. If set to 0 or a negative number, no rescale is done.

    Example:

    ```python
    >>> from transformers import RwkvConfig, RwkvModel

    >>> # Initializing a Rwkv configuration
    >>> configuration = RwkvConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = RwkvModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}

    vocab_size: int = 50277
    context_length: int = 1024
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    attention_hidden_size: int | None = None
    intermediate_size: int | None = None
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    rescale_every: int = 6
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        self.attention_hidden_size = (
            self.attention_hidden_size if self.attention_hidden_size is not None else self.hidden_size
        )
        self.intermediate_size = self.intermediate_size if self.intermediate_size is not None else 4 * self.hidden_size

        super().__post_init__(**kwargs)


__all__ = ["RwkvConfig"]
