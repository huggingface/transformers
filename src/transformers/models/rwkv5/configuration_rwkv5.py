# coding=utf-8
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
""" RWKV configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

RWKV5_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class Rwkv5Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Rwkv5Model`]. It is used to instantiate a RWKV5
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RWVK-4
    [RWKV/rwkv-5-world-1b5](https://huggingface.co/RWKV/rwkv-5-world-1b5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the RWKV5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Rwkv5Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the model.
        attention_hidden_size (`int`, *optional*):
            Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.
        num_attention_heads (`int`, *optional*, defaults to 64):
            The attention heads to use in rwkv5 self_attention module.
        head_size (`int`, *optional*, defaults to 64): head_size of rwkv5 self_attention module.
        intermediate_size (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. Will default to 4 times `hidden_size` if unset.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary. Defaults to 0 as RWKV5 uses the same tokenizer
            as GPTNeoX.
        eos_token_id (`int`, *optional*, defaults to 0):
            The id of the end of sentence token in the vocabulary. Defaults to 0 as RWKV5 uses the same tokenizer as
            GPTNeoX.
        rescale_every (`int`, *optional*, defaults to 6):
            At inference, the hidden states (and weights of the correponding output layers) are divided by 2 every
            `rescale_every` layer. If set to 0 or a negative number, no rescale is done.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the input token embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last state.


    Example:

    ```python
    >>> from transformers import Rwkv5Config, Rwkv5Model

    >>> # Initializing a Rwkv5 configuration
    >>> configuration = Rwkv5Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Rwkv5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwkv5"

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=768,
        num_hidden_layers=24,
        attention_hidden_size=None,
        num_attention_heads=64,
        head_size=64,
        intermediate_size=None,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=0,
        rescale_every=6,
        tie_word_embeddings=False,
        use_cache=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_hidden_size = attention_hidden_size if attention_hidden_size is not None else hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.intermediate_size = None
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rescale_every = rescale_every
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
        )
