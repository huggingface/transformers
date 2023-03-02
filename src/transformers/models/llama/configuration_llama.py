# coding=utf-8
# Copyright 2023 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
""" LLaMa model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/llama": "https://huggingface.co/facebook/llama/resolve/main/config.json",
}



class LLaMaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LLaMaModel`]. It is used to instantiate an
    LLaMa model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the LLaMa
    [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the LLaMa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LLaMaModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import LLaMaConfig, LLaMaModel

    >>> # Initializing a LLaMa gpt-neox-20b style configuration
    >>> configuration = LLaMaConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = LLaMaModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_position_embeddings=2048,
        layer_norm_eps=1e-5,
        use_cache=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
