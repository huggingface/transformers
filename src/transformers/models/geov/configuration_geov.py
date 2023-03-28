# coding=utf-8
# Copyright 2023 Better Planet Investments and labml.ai team. ALl rights reserved.
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
""" GeoV model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

GEOV_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "GeoV/GeoV-9b": "https://huggingface.co/GeoV/GeoV-9b/resolve/main/config.json",
}



class GeoVConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GeoVModel`]. It is used to instantiate an
    GeoV model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GeoV
    [GeoV/GeoV-9b](https://huggingface.co/GeoV/GeoV-9b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GeoV model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GeoVModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-4):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_extra_biases_ffn (`bool`, *optional*, defaults to `False`):
            Whether or not to have extra bias parameters in the final layer of FFN modules.
        Example:

    ```python
    >>> from transformers import GeoVConfig, GeoVModel

    >>> # Initializing a GeoV gpt-neox-20b style configuration
    >>> configuration = GeoVConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = GeoVModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""
    model_type = "geov"

    def __init__(
            self,
            vocab_size=65536,
            hidden_size=1024 * 5,
            num_hidden_layers=32,
            num_attention_heads=40,
            intermediate_size=1024 * 5 * 4,
            layer_norm_eps=1e-4,
            rotary_emb_base=10000,
            max_position_embeddings=2049,
            use_extra_biases_ffn=False,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         tie_word_embeddings=tie_word_embeddings,
                         **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.rotary_emb_base = rotary_emb_base
        self.use_cache = use_cache
        self.layer_norm_eps = layer_norm_eps
        self.use_extra_biases_ffn = use_extra_biases_ffn
