# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" CPMAnt model configuration"""

from typing import List, Optional, Tuple

import torch

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "cpm-ant-10b": "https://huggingface.co/cpm-ant-10b/resolve/main/config.json",
    # See all CPMAnt models at https://huggingface.co/models?filter=cpmant
}


class CPMAntConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~CPMAntModel`]. It is used to instantiate an
    CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CPMAnt
    [cpm-ant-10b](https://huggingface.co/cpm-ant-10b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the CPMAnt model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~CPMAntModel`] or [`~TFCPMAntModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~CPMAntModel`] or [`~TFCPMAntModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import CPMAntModel, CPMAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CPMAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CPMAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "cpmant"

    def __init__(
        self,
        vocab_size=30720,
        dim_model=4096,
        num_heads=32,
        dim_head=128,
        dim_ff=10240,
        num_layers=48,
        dropout_p=0.0,
        position_bias_num_buckets=512,
        position_bias_max_distance=2048,
        eps=1e-6,
        half: bool = False,
        prompt_types: int = 32,
        prompt_length: int = 32,
        segment_types: int = 32,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_types = prompt_types
        self.prompt_length = prompt_length
        self.segment_types = segment_types
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.dropout_p = dropout_p
        self.eps = eps
        self.torch_dtype = torch.float
        self.vocab_size = vocab_size
        self.mask_modules = mask_modules
