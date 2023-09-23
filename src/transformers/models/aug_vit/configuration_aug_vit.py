# coding=utf-8
# Copyright 2022 Tensorgirl and The HuggingFace Inc. team. All rights reserved.
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
""" AugViT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging

from typing import List
logger = logging.get_logger(__name__)

AUG_VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tensorgirl/TFaugvit": "https://huggingface.co/tensorgirl/TFaugvit/resolve/main/config.json",
    # See all AugViT models at https://huggingface.co/models?filter=aug_vit
}


class AugViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~AugViTModel`].
    It is used to instantiate an AugViT model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the AugViT [tensorgirl/TFaugvit](https://huggingface.co/tensorgirl/TFaugvit) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the AugViT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~AugViTModel`] or
            [`~TFAugViTModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~AugViTModel`] or
            [`~TFAugViTModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import AugViTModel, AugViTConfig

    >>> # Initializing a AugViT tensorgirl/TFaugvit style configuration
    >>> configuration = AugViTConfig()

    >>> # Initializing a model from the tensorgirl/TFaugvit style configuration
    >>> model = AugViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "aug_vit"
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        num_classes: int = 1000,
        dim: int = 128,
        depth: int = 2,
        heads: int = 16,
        mlp_dim: int = 256,
        dropout: int = 0.1,
        emb_dropout: int = 0.1,
        num_channels:int=3,
        **kwargs,
    ):

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.num_channels=num_channels
        super().__init__(**kwargs)

    