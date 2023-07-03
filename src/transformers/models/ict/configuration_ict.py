# coding=utf-8
# Copyright 2023 Authors at City University of Hong Kong, Microsoft Cloud + AI,
# The HuggingFace Inc. team. All rights reserved.
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
""" ICT model configuration"""

import numpy as np

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ICT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sheonhan/ict-imagenet-256": "https://huggingface.co/sheonhan/ict-imagenet-256/resolve/main/config.json",
}


class IctConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`IctModel`]. It is used to instantiate an ICT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ICT model trained with the ImageNet dataset
    [sheonhan/ict-imagenet-256](https://huggingface.co/sheonhan/ict-imagenet-256).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 512):
            Vocabulary size of the ICT model. Defines the number of different tokens that can be represented by the
            `pixel_values` passed when calling [`IctTransformer`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 35):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_residual_blocks (`int`, *optional*, defaults to 8):
            The number of residual blocks in [`IctGuidedUpsampler`].
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function (can be one of the activation functions defined in src/transformers/activations.py).
            Defaults to "quick_gelu".
        embedding_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        residual_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities that will be used by default in the `generate` method
            of the model. Must be strictly positive.
        top_k (`int`, *optional*, defaults to 50):
            Number of highest probability vocabulary tokens to keep for top-k-filtering that will be used by default in
            the `generate` method of the model.
        gan_loss_function (`str`, *optional*, defaults to `"nsgan"`):
            GAN loss function for the guided upsampler. Choose one of `"nsgan"`, `"lsgan"`, `"hinge"`. Defaults to
            "nsgan".
        output_image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of the output image.
        clusters (`np.ndarray`, *optional*, defaults to `None`):
            Clusters used to quantize the image of shape `(n_clusters, 3)`. Provide the same `clusters` used for
            `IctImageProcessor`.

    Example:

    ```python
    >>> from transformers import IctConfig, IctModel

    >>> # Initializing a ICT ict-imagenet-256 style configuration
    >>> configuration = IctConfig()

    >>> # Initializing a model (with random weights) from the ict-imagenet-256 style configuration
    >>> model = IctModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "ict"

    def __init__(
        self,
        vocab_size=512,
        hidden_size=1024,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_residual_blocks=8,
        intermediate_size=4096,
        activation_function="gelu",
        embedding_dropout_prob=0.0,
        residual_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=32,
        num_channels=3,
        qkv_bias=True,
        temperature=1.0,
        top_k=50,
        gan_loss_function="nsgan",
        output_image_size=256,
        clusters=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_residual_blocks = num_residual_blocks
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.embedding_dropout_prob = embedding_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.temperature = temperature
        self.top_k = top_k
        self.gan_loss_function = gan_loss_function
        self.output_image_size = output_image_size
        self.clusters = np.array(clusters) if clusters is not None else None
