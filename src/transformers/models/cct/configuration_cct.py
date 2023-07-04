# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" CCT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CCT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "rishabbala/cct_14_7x2_384": "https://huggingface.co/rishabbala/cct_14_7x2_384/blob/main/config.json",
}


class CctConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CctModel`]. It is used to instantiate a CCT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CCT
    [rishabbala/cct](https://huggingface.co/rishabbala/cct) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        img_size (`int`, *optional*, defaults to 384):
            The size of the input image
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`List[int]`, *optional*, defaults to [64, 384]):
            The number of output channels of each conv layer.
        conv_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size of convolutional layers in patch embedding.
        conv_stride (`int`, *optional*, defaults to 2):
            The stride size of convolutional layers in patch embedding.
        conv_padding (`int`, *optional*, defaults to 3):
            The padding size of convolutional layers in patch embedding.
        conv_bias (`bool`, *optional*, defaults to False):
            Whether the convolutional layers have bias
        pool_kernel_size (`int`, *optional*, defaults to 7):
            The kernel size of max pool layers in patch embedding.
        pool_stride (`int`, *optional*, defaults to 2):
            The stride size of max pool layers in patch embedding.
        pool_padding (`int`, *optional*, defaults to 3):
            The padding size of max pool layers in patch embedding.
        num_conv_layers (`int`, *optional*, defaults to 2):
            Number of convolutional embedding layers
        embed_dim (`int`, *optional*, defaults to 384):
            Dimension of each of the encoder blocks.
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratio (`float`, *optional*, defaults to 3.0):
            Ratio of the size of the hidden layer compared to the size of the input layer of the FFNs in the encoder
            blocks.
        attention_drop_rate (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        drop_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio following linear projections.
        drop_path_rate (`float`, *optional*, defaults to `0.0`):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        qkv_bias (`List[bool]`, *optional*, defaults to `[False, False, False]`):
            The bias bool for query, key and value in attentions
        qkv_projection_method (`List[string]`, *optional*, defaults to ["avg", "avg", "avg"]`):
            The projection method for query, key and value Default is depth-wise convolutions with batch norm. For
            Linear projection use "avg".
        num_transformer_layers(`int`, *optional*, defaults to 14):
            Number of transformer self-attention layers
        pos_emb_type (`str`, *optional*, defaults to 'learnable'):
            Type of positional embedding used. Alternative: 'sinusoidal'
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import CctConfig, CctModel

    >>> # Initializing a Cct msft/cct style configuration
    >>> configuration = CctConfig()

    >>> # Initializing a model (with random weights) from the msft/cct style configuration
    >>> model = CctModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "cct"

    def __init__(
        self,
        img_size=384,
        in_channels=3,
        out_channels=[64, 384],
        conv_kernel_size=7,
        conv_stride=2,
        conv_padding=3,
        conv_bias=False,
        pool_kernel_size=3,
        pool_stride=2,
        pool_padding=1,
        num_conv_layers=2,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=3,
        attention_drop_rate=0.1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=[False, False, False],
        qkv_projection_method=["avg", "avg", "avg"],
        num_transformer_layers=14,
        pos_emb_type="learnable",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = out_channels[-1]
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_bias = conv_bias
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.num_conv_layers = num_conv_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_drop_rate = attention_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qkv_projection_method = qkv_projection_method
        self.num_transformer_layers = num_transformer_layers
        self.pos_emb_type = pos_emb_type
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
