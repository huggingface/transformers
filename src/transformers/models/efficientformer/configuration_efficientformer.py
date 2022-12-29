# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" EfficientFormer model configuration"""

from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "huggingface/efficientformer-l1": "https://huggingface.co/huggingface/efficientformer-l1/resolve/main/config.json",
}


class EfficientFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientFormerModel`]. It is used to instantiate
    an EfficientFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EfficientFormer
    [snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        layers (`List(int)`, *optional*, defaults to `[3, 2, 6, 4]`)
            Depth of each stage.
        embed_dims (`List(int)`, *optional*, defaults to `[48, 96, 224, 448]`)
            Dimensionality of each stage.
        downsamples (`List(bool)`, *optional*, defaults to `[True, True, True, True]`)
            Whether or not to downsample inputs between two stages.
        dim (`int`, *optional*, defaults to 448):
            Number of input channels
        key_dim (`int`, *optional*, defaults to 32):
            The size of the key in meta3D block.
        attn_ratio (`int`, *optional*, defaults to 4):
            Ratio of the size of the output dimension compared to input dimension of attention layers.
        resolution (`int`, *optional*, defaults to 5)
            Size of each patch
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the 3D MetaBlock.
        mlp_expansion_ratio (`int`, *optional*, defaults to 4):
            Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        pool_size (`int`, *optional*, defaults to 3):
            Kernel size of pooling layers.
        downsample_patch_size (`int`, *optional*, defaults to 3):
            The size of patches in downsampling layers.
        downsample_stride (`int`, *optional*, defaults to 2):
            The stride of convolution kernels in downsampling layers.
        downsample_pad (`int`, *optional*, defaults to 1):
            Padding in downsampling layers.
        drop_path_rate (`int`, *optional*, defaults to 0):
            Rate at which to increase dropout probability in DropPath.
        vit_num (`int`, *optional*, defaults to 1):
            The number of 3D MetaBlocks in the last stage.
        distillation (`bool`, *optional*, defaults to `True`):
            Whether to add a distillation head.
        use_layer_scale (`bool`, *optional*, defaults to `True`):
            Whether to scale outputs from token mixers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-5):
            Factor by which outputs from token mixers are scaled.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import EfficientFormerModel, EfficientFormerConfig

    >>> # Initializing default EfficientFormer configuration
    >>> config = EfficientFormerConfig()

    >>> # Initializing a model default configuration
    >>> model = EfficientFormerModel(config)

    >>> # Accessing the model configuration
    >>> config = model.config
    ```"""
    model_type = "efficientformer"

    def __init__(
        self,
        layers: List[int] = [3, 2, 6, 4],
        embed_dims: List[int] = [48, 96, 224, 448],
        downsamples: List[bool] = [True, True, True, True],
        dim: int = 448,
        key_dim: int = 32,
        attn_ratio: int = 4,
        resolution: int = 7,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 8,
        mlp_expansion_ratio: int = 4,
        hidden_dropout_prob: float = 0.0,
        patch_size: int = 16,
        num_channels: int = 3,
        pool_size: int = 3,
        downsample_patch_size: int = 3,
        downsample_stride: int = 2,
        downsample_pad: int = 1,
        drop_path_rate: float = 0.0,
        vit_num: int = 1,
        distillation: bool = True,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        num_labels: int = 1000,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.layers = layers
        self.embed_dims = embed_dims
        self.mlp_expansion_ratio = mlp_expansion_ratio
        self.downsamples = downsamples
        self.dim = dim
        self.key_dim = key_dim
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        self.pool_size = pool_size
        self.downsample_patch_size = downsample_patch_size
        self.downsample_stride = downsample_stride
        self.downsample_pad = downsample_pad
        self.drop_path_rate = drop_path_rate
        self.vit_num = vit_num
        self.distillation = distillation
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.num_labels = num_labels
