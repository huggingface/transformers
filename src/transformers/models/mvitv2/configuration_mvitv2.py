# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""MViTV2 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MViTV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`MViTV2Model`]. It is used to instantiate a
    MViTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MViTV2
    [KamilaMile/mvitv2-base](https://huggingface.co/KamilaMile/mvitv2-base)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        depths (`tuple[int, int, int, int]`, *optional*, defaults to (2, 3, 16, 3))
            The depth (number of blocks) of each stage.
        in_channels (`int`, *optional*, defaults to 3)
            The number of input channels.
        hidden_size (`int`, *optional*, defaults to 96):
            Dimensionality of the first stage.
        num_heads (`int`, *optional*, defaults to 1):
            Number of heads in the first stage.
        image_size (`tuple[int, int]`, *optional*, defaults to (224, 224)):
            The size (resolution) of each image.
        patch_kernel_size (`tuple[int, int]`, *optional*, defaults to (7, 7)):
            The size of the kernel used for patching.
        patch_stride_size (`tuple[int, int]`, *optional*, defaults to (4, 4)):
            The stride used during patching.
        patch_padding_size (`tuple[int, int]`, *optional*, defaults to (3, 3)):
            The padding used during patching.
        use_cls_token (`bool`, *optional*, defaults to False):
            Whether or not a cls token should be used.
        use_absolute_positional_embeddings (`bool`, *optional*, defaults to False):
            Whether or not absolute positional embeddings should be used.
        attention_pool_first (`bool`, *optional*, defaults to False):
            If set to True, the model will reduce the feature size first (pooling) before expanding the feature dimension to twice its size (2n).
            If False, the model will project (expand) the features first before pooling.
        expand_feature_dimension_in_attention (`bool`, *optional*, defaults to True):
            If set to True, the model will expand the feature dimension in the attention mechanism in the first block of a given stage.
            If False, the model will expand the feature dimension in the feed-forward network in the last block of a given stage.
        mode (`str`, *optional*, defaults to 'conv'):
            Defines how pooling is done. Takes values 'conv', 'conv_unshared', 'max' and 'avg'.
        kernel_qkv (`tuple[int, int]`, *optional*, defaults to (3, 3)):
            The size of the kernel used for pooling.
        stride_q (`tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]`, *optional*, defaults to ((1, 1), (2, 2), (2, 2), (2, 2))):
            The stride used for pooling queries across different stages.
        stride_kv_adaptive (`tuple[int, int]`, *optional*, defaults to (4, 4)):
            The initial stride used for pooling keys and values. Used to calculate stride_kv.
        stride_kv (`tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]`, *optional*, defaults to None):
            The stride used for pooling keys and values across different stages.
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether or not bias should be used when projecting inputs into qkv.
        residual_pooling (`bool`, *optional*, defaults to True):
            Whether or not the input query should be added to the attention output.
        relative_positional_embeddings_type (`str`, *optional*, defaults to 'spatial'):
            Defines the strategy for relative positional embeddings. Takes values 'spatial' and '' (no relative embeddings).
        mlp_ratio (`int`, *optional*, defaults to 4):
            Defines how much the feature dimention is expanded in the Feed Forward network.
        hidden_activation_function (`str`, *optional*, defaults to 'gelu'):
            Defines the activation function used in the Feed Forward network.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout ratio for stochastic depth.
        drop_rate (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the classifier head.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import MViTV2Config, MViTV2Model

    >>> # Initializing a MViTV2 mvitv2-base style configuration
    >>> configuration = MViTV2Config()

    >>> # Initializing a model from the configuration
    >>> model = MViTV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mvitv2"

    def __init__(
        self,
        depths=(2, 3, 16, 3),
        in_channels=3,
        hidden_size=96,
        num_heads=1,
        image_size=(224, 224),
        patch_kernel_size=(7, 7),
        patch_stride_size=(4, 4),
        patch_padding_size=(3, 3),
        use_cls_token=False,
        use_absolute_positional_embeddings=False,
        attention_pool_first=False,
        expand_feature_dimension_in_attention=True,
        mode="conv",
        kernel_qkv=(3, 3),
        stride_q=((1, 1), (2, 2), (2, 2), (2, 2)),
        stride_kv_adaptive=(4, 4),
        stride_kv=None,
        qkv_bias=True,
        residual_pooling=True,
        relative_positional_embeddings_type="spatial",
        mlp_ratio=4,
        hidden_activation_function="gelu",
        drop_path_rate=0.1,
        drop_rate=0.1,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depths = depths
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride_size = patch_stride_size
        self.patch_padding_size = patch_padding_size
        self.use_cls_token = use_cls_token
        self.use_absolute_positional_embeddings = use_absolute_positional_embeddings

        self.attention_pool_first = attention_pool_first
        self.expand_feature_dimension_in_attention = expand_feature_dimension_in_attention
        self.mode = mode
        self.kernel_qkv = kernel_qkv
        self.stride_q = stride_q
        self.stride_kv_adaptive = stride_kv_adaptive
        self.stride_kv = stride_kv
        self.qkv_bias = qkv_bias
        self.residual_pooling = residual_pooling
        self.relative_positional_embeddings_type = relative_positional_embeddings_type

        self.mlp_ratio = mlp_ratio
        self.hidden_activation_function = hidden_activation_function

        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon

    @property
    def classifier_hidden_size(self):
        return self.hidden_size * (2 ** (len(self.depths) - 1))


__all__ = ["MViTV2Config"]
