# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" Omnivore model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

OMNIVORE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "anugunj/omnivore_swinT": "https://huggingface.co/anugunj/omnivore_swinT/resolve/main/config.json",
}


class OmnivoreConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmnivoreModel`]. It is used to instantiate an
    Omnivore model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Omnivore
    [anugunj/omnivore](https://huggingface.co/anugunj/omnivore) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_image_labels (`int`, *optional*, defaults to 1000):
            The number of labels for image head.
        num_video_labels (`int`, *optional*, defaults to 400):
            The number of labels for video head.
        num_rgbd_labels (`int`, *optional*, defaults to 19):
            The number of labels for rgbd head.
        input_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int` | `List[int]`, *optional*, defaults to [4, 4, 4]):
            Patch size to use in the patch embedding layer.
        embed_dim (`int`, *optional*, defaults to 96):
            Number of linear projection output channels.
        depths (`List[int]`, *optional*, defaults to [2, 2, 6, 2],):
            Depth (number of layers) for each stage.
        num_heads (`List[int]`, *optional*, defaults to [3, 6, 12, 24]):
            Number of attention head of each stage.
        window_size (`int`, *optional*, defaults to 7)
            Size of the window used by swin transformer in the model,
        mlp_ratios (`float`, *optional*, defaults to 4.0):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the patch embeddings probabilities and projections in attention.
        drop_path_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.1]`):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        qkv_bias (`bool`, *optional*, defaults to True):
            The bias bool for query, key and value in attentions
        qk_scale (`bool`, *optional*, defaults to None):
            Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (`nn.Module`, *optional*, defaults to nn.LayerNorm):
            Normalization layer for the model
        patch_norm (`bool`, *optional*, defaults to False):
            If True, add normalization after patch embedding.
        frozen_stages (`int`, *optional*, defaults to -1):
            Stages to be frozen (stop grad and set eval mode) -1 means not freezing any parameters.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import OmnivoreModel, OmnivoreConfig

    >>> # Initializing a Omnivore omnivore-tiny-224 style configuration
    >>> configuration = OmnivoreConfig()
    >>> # Initializing a model from the omnivore-tiny-224 style configuration
    >>> model = OmnivoreModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "omnivore"

    def __init__(
        self,
        num_image_labels=1000,
        num_video_labels=400,
        num_rgbd_labels=19,
        input_channels=3,
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        frozen_stages=-1,
        depth_mode="summed_rgb_d_tokens",
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_image_labels = num_image_labels
        self.num_video_labels = num_video_labels
        self.num_rgbd_labels = num_rgbd_labels
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.depth_mode = depth_mode
        self.initializer_range = initializer_range
