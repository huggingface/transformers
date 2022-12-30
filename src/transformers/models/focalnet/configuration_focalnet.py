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
""" FocalNet model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/focalnet-tiny": "https://huggingface.co/microsoft/focalnet-tiny/resolve/main/config.json",
}


class FocalNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FocalNetModel`]. It is used to instantiate a FocalNet
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FocalNet
    [microsoft/focalnet-tiny](https://huggingface.co/microsoft/focalnet-tiny)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 96):
            Dimensionality of patch embedding.
        depths (`list(int)`, *optional*, defaults to [2, 2, 6, 2]):
            Depth of each layer in the Transformer encoder.
        num_heads (`list(int)`, *optional*, defaults to [3, 6, 12, 24]):
            Number of attention heads in each layer of the Transformer encoder.
        window_size (`int`, *optional*, defaults to 7):
            Size of windows.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether or not a learnable bias should be added to the queries, keys and values.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        patch_norm (`bool`, *optional*, defaults to True):
            Whether or not to add layer normalization after patch embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import FocalNetConfig, FocalNetModel

    >>> # Initializing a FocalNet microsoft/focalnet-tiny style configuration
    >>> configuration = FocalNetConfig()

    >>> # Initializing a model (with random weights) from the microsoft/focalnet-tiny style configuration
    >>> model = FocalNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "focalnet"

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        hidden_act="gelu",
        mlp_ratio=4.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        use_conv_embed=False,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_post_layernorm=False,
        use_post_layernorm_in_modulation=False,
        normalize_modulator=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.use_conv_embed = use_conv_embed
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_post_layernorm = use_post_layernorm
        self.use_post_layernorm_in_modulation = use_post_layernorm_in_modulation
        self.normalize_modulator = normalize_modulator
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
