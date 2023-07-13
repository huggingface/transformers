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
""" TinyVit Transformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

TINYVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/tinyvit-21m-224": "https://huggingface.co/microsoft/tinyvit-21m-224/resolve/main/config.json",
}



class TinyVitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TinyVitModel`]. It is used to instantiate a TinyVit
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TinyVit
    [microsoft/tinyvit-21m-224](https://huggingface.co/microsoft/tinyvit-21m-224)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_sizes (`List[int]`, *optional*, defaults to [96, 192, 384, 768]):
            Dimensionality of the embeddings in each of the stages.
        depths (`List[int]`, *optional*, defaults to [2, 2, 6, 2]):
            Depth of each layer in the Transformer encoder.
        num_heads (`List[int]`, *optional*, defaults to [3, 6, 12, 24]):
            Number of attention heads in each layer of the Transformer encoder.
        window_sizes (`List[int]`, *optional*, defaults to [7, 7, 14, 7]):
            Size of the windows.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of MLP hidden dimensionality to embedding dimensionality.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        local_conv_size ():
            The kernel size of the depthwise convolution between attention and MLP.

    Example:

    ```python
    >>> from transformers import TinyVitConfig, TinyVitModel

    >>> # Initializing a TinyVit microsoft/tinyvit-21m-224 style configuration
    >>> configuration = TinyVitConfig()

    >>> # Initializing a model (with random weights) from the microsoft/tinyvit-21m-224 style configuration
    >>> model = TinyVitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "tinyvit"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_stages",
    }

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        hidden_sizes=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        local_conv_size=3,
        mbconv_expand_ratio=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.num_stages = len(depths)
        self.num_heads = num_heads
        self.window_sizes = window_sizes
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.local_conv_size = local_conv_size
        self.mbconv_expand_ratio = mbconv_expand_ratio