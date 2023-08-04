# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" ViT model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shauray/ViTPose": "https://huggingface.co/shauray/ViTPose/blob/main/config.json",
}


class ViTPoseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTPoseModel`]. It is used to instantiate an ViTPose
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViTPose
    [shauray/ViTPose](https://huggingface.co/shauray/ViTPose) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, `optional`, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import ViTPoseConfig, ViTPoseModel

    >>> # Initializing a ViTPose style configuration
    >>> configuration = ViTPoseConfig()

    >>> # Initializing a model (with random weights) from the vitpose style configuration
    >>> model = ViTPoseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vitpose"

    def __init__(
        self,
        img_size = (256, 192),
        patch_size=16,
        embed_dim = 384,
        depth = 12,
        num_attention_heads = 12,
        ratio = 1,
        use_checkpoint = False,
        mlp_ratio = 4,
        qkv_bias=True,
        drop_path_rate = .1,
        keypoint_in_channels = 382,
        keypoint_num_deconv_layer = 2,
        keypoint_num_deconv_filters = (256, 256),
        keypoint_num_deconv_kernels= (4,4),
        dropout_p = 0.0,
        num_output_channels = 17,
        initializer_range = 1,
        num_channels = 3,
        num_joints = 17,
        flip_test = False,
        udp = True,
        target_type = "GaussianHeatMap",
        kernel = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_attention_heads = num_attention_heads
        self.ratio = ratio
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.keypoint_in_channels = keypoint_in_channels
        self.keypoint_num_deconv_layer = keypoint_num_deconv_layer
        self.keypoint_num_deconv_filters = keypoint_num_deconv_filters
        self.keypoint_num_deconv_kernels = keypoint_num_deconv_kernels
        self.dropout_p = dropout_p
        self.num_output_channels = num_output_channels
        self.initializer_range = initializer_range
        self.num_channels = num_channels
        self.num_joints = num_joints
        self.flip_test = flip_test
        self.udp = udp
        self.target_type = target_type
        self.kernel = kernel


