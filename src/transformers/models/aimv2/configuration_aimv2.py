# coding=utf-8
# Copyright 2024  The HuggingFace Inc. team. All rights reserved.
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
"""AIMv2 model configuration"""

from collections import OrderedDict
from typing import Mapping
import functools
import torch.nn as nn

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Aimv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Aimv2Model`]. It is used to instantiate a AIM-v2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the AIM-v2 [apple/aimv2-large-patch14-224](...)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to use a classification token.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether or not to use a mask token.
        use_pos_embed (`str`, *optional*, defaults to `"absolute"`):
            Positional embedding type. Choose from 'absolute', 'sincos', or 'none'.
        norm_layer (`[torch.nn.Module]`, *optional*, defaults to `torch.nn.LayerNorm`):
            Normalization layer to use.
        Example:

        ```python
        >>> from aim.v2.configuration_aimv2 import Aimv2Config
        >>> from aim.v2.modeling_aimv2 import Aimv2Model

        >>> # Initializing a aimv2-large-patch14-224 style configuration
        >>> configuration = Aimv2Config()

        >>> # Initializing a model (with random weights) from the aimv2-large-patch14-224 style configuration
        >>> model = Aimv2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "aimv2"

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cls_token: bool = False,
        use_mask_token: bool = False,
        use_pos_embed: str = "absolute",
        qkv_bias: bool = False,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cls_token = use_cls_token
        self.use_mask_token = use_mask_token
        self.use_pos_embed = use_pos_embed  # we will use "sincos" or "absolute"
        self.qkv_bias = qkv_bias
        # If norm_layer is provided, use it, otherwise, default to nn.LayerNorm with the specified eps
        self.norm_layer = (
            norm_layer
            if norm_layer is not None
            else functools.partial(nn.LayerNorm, eps=layer_norm_eps)
        )


class AIMv2OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4
