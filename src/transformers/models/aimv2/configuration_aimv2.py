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
"""AIMv2 model configuration"""

from collections import OrderedDict
from typing import Mapping
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any
import functools
import torch.nn as nn

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class AIMv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AIMv2Model`]. It is used to instantiate a AIM-v2
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
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in query, key, value.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in all linear layers.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to use a classification token.
        pos_embed_type (`str`, *optional*, defaults to `"absolute"`):
            Positional embedding type. Choose from 'absolute', 'sincos', or 'none'.
        use_rms_norm (`bool`, *optional*, defaults to `False`):
            Whether or not to use RMS norm.
        post_trunk_norm (`bool`, *optional*, defaults to `False`):
            Whether or not to use norm layer after the transformer blocks (layers).
        probe_layers (`int`, *optional*, defaults to 6):   # Change this to a list
            The layer ids to use for selecting features.
        reduce (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce features using mean.
        ffn_target_type (`str`, *optional*, defaults to `"swiglu"`):
            Type of feedforward network (FFN) to use.
        is_causal (`bool`, *optional*, defaults to `False`):
            Whether or not to use causal attention.
        norm_layer (`[torch.nn.Module]`, *optional*, defaults to `torch.nn.LayerNorm`):
            Normalization layer to use.
        num_queries (`int`, *optional*, defaults to 1):
            Number of query tokens for attention pooling.
        use_batch_norm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization in attention pooling.
        proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the projection layer of the attention pooling.
        average_pool (`bool`, *optional*, defaults to `True`):
            Whether to use average pooling in the attention pooling.
        num_labels (`int`, *optional*, defaults to 1000):
            The number of labels for classification tasks.
        **kwargs:
            Remaining keyword arguments are passed to the superclass.

    Example:

        ```python
        >>> from aim.v2.configuration_aimv2 import AIMv2Config

        >>> # Initializing a aimv2-large-patch14-224 style configuration
        >>> configuration = AIMv2Config()

        >>> # Accessing the model configuration
        >>> print(configuration)
        ```
    """

    model_type = "aimv2"

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        num_channels: int = 3,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        #mlp_ratio: float = 4.0,
        hidden_act: Union[str, Callable] = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        intermediate_size=2816,
        layer_norm_eps: float = 1e-5,
        qkv_bias: bool = False,
        use_bias: bool = False,
        use_cls_token: bool = False,
        pos_embed_type: str = "absolute",
        #use_rms_norm: bool = False,
        post_trunk_norm: bool = True,
        probe_layers: Union[int, Tuple[int, ...]] = [6], # Change this to a list
        reduce: bool = False,
        ffn_target_type: str = "swiglu",
        is_causal: bool = False,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.RMSNorm,
        num_queries: int = 1,
        use_batch_norm: bool = True,
        proj_bias: bool = False,
        average_pool: bool = True,
        num_labels: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        #self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size=intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias
        self.use_cls_token = use_cls_token
        self.pos_embed_type = pos_embed_type
        #self.use_rms_norm = use_rms_norm
        self.post_trunk_norm = post_trunk_norm
        self.probe_layers = probe_layers
        self.reduce = reduce
        self.ffn_target_type = ffn_target_type
        self.is_causal = is_causal
        self.norm_layer = norm_layer
        self.num_queries = num_queries
        self.use_batch_norm = use_batch_norm
        self.proj_bias = proj_bias
        self.average_pool = average_pool
        self.num_labels = num_labels


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
