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
"""DepthPro model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging
from transformers.utils.backbone_utils import get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class DepthProConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthProModel`]. It is used to instantiate a
    DepthPro model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DepthPro
    [apple/DepthPro](https://huggingface.co/apple/DepthPro) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of the hidden size of the MLPs relative to the `hidden_size`.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        layerscale_value (`float`, *optional*, defaults to 1.0):
           Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        apply_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to the feature maps in case the model is used as backbone.
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
            case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
            seq_len, hidden_size)`.

    Example:

    ```python
    >>> from transformers import DepthProConfig, DepthProModel

    >>> # Initializing a DepthPro apple/DepthPro style configuration
    >>> configuration = DepthProConfig()

    >>> # Initializing a model (with random weights) from the apple/DepthPro style configuration
    >>> model = DepthProModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "depth_pro"

    def __init__(
        self,
        hidden_size=1024,
        decoder_hidden_size=256,
        num_hidden_layers=24,
        num_attention_heads=16,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=1536,
        patch_size=384,
        num_channels=3,
        patch_embeddings_size=16,
        qkv_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        patch_encoder_hook_ids = [5, 11],
        intermediate_feature_dims = [256, 256],
        high_res_feature_dims = 512,
        med_res_feature_dims = 1024,
        low_res_feature_dims = 1024,
        image_feature_dims = 1024,
        global_feature_dims = 1024,
        use_batch_norm_in_decoder=False,
        use_fov_model=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_embeddings_size = patch_embeddings_size
        self.qkv_bias = qkv_bias
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.patch_encoder_hook_ids = patch_encoder_hook_ids
        self.use_batch_norm_in_decoder = use_batch_norm_in_decoder
        self.use_fov_model = use_fov_model
        self.intermediate_feature_dims = intermediate_feature_dims
        self.high_res_feature_dims = high_res_feature_dims
        self.med_res_feature_dims = med_res_feature_dims
        self.low_res_feature_dims = low_res_feature_dims
        self.image_feature_dims = image_feature_dims
        self.global_feature_dims = global_feature_dims
