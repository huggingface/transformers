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
"""ViT model configuration"""

from collections import OrderedDict
from collections.abc import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VGGTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

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
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, *optional*, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.
        pooler_output_size (`int`, *optional*):
           Dimensionality of the pooler layer. If None, defaults to `hidden_size`.
        pooler_act (`str`, *optional*, defaults to `"tanh"`):
           The activation function to be used by the pooler. Keys of ACT2FN are supported for Flax and
           Pytorch, and elements of https://www.tensorflow.org/api_docs/python/tf/keras/activations are
           supported for Tensorflow.

    Example:

    ```python
    >>> from transformers import ViTConfig, ViTModel

    >>> # Initializing a ViT vit-base-patch16-224 style configuration
    >>> configuration = ViTConfig()

    >>> # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
    >>> model = ViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vggt"

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        camera_trunk_depth=4,
        camera_pose_encoding_type="absT_quaR_FoV",
        dpt_features=256,
        dpt_out_channels=[256, 512, 1024, 1024],
        dpt_intermediate_layer_idx=[4, 11, 17, 23],
        track_features=128,
        track_iters=4,
        track_corr_levels=7,
        track_corr_radius=4,
        track_hidden_size=384,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # Core architecture parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        
        # Attention and embedding parameters
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.patch_embed = patch_embed
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.qk_norm = qk_norm
        self.rope_freq = rope_freq
        self.init_values = init_values
        
        # Task head enable flags
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track
        
        # Camera head parameters
        self.camera_trunk_depth = camera_trunk_depth
        self.camera_pose_encoding_type = camera_pose_encoding_type
        
        # DPT head parameters
        self.dpt_features = dpt_features
        self.dpt_out_channels = dpt_out_channels
        self.dpt_intermediate_layer_idx = dpt_intermediate_layer_idx
        
        # Track head parameters
        self.track_features = track_features
        self.track_iters = track_iters
        self.track_corr_levels = track_corr_levels
        self.track_corr_radius = track_corr_radius
        self.track_hidden_size = track_hidden_size


class ViTOnnxConfig(OnnxConfig):
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


__all__ = ["ViTConfig", "ViTOnnxConfig"]
