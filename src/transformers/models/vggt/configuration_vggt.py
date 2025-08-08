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
"""VGGT model configuration"""

from collections import OrderedDict
from collections.abc import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class VGGTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VGGTModel`]. It is used to instantiate an VGGT
    model according to the specified arguments, defining the model architecture.
    """

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
        drop_path_rate=0.0,
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

        # Core architecture parameters - 与原始Aggregator保持一致
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
        self.drop_path_rate = drop_path_rate
        
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


class VGGTOnnxConfig(OnnxConfig):
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


# 添加必要的常量
VGGT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

__all__ = ["VGGTConfig", "VGGTOnnxConfig", "VGGT_PRETRAINED_CONFIG_ARCHIVE_MAP"]
