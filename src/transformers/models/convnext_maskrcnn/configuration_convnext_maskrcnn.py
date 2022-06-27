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
""" ConvNextMaskRCNN model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CONVNEXTMASKRCNN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/convnext-tiny-maskrcnn": (
        "https://huggingface.co/facebook/convnext-tiny-maskrcnn/resolve/main/config.json"
    ),
}


class ConvNextMaskRCNNConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ConvNextMaskRCNNModel`]. It is used to
    instantiate an ConvNextMaskRCNN model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the ConvNextMaskRCNN
    [facebook/convnext-tiny-maskrcnn](https://huggingface.co/facebook/convnext-tiny-maskrcnn) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_size (`int`, optional, defaults to 4):
            Patch size to use in the patch embedding layer.
        num_stages (`int`, optional, defaults to 4):
            The number of stages in the model.
        hidden_sizes (`List[int]`, *optional*, defaults to [96, 192, 384, 768]):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to [3, 3, 9, 3]):
            Depth (number of blocks) for each stage.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in each block. If string, `"gelu"`, `"relu"`,
            `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        layer_scale_init_value (`float`, *optional*, defaults to 1e-6):
            The initial value for the layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop rate for stochastic depth.
        backbone_out_indices (`List[int]`, *optional*, defaults to [0, 1, 2, 3]):
            Indices of the intermediate hidden states to use from backbone.
        fpn_out_channels (`int`, optional, defaults to 256):
            Number of output channels (feature dimension) of the output feature maps of the Feature Pyramid Network
            (FPN).
        fpn_num_outputs (`int`, optional, defaults to 5):
            Number of output feature maps of the Feature Pyramid Network (FPN).

    Example:
    ```python
    >>> from transformers import ConvNextMaskRCNNModel, ConvNextMaskRCNNConfig

    >>> # Initializing a ConvNextMaskRCNN convnext_maskrcnn-tiny-224 style configuration
    >>> configuration = ConvNextMaskRCNNConfig()
    >>> # Initializing a model from the convnext_maskrcnn-tiny-224 style configuration
    >>> model = ConvNextMaskRCNNModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "convnext_maskrcnn"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=4,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        layer_scale_init_value=1e-6,
        drop_path_rate=0.0,
        image_size=224,
        # FPN
        backbone_out_indices=[0, 1, 2, 3],
        fpn_out_channels=256,
        fpn_num_outputs=5,
        # Anchor generator
        anchor_generator_scales=[8],
        anchor_generator_ratios=[0.5, 1.0, 2.0],
        anchor_generator_strides=[4, 8, 16, 32, 64],
        # RPN
        rpn_in_channels=256,
        rpn_feat_channels=256,
        rpn_loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        rpn_loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        rpn_test_cfg=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [96, 192, 384, 768] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        # FPN
        self.backbone_out_indices = backbone_out_indices
        self.fpn_out_channels = fpn_out_channels
        self.fpn_num_outputs = fpn_num_outputs
        # Anchor generator
        self.anchor_generator_scales = anchor_generator_scales
        self.anchor_generator_ratios = anchor_generator_ratios
        self.anchor_generator_strides = anchor_generator_strides
        # RPN
        self.rpn_in_channels = rpn_in_channels
        self.rpn_feat_channels = rpn_feat_channels
        self.rpn_loss_cls = rpn_loss_cls
        self.rpn_loss_bbox = rpn_loss_bbox
        self.rpn_test_cfg = rpn_test_cfg


class ConvNextMaskRCNNOnnxConfig(OnnxConfig):

    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-5
