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
        (...)

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
        rpn_bbox_coder_target_means=[0.0, 0.0, 0.0, 0.0],
        rpn_bbox_coder_target_stds=[1.0, 1.0, 1.0, 1.0],
        rpn_in_channels=256,
        rpn_feat_channels=256,
        rpn_loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        rpn_loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        rpn_test_cfg=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        # RoI heads (box + mask)
        rcnn_test_cfg=dict(
            score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5
        ),
        bbox_roi_extractor_roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
        bbox_roi_extractor_out_channels=256,
        bbox_roi_extractor_featmap_strides=[4, 8, 16, 32],
        bbox_head_in_channels=256,
        bbox_head_bbox_coder_target_means=[0.0, 0.0, 0.0, 0.0],
        bbox_head_bbox_coder_target_stds=[0.1, 0.1, 0.2, 0.2],
        mask_roi_extractor_roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
        mask_roi_extractor_out_channels=256,
        mask_roi_extractor_featmap_strides=[4, 8, 16, 32],
        # Training configurations: RPN
        rpn_train_cfg=dict(allowed_border=-1, pos_weight=-1, debug=False),
        rpn_assigner_pos_iou_thr=0.7,
        rpn_assigner_neg_iou_thr=0.3,
        rpn_assigner_min_pos_iou=0.3,
        rpn_assigner_match_low_quality=True,
        rpn_assigner_ignore_iof_thr=-1,
        rpn_sampler_num=256,
        rpn_sampler_pos_fraction=0.5,
        rpn_sampler_neg_pos_ub=-1,
        rpn_sampler_add_gt_as_proposals=False,
        rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        # Training configurations: RCNN
        rcnn_train_cfg=dict(mask_size=28, pos_weight=-1, debug=False),
        rcnn_assigner_pos_iou_thr=0.5,
        rcnn_assigner_neg_iou_thr=0.5,
        rcnn_assigner_min_pos_iou=0.5,
        rcnn_assigner_match_low_quality=True,
        rcnn_assigner_ignore_iof_thr=-1,
        rcnn_sampler_num=512,
        rcnn_sampler_pos_fraction=0.25,
        rcnn_sampler_neg_pos_ub=-1,
        rcnn_sampler_add_gt_as_proposals=True,
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
        self.rpn_bbox_coder_target_means = rpn_bbox_coder_target_means
        self.rpn_bbox_coder_target_stds = rpn_bbox_coder_target_stds
        self.rpn_in_channels = rpn_in_channels
        self.rpn_feat_channels = rpn_feat_channels
        self.rpn_loss_cls = rpn_loss_cls
        self.rpn_loss_bbox = rpn_loss_bbox
        self.rpn_test_cfg = rpn_test_cfg
        # RoI heads
        self.rcnn_test_cfg = rcnn_test_cfg
        self.bbox_roi_extractor_roi_layer = bbox_roi_extractor_roi_layer
        self.bbox_roi_extractor_out_channels = bbox_roi_extractor_out_channels
        self.bbox_roi_extractor_featmap_strides = bbox_roi_extractor_featmap_strides
        self.bbox_head_in_channels = bbox_head_in_channels
        self.bbox_head_bbox_coder_target_means = bbox_head_bbox_coder_target_means
        self.bbox_head_bbox_coder_target_stds = bbox_head_bbox_coder_target_stds
        self.mask_roi_extractor_roi_layer = mask_roi_extractor_roi_layer
        self.mask_roi_extractor_out_channels = mask_roi_extractor_out_channels
        self.mask_roi_extractor_featmap_strides = mask_roi_extractor_featmap_strides
        # Training configurations: RPN
        self.rpn_train_cfg = rpn_train_cfg
        self.rpn_assigner_pos_iou_thr = rpn_assigner_pos_iou_thr
        self.rpn_assigner_neg_iou_thr = rpn_assigner_neg_iou_thr
        self.rpn_assigner_min_pos_iou = rpn_assigner_min_pos_iou
        self.rpn_assigner_match_low_quality = rpn_assigner_match_low_quality
        self.rpn_assigner_ignore_iof_thr = rpn_assigner_ignore_iof_thr
        self.rpn_sampler_num = rpn_sampler_num
        self.rpn_sampler_pos_fraction = rpn_sampler_pos_fraction
        self.rpn_sampler_neg_pos_ub = rpn_sampler_neg_pos_ub
        self.rpn_sampler_add_gt_as_proposals = rpn_sampler_add_gt_as_proposals
        self.rpn_proposal = rpn_proposal
        # Training configurations: RCNN
        self.rcnn_train_cfg = rcnn_train_cfg
        self.rcnn_assigner_pos_iou_thr = rcnn_assigner_pos_iou_thr
        self.rcnn_assigner_neg_iou_thr = rcnn_assigner_neg_iou_thr
        self.rcnn_assigner_min_pos_iou = rcnn_assigner_min_pos_iou
        self.rcnn_assigner_match_low_quality = rcnn_assigner_match_low_quality
        self.rcnn_assigner_ignore_iof_thr = rcnn_assigner_ignore_iof_thr
        self.rcnn_sampler_num = rcnn_sampler_num
        self.rcnn_sampler_pos_fraction = rcnn_sampler_pos_fraction
        self.rcnn_sampler_neg_pos_ub = rcnn_sampler_neg_pos_ub
        self.rcnn_sampler_add_gt_as_proposals = rcnn_sampler_add_gt_as_proposals


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
