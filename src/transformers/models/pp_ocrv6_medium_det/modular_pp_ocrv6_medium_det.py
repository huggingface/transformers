# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...backbone_utils import (
    consolidate_backbone_kwargs_to_config,
)
from ...configuration_utils import PreTrainedConfig
from ...utils import (
    auto_docstring,
    logging,
)
from ..pp_ocrv5_mobile_det.modeling_pp_ocrv5_mobile_det import (
    PPOCRV5MobileDetHead,
)
from ..pp_ocrv5_server_det.configuration_pp_ocrv5_server_det import PPOCRV5ServerDetConfig
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import (
    PPOCRV5ServerDetForObjectDetection,
    PPOCRV5ServerDetIntraclassBlock,
    PPOCRV5ServerDetNeck,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_medium_det_safetensors")
@strict
class PPOCRV6MediumDetConfig(PPOCRV5ServerDetConfig):
    hidden_act = AttributeError()

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v4",
            **kwargs,
        )

        # For object detection pipeline compatibility: single class "text"
        self.id2label = {0: "text"} if self.id2label is None else self.id2label
        PreTrainedConfig.__post_init__(**kwargs)


class PPOCRV6MediumDetHead(PPOCRV5MobileDetHead):
    pass


class PPOCRV6MediumDetIntraclassBlock(PPOCRV5ServerDetIntraclassBlock):
    pass


class PPOCRV6MediumDetNeck(PPOCRV5ServerDetNeck):
    """
    The only difference from PPOCRV5ServerDetNeck is that
    feature_projection_convolution and pan_lateral_convolution require bias=True
    to load weights fused from depthwise reparameterization, pointwise convolution, and BatchNorm.
    """

    def __init__(self, config):
        nn.Module.__init__(self)
        self.interpolate_mode = config.interpolate_mode
        self.scale_factor_list = config.scale_factor_list
        self.num_backbone_stages = len(config.backbone_config.stage_out_channels)

        self.input_channel_adjustment_convolution = nn.ModuleList()
        self.input_feature_projection_convolution = nn.ModuleList()
        self.path_aggregation_head_convolution = nn.ModuleList()
        self.path_aggregation_lateral_convolution = nn.ModuleList()

        backbone_stage_output_channels = config.backbone_config.stage_out_channels

        for backbone_stage_index in range(len(backbone_stage_output_channels)):
            channel_adjustment_convolution = nn.Conv2d(
                in_channels=backbone_stage_output_channels[backbone_stage_index],
                out_channels=config.neck_out_channels,
                kernel_size=1,
                bias=False,
            )
            self.input_channel_adjustment_convolution.append(channel_adjustment_convolution)

            # [Key Change] bias=True
            feature_projection_convolution = nn.Conv2d(
                in_channels=config.neck_out_channels,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=True,
            )
            self.input_feature_projection_convolution.append(feature_projection_convolution)

            if backbone_stage_index > 0:
                pan_head_convolution = nn.Conv2d(
                    in_channels=config.neck_out_channels // 4,
                    out_channels=config.neck_out_channels // 4,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                )
                self.path_aggregation_head_convolution.append(pan_head_convolution)

            # [Key Change] bias=True
            pan_lateral_convolution = nn.Conv2d(
                in_channels=config.neck_out_channels // 4,
                out_channels=config.neck_out_channels // 4,
                kernel_size=9,
                padding=4,
                bias=True,
            )
            self.path_aggregation_lateral_convolution.append(pan_lateral_convolution)

        self.intraclass_blocks = nn.ModuleList()
        for _ in range(config.intraclass_block_number):
            self.intraclass_blocks.append(
                PPOCRV6MediumDetIntraclassBlock(
                    config.intraclass_block_config, config.neck_out_channels // 4, reduce_factor=config.reduce_factor
                )
            )


@auto_docstring(custom_intro="PPOCRV6MediumDet model for text detection tasks.")
class PPOCRV6MediumDetForObjectDetection(PPOCRV5ServerDetForObjectDetection):
    pass


__all__ = [
    "PPOCRV6MediumDetForObjectDetection",
    "PPOCRV6MediumDetConfig",
    "PPOCRV6MediumDetModel",  # noqa: F822
    "PPOCRV6MediumDetPreTrainedModel",  # noqa: F822
]
