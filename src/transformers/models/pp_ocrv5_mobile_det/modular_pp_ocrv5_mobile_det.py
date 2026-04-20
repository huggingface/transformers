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


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    logging,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..pp_ocrv5_server_det.modeling_pp_ocrv5_server_det import (
    PPOCRV5ServerDetForObjectDetection,
    PPOCRV5ServerDetPreTrainedModel,
    PPOCRV5ServerDetSegmentationHead,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv5_mobile_det_safetensors")
@strict
class PPOCRV5MobileDetConfig(PreTrainedConfig):
    r"""
    reduction (`int`, *optional*, defaults to 4):
        The reduction factor for feature channel dimensions, used to reduce the number of model parameters and
        computational complexity while maintaining feature representability.
    neck_out_channels (`int`, *optional*, defaults to 96):
        The number of output channels from the neck network, which is responsible for feature fusion and
        refinement before passing features to the head network.
    interpolate_mode (`str`, *optional*, defaults to `"nearest"`):
        The interpolation mode used for upsampling or downsampling feature maps in the neck network. Supported
        modes include `"nearest"` (nearest neighbor interpolation) and `"bilinear"`.
    kernel_list (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
        The list of kernel sizes for convolutional layers in the head network, used for multi-scale feature
        extraction to detect text regions of different sizes.
    layer_list_out_channels (`List[int]`, *optional*, defaults to `[12, 18, 42, 360]`):
        The list of output channels for each backbone stage, used to configure the input channels of the RSE layers
        in the neck network for multi-scale feature fusion.
    """

    model_type = "pp_ocrv5_mobile_det"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    reduction: int = 4
    neck_out_channels: int = 96
    interpolate_mode: str = "nearest"
    kernel_list: list[int] | tuple[int, ...] = (3, 2, 2)
    layer_list_out_channels: list[int] | tuple[int, ...] = (12, 18, 42, 360)

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet_v3",
            default_config_kwargs={
                "scale": 0.75,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


@auto_docstring
class PPOCRV5MobileDetPreTrainedModel(PPOCRV5ServerDetPreTrainedModel):
    pass


class PPOCRV5MobileDetSqueezeExcitationModule(nn.Module):
    """
    Simplified Squeeze-and-Excitation (SE) Module for the neck network.
    Applies channel-wise recalibration with a clamped activation to stabilize training.
    """

    def __init__(self, in_channels, reduction, activation="relu"):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act_fn = ACT2FN[activation]

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.avg_pool(hidden_states)
        hidden_states = self.conv2(self.act_fn(self.conv1(hidden_states)))
        hidden_states = torch.clamp(0.2 * hidden_states + 0.5, min=0.0, max=1.0)
        return residual * hidden_states


class PPOCRV5MobileDetResidualSqueezeExcitationLayer(nn.Module):
    """
    Residual Squeeze-and-Excitation (RSE) Layer for the neck network.
    Combines a 1x1/3x3 convolution with an SE Module.
    """

    def __init__(self, in_channels, out_channels, kernel_size, reduction):
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.squeeze_excitation_block = PPOCRV5MobileDetSqueezeExcitationModule(out_channels, reduction)

    def forward(self, hidden_states):
        hidden_states = self.in_conv(hidden_states)
        hidden_states = hidden_states + self.squeeze_excitation_block(hidden_states)

        return hidden_states


class PPOCRV5MobileDetNeck(nn.Module):
    """
    Neck network for PPOCRV5 Mobile Det, responsible for multi-scale feature fusion.
    Uses Residual Squeeze-and-Excitation layers to process backbone features and upsampling to fuse features at the same spatial scale,
    then concatenates the fused features for input to the head network.
    """

    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__()
        self.interpolate_mode = config.interpolate_mode

        self.insert_conv = nn.ModuleList()
        self.input_conv = nn.ModuleList()
        for i in range(len(config.layer_list_out_channels)):
            self.insert_conv.append(
                PPOCRV5MobileDetResidualSqueezeExcitationLayer(
                    config.layer_list_out_channels[i],
                    config.neck_out_channels,
                    1,
                    config.reduction,
                )
            )
            self.input_conv.append(
                PPOCRV5MobileDetResidualSqueezeExcitationLayer(
                    config.neck_out_channels, config.neck_out_channels // 4, 3, config.reduction
                )
            )

    def forward(self, feature_maps):
        fused = []
        for conv, feature in zip(self.insert_conv, feature_maps):  # [p2, p3, p4, p5]
            hidden_states = conv(feature)
            fused.append(hidden_states)

        for i in range(2, -1, -1):  # p4 -> p3-> p2
            fused[i] = fused[i] + F.interpolate(fused[i + 1], scale_factor=2, mode=self.interpolate_mode)

        features = []
        for conv, feat in zip(self.input_conv, [fused[0], fused[1], fused[2], fused[3]]):
            features.append(conv(feat))

        processed = []
        upsample_scales = [1, 2, 4, 8]  # p2, p3, p4, p5
        for feat, scale in zip(features, upsample_scales):
            if scale != 1:
                hidden_states = F.interpolate(feat, scale_factor=scale, mode=self.interpolate_mode)
            else:
                hidden_states = feat
            processed.append(hidden_states)

        fused_feature_map = torch.cat(processed[::-1], dim=1)  # [p5, p4, p3, p2]
        return fused_feature_map


class PPOCRV5MobileDetHead(PPOCRV5ServerDetSegmentationHead):
    # MobileDet does not return residual features
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.conv_down(hidden_states)
        hidden_states = self.conv_up(hidden_states)
        hidden_states = self.conv_final(hidden_states)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Core PP-OCRv5_mobile_det, consisting of Backbone, Neck, and Head networks.
    Generates binary text segmentation maps for text detection tasks.
    """
)
class PPOCRV5MobileDetModel(PPOCRV5MobileDetPreTrainedModel):
    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__(config)

        self.config = config
        self.backbone = load_backbone(config)
        out_channels = [self.backbone.num_features[i] for i in self.backbone.out_indices]
        self.layer = nn.ModuleList()
        for idx, out_channel in enumerate(out_channels):
            self.layer.append(nn.Conv2d(out_channel, config.layer_list_out_channels[idx], 1, 1, 0))

        self.neck = PPOCRV5MobileDetNeck(config)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.backbone(hidden_states, **kwargs)
        feature_maps = outputs.feature_maps
        processed_features = []
        for i in range(len(feature_maps)):
            processed_features.append(self.layer[i](feature_maps[i]))
        hidden_states = self.neck(processed_features)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=outputs.hidden_states)


@auto_docstring(
    custom_intro="""
    PPOCRV5 Mobile Det model for object (text) detection tasks. Wraps the core PPOCRV5MobileDetModel
    and returns outputs compatible with the Transformers object detection API.
    """
)
class PPOCRV5MobileDetForObjectDetection(PPOCRV5ServerDetForObjectDetection):
    pass


__all__ = [
    "PPOCRV5MobileDetForObjectDetection",
    "PPOCRV5MobileDetConfig",
    "PPOCRV5MobileDetModel",
    "PPOCRV5MobileDetPreTrainedModel",
]
