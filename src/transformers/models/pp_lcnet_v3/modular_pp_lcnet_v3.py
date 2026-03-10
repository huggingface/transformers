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

from ...activations import ACT2FN
from ...utils import (
    auto_docstring,
)
from ..mobilenet_v2.modeling_mobilenet_v2 import make_divisible
from ..pp_lcnet.configuration_pp_lcnet import PPLCNetConfig
from ..pp_lcnet.modeling_pp_lcnet import (
    PPLCNetBackbone,
    PPLCNetConvLayer,
    PPLCNetDepthwiseSeparableConvLayer,
    PPLCNetEmbeddings,
    PPLCNetPreTrainedModel,
)


@auto_docstring(
    custom_intro="""
    """
)
class PPLCNetV3Config(PPLCNetConfig):
    model_type = "pp_lcnet_v3"

    def __init__(
        self,
        num_channels=3,
        scale=1.0,
        hidden_act="hardswish",
        out_features=None,
        out_indices=None,
        stem_channels=16,
        stem_stride=2,
        block_configs=None,
        reduction=4,
        dropout_prob=0.2,
        class_expand=1280,
        use_last_convolution=True,
        divisor=8,
        conv_kxk_num=4,
        **kwargs,
    ):
        self.conv_kxk_num = conv_kxk_num
        super().__init__()
        # Default block configs for PP-LCNetV3
        # Each tuple: (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation)
        self.block_configs = (
            [
                # Stage 1 (blocks2)
                [[3, 16, 32, 1, False]],
                # Stage 2 (blocks3)
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                # Stage 3 (blocks4)
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                # Stage 4 (blocks5)
                [
                    [3, 128, 256, 2, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                # Stage 5 (blocks6)
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True], [5, 512, 512, 1, False], [5, 512, 512, 1, False]],
            ]
            if block_configs is None
            else block_configs
        )


class PPLCNetV3ConvLayer(PPLCNetConvLayer):
    pass


class PPLCNetV3Embeddings(PPLCNetEmbeddings):
    def __init__(self, config: PPLCNetV3Config):
        super().__init__()
        self.convolution = PPLCNetV3ConvLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(config.stem_channels * config.scale, config.divisor),
            stride=config.stem_stride,
            activation=None,
        )


class PPLCNetV3LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, hidden_state):
        hidden_state = self.scale * hidden_state + self.bias
        return hidden_state


class PPLCNetV3Act(nn.Module):
    """
    Activation block with a trainable affine transformation applied after the non-linear activation.
    """

    def __init__(self, activation="hardswish"):
        super().__init__()
        self.act = ACT2FN[activation] if activation is not None else nn.Identity()
        self.lab = PPLCNetV3LearnableAffineBlock()

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.act(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class PPLCNetV3LearnableRepLayer(nn.Module):
    """
    Learnable Reparameterization Layer (RepLayer) that fuses multiple convolution branches
    (kxk and 1x1) with an optional identity branch. This layer enables structural reparameterization
    for efficient inference while maintaining training flexibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        stride: int,
        num_conv_branches: int,
        groups: int = 1,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

        self.conv_kxk = nn.ModuleList(
            [
                PPLCNetV3ConvLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    activation=None,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            PPLCNetV3ConvLayer(in_channels, out_channels, 1, stride, groups=groups, activation=None)
            if kernel_size > 1
            else None
        )

        self.lab = PPLCNetV3LearnableAffineBlock()
        self.act = PPLCNetV3Act(activation=activation)

    def forward(self, hidden_state: torch.Tensor):
        """
        Forward pass of the PPLCNetV3LearnableRepLayer, fusing all enabled branches and applying post-processing.

        Args:
            hidden_state (torch.Tensor): Input feature tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature tensor of shape (B, out_channels, H', W').
        """
        output = 0
        if self.identity is not None:
            output += self.identity(hidden_state)

        if self.conv_1x1 is not None:
            output += self.conv_1x1(hidden_state)

        for conv in self.conv_kxk:
            output += conv(hidden_state)

        hidden_state = self.lab(output)
        if self.stride != 2:
            hidden_state = self.act(hidden_state)
        return hidden_state


class PPLCNetV3DepthwiseSeparableConvLayer(PPLCNetDepthwiseSeparableConvLayer):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> SE Module (optional) -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        use_squeeze_excitation,
        config,
    ):
        super().__init__()
        self.depthwise_convolution = PPLCNetV3LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=config.conv_kxk_num,
            activation=config.hidden_act,
        )
        self.pointwise_convolution = PPLCNetV3LearnableRepLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            num_conv_branches=config.conv_kxk_num,
            activation=config.hidden_act,
        )


class PPLCNetV3PreTrainedModel(PPLCNetPreTrainedModel):
    pass


class PPLCNetV3Backbone(PPLCNetBackbone):
    pass


__all__ = [
    "PPLCNetV3Backbone",
    "PPLCNetV3Config",
    "PPLCNetV3PreTrainedModel",
]
