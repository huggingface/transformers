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
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
)
from ..hgnet_v2.modeling_hgnet_v2 import HGNetV2LearnableAffineBlock
from ..mobilenet_v2.modeling_mobilenet_v2 import make_divisible
from ..pp_lcnet.configuration_pp_lcnet import PPLCNetConfig
from ..pp_lcnet.modeling_pp_lcnet import (
    PPLCNetBackbone,
    PPLCNetConvLayer,
    PPLCNetDepthwiseSeparableConvLayer,
    PPLCNetEncoder,
    PPLCNetPreTrainedModel,
)


@auto_docstring(checkpoint="PaddlePaddle/Not_yet_released")
@strict
class PPLCNetV3Config(PPLCNetConfig):
    r"""
    scale (`float`, *optional*, defaults to 1.0):
        The scaling factor for the model's channel dimensions, used to adjust the model size and computational cost
        without changing the overall architecture (e.g., 0.25, 0.5, 1.0, 1.5).
    block_configs (`list[list[tuple]]`, *optional*, defaults to `None`):
        Configuration for each block in each stage. Each tuple contains:
        (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation).
        If `None`, uses the default PP-LCNet configuration.
    stem_channels (`int`, *optional*, defaults to 16):
        The number of output channels for the stem layer.
    stem_stride (`int`, *optional*, defaults to 2):
        The stride for the stem convolution layer.
    reduction (`int`, *optional*, defaults to 4):
        The reduction factor for feature channel dimensions in the squeeze-and-excitation (SE) blocks, used to
        reduce the number of model parameters and computational complexity while maintaining feature representability.
    divisor (`int`, *optional*, defaults to 8):
        The divisor used to ensure that various model parameters (e.g., channel dimensions, kernel sizes) are
        multiples of this value, promoting efficient model implementation and resource utilization.
    conv_symmetric_num (`int`, *optional*, defaults to `4`):
        The number of kxk convolution branches in the learnable reparameterization layer, used to enhance feature
        extraction capability through multi-branch architecture during training while enabling efficient inference
        via structural reparameterization.
    """

    model_type = "pp_lcnet_v3"

    conv_symmetric_num: int = 4
    hidden_dropout_prob = AttributeError()
    class_expand = AttributeError()

    def __post_init__(self, **kwargs):
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
            if self.block_configs is None
            else self.block_configs
        )
        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        PreTrainedConfig.__post_init__(**kwargs)


class PPLCNetV3ConvLayer(PPLCNetConvLayer):
    pass


class PPLCNetV3LearnableAffineBlock(HGNetV2LearnableAffineBlock):
    pass


class PPLCNetV3ActLearnableAffineBlock(nn.Module):
    """
    Activation block with a trainable affine transformation applied after the non-linear activation.
    """

    def __init__(self, activation="hardswish"):
        super().__init__()
        self.act = ACT2FN[activation]
        self.lab = PPLCNetV3LearnableAffineBlock()

    def forward(self, hidden_state: torch.Tensor):
        return self.lab(self.act(hidden_state))


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

        self.conv_symmetric = nn.ModuleList(
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

        self.conv_small_symmetric = (
            PPLCNetV3ConvLayer(in_channels, out_channels, 1, stride, groups=groups, activation=None)
            if kernel_size > 1
            else None
        )

        self.lab = PPLCNetV3LearnableAffineBlock()
        self.act = PPLCNetV3ActLearnableAffineBlock(activation=activation)

    def forward(self, hidden_state: torch.Tensor):
        output = None

        if self.identity is not None:
            output = self.identity(hidden_state)

        if self.conv_small_symmetric is not None:
            residual = self.conv_small_symmetric(hidden_state)
            output = residual if output is None else output + residual

        for conv in self.conv_symmetric:
            residual = conv(hidden_state)
            output = residual if output is None else output + residual

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
            num_conv_branches=config.conv_symmetric_num,
            activation=config.hidden_act,
        )
        self.pointwise_convolution = PPLCNetV3LearnableRepLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            num_conv_branches=config.conv_symmetric_num,
            activation=config.hidden_act,
        )


class PPLCNetV3PreTrainedModel(PPLCNetPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, (PPLCNetV3LearnableAffineBlock)):
            nn.init.ones_(module.scale)
            nn.init.zeros_(module.bias)


class PPLCNetV3Backbone(PPLCNetBackbone):
    pass


class PPLCNetV3Encoder(PPLCNetEncoder):
    def __init__(self, config: PPLCNetV3Config):
        super().__init__(config)
        self.config = config

        # stem
        self.convolution = PPLCNetV3ConvLayer(
            in_channels=3,
            kernel_size=3,
            out_channels=make_divisible(config.stem_channels * config.scale, config.divisor),
            stride=config.stem_stride,
            activation=None,
        )


__all__ = [
    "PPLCNetV3Backbone",
    "PPLCNetV3Config",
    "PPLCNetV3PreTrainedModel",
]
