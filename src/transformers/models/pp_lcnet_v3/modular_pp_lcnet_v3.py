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


@auto_docstring(checkpoint="PaddlePaddle/Not_yet_released")
class PPLCNetV3Config(PPLCNetConfig):
    r"""
    This is the configuration class to store the configuration of a [`PPLCNetV3`]. It is used to instantiate a
    PP-LCNet backbone according to the specified arguments, defining the model architecture.

    Args:
        scale (`float`, *optional*, defaults to 1.0):
            The scaling factor for the model's channel dimensions, used to adjust the model size and computational cost
            without changing the overall architecture (e.g., 0.25, 0.5, 1.0, 1.5).
        hidden_act (`str`, *optional*, defaults to `"hardswish"`):
            The non-linear activation function used in the model's hidden layers. Supported functions include
            `"hardswish"`, `"relu"`, `"silu"`, and `"gelu"`. `"hardswish"` is preferred for lightweight and efficient
            inference on edge devices.
        out_features (`list[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        stem_channels (`int`, *optional*, defaults to 16):
            The number of output channels for the stem layer.
        stem_stride (`int`, *optional*, defaults to 2):
            The stride for the stem convolution layer.
        block_configs (`list[list[tuple]]`, *optional*, defaults to `None`):
            Configuration for each block in each stage. Each tuple contains:
            (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation).
            If `None`, uses the default PP-LCNet configuration.
        reduction (`int`, *optional*, defaults to 4):
            The reduction factor for feature channel dimensions in the squeeze-and-excitation (SE) blocks, used to
            reduce the number of model parameters and computational complexity while maintaining feature representability.
        dropout_prob (`float`, *optional*, defaults to 0.2):
            The dropout probability for the classification head, used to prevent overfitting by randomly zeroing out
            a fraction of the neurons during training.
        class_expand (`int`, *optional*, defaults to 1280):
            The number of hidden units in the expansion layer of the classification head, used to enhance the model's
            feature representation capability before the final classification layer.
        use_last_convolution (`bool`, *optional*, defaults to `True`):
            Whether to use the final convolutional layer in the classification head. Setting this to `True` helps
            extract more discriminative features for the classification task.
        divisor (`int`, *optional*, defaults to 8):
            The divisor used to ensure that various model parameters (e.g., channel dimensions, kernel sizes) are
            multiples of this value, promoting efficient model implementation and resource utilization.
        conv_kxk_num (`int`, *optional*, defaults to `4`):
            The number of kxk convolution branches in the learnable reparameterization layer, used to enhance feature
            extraction capability through multi-branch architecture during training while enabling efficient inference
            via structural reparameterization.
    """

    model_type = "pp_lcnet_v3"

    def __init__(
        self,
        scale=1.0,
        hidden_act="hardswish",
        out_features=None,
        out_indices=None,
        stem_channels=16,
        stem_stride=2,
        block_configs=None,
        reduction=4,
        divisor=8,
        conv_kxk_num=4,
        **kwargs,
    ):
        self.conv_kxk_num = conv_kxk_num
        super().__init__()
        del self.dropout_prob
        del self.class_expand
        del self.use_last_convolution
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
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, hidden_state):
        return hidden_state * self.scale + self.bias


class PPLCNetV3Act(nn.Module):
    """
    Activation block with a trainable affine transformation applied after the non-linear activation.
    """

    def __init__(self, activation="hardswish"):
        super().__init__()
        self.act = ACT2FN[activation] if activation is not None else nn.Identity()
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
        output = None

        if self.identity is not None:
            output = self.identity(hidden_state)

        if self.conv_1x1 is not None:
            branch = self.conv_1x1(hidden_state)
            output = branch if output is None else output + branch

        for conv in self.conv_kxk:
            branch = conv(hidden_state)
            output = branch if output is None else output + branch

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
