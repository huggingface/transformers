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

from ...utils import (
    auto_docstring,
)
from ..pp_lcnet.configuration_pp_lcnet import PPLCNetConfig
from ..pp_lcnet.modeling_pp_lcnet import (
    PPLCNetBackbone,
    PPLCNetConvLayer,
    PPLCNetEmbeddings,
    PPLCNetEncoder,
    PPLCNetPreTrainedModel,
    PPLCNetSEModule,
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
        **kwargs,
    ):
        super().__init__()
        # Default block configs for PP-LCNetV3
        # Each tuple: (kernel_size, in_channels, out_channels, stride, use_squeeze_excitation)
        self.block_configs = [
            # Stage 1 (blocks2)
            [[3, 16, 32, 1, False]],
            # Stage 2 (blocks3)
            [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
            # Stage 3 (blocks4)
            [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
            # Stage 4 (blocks5)
            [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
            [5, 256, 256, 1, False], [5, 256, 256, 1, False],
            [5, 256, 256, 1, False]],
            # Stage 5 (blocks6)
            [[5, 256, 512, 2, True], [5, 512, 512, 1, True],
            [5, 512, 512, 1, False], [5, 512, 512, 1, False]],
        ] if block_configs is None else block_configs


class PPLCNetV3ConvLayer(PPLCNetConvLayer):
    pass


class PPLCNetDepthwiseSeparableConvLayer(nn.Module):
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
        reduction: int,
        kernel_size=3,
        use_squeeze_excitation=False,
        activation="hardswish",
    ):
        """
        Initialize the PPLCNetDepthwiseSeparableConvLayer module.

        Args:
            in_channels (int): Number of channels of the input feature map.
            out_channels (int): Number of channels of the output feature map.
            stride (int): Stride of the depthwise convolution.
            reduction (int): Reduction ratio for SE module.
            depthwise_size (int, optional): Kernel size of depthwise convolution. Defaults to 3.
            use_squeeze_excitation (bool, optional): Whether to use SE module. Defaults to False.
            activation (str, optional): Name of activation function. Defaults to "hardswish".
        """
        super().__init__()
        self.use_squeeze_excitation = use_squeeze_excitation
        self.depthwise_convolution = PPLCNetV3ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=activation,
        )
        self.squeeze_excitation_module = (
            PPLCNetV3SEModule(in_channels, reduction) if use_squeeze_excitation else nn.Identity()
        )
        self.pointwise_convolution = PPLCNetV3ConvLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            activation=activation,
        )

    def forward(self, hidden_state):
        """
        Forward propagation logic.

        Args:
            hidden_state (FloatTensor): Input feature map with shape [B, C, H, W].

        Returns:
            FloatTensor: Output feature map with shape [B, out_channels, H', W'].
        """
        hidden_state = self.depthwise_convolution(hidden_state)
        hidden_state = self.squeeze_excitation_module(hidden_state)
        hidden_state = self.pointwise_convolution(hidden_state)
        return hidden_state


class PPLCNetV3SEModule(PPLCNetSEModule):
    pass


class PPLCNetV3PreTrainedModel(PPLCNetPreTrainedModel):
    pass


class PPLCNetV3Embeddings(PPLCNetEmbeddings):
    pass


class PPLCNetV3Encoder(PPLCNetEncoder):
    pass


class PPLCNetV3Backbone(PPLCNetBackbone):
    pass



__all__ = [
    "PPLCNetV3Backbone",
    "PPLCNetV3Config",
    "PPLCNetV3PreTrainedModel",
]
