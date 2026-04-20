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


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..pp_lcnet.modeling_pp_lcnet import PPLCNetConvLayer
from ..slanext.configuration_slanext import SLANeXtConfig
from ..slanext.modeling_slanext import (
    SLANeXtForTableRecognitionOutput,
    SLANeXtPreTrainedModel,
    SLANeXtSLAHead,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/SLANet_plus_safetensors")
@strict
class SLANetConfig(SLANeXtConfig):
    r"""
    post_conv_out_channels (`int`, *optional*, defaults to 96):
        Number of output channels for the post-encoder convolution layer.
    out_channels (`int`, *optional*, defaults to 50):
        Vocabulary size for the table structure token prediction head, i.e., the number of distinct structure
        tokens the model can predict.
    hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the hidden states in the attention GRU cell and the structure/location prediction heads.
    max_text_length (`int`, *optional*, defaults to 500):
        Maximum number of autoregressive decoding steps (tokens) for the structure and location decoder.
    csp_kernel_size (`int`, *optional*, defaults to 5):
        The kernel size of the CSP layer.
    csp_blocks_num (`int`, *optional*, defaults to 1):
        Number of the CSP layer.
    """

    sub_configs = {"backbone_config": AutoConfig}

    vision_config = AttributeError()
    backbone_config: dict | PreTrainedConfig | None = None

    post_conv_in_channels = AttributeError()
    post_conv_out_channels: int = 96
    out_channels: int = 50
    hidden_size: int = 256
    max_text_length: int = 500

    hidden_act: str = "hardswish"
    csp_kernel_size: int = 5
    csp_blocks_num: int = 1

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="pp_lcnet",
            default_config_kwargs={
                "scale": 1,
                "out_features": ["stage2", "stage3", "stage4", "stage5"],
                "out_indices": [2, 3, 4, 5],
                "divisor": 16,
            },
            **kwargs,
        )
        PreTrainedConfig.__post_init__(**kwargs)


class SLANetPreTrainedModel(SLANeXtPreTrainedModel):
    base_model_prefix = "slanet"
    supports_gradient_checkpointing = False
    _keep_in_fp32_modules_strict = []

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(module)

        # Initialize GRUCell (replicates PyTorch default reset_parameters)
        if isinstance(module, nn.GRUCell):
            std = 1.0 / math.sqrt(module.hidden_size) if module.hidden_size > 0 else 0
            init.uniform_(module.weight_ih, -std, std)
            init.uniform_(module.weight_hh, -std, std)
            if module.bias_ih is not None:
                init.uniform_(module.bias_ih, -std, std)
            if module.bias_hh is not None:
                init.uniform_(module.bias_hh, -std, std)

        # Initialize SLAHead layers
        if isinstance(module, SLANetSLAHead):
            std = 1.0 / math.sqrt(self.config.hidden_size * 1.0)
            # Initialize structure_generator and loc_generator layers
            for generator in (module.structure_generator,):
                for layer in generator.children():
                    if isinstance(layer, nn.Linear):
                        init.uniform_(layer.weight, -std, std)
                        if layer.bias is not None:
                            init.uniform_(layer.bias, -std, std)


class SLANetForTableRecognitionOutput(SLANeXtForTableRecognitionOutput):
    pass


class SLANetSLAHead(SLANeXtSLAHead):
    pass


class SLANetConvLayer(PPLCNetConvLayer):
    pass


class SLANetDepthwiseSeparableConvLayer(nn.Module):
    """
    Depthwise Separable Convolution Layer: Depthwise Conv -> Pointwise Conv
    Core component of lightweight models (e.g., MobileNet, PP-LCNet) that significantly reduces
    the number of parameters and computational cost.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        activation,
    ):
        super().__init__()
        self.depthwise_convolution = SLANetConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=activation,
        )
        self.pointwise_convolution = SLANetConvLayer(
            in_channels=in_channels,
            kernel_size=1,
            out_channels=out_channels,
            stride=1,
            activation=activation,
        )

    def forward(self, hidden_states):
        hidden_states = self.depthwise_convolution(hidden_states)
        hidden_states = self.pointwise_convolution(hidden_states)

        return hidden_states


class SLANetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
    ):
        super().__init__()
        self.conv1 = SLANetConvLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, activation=activation
        )
        self.conv2 = SLANetDepthwiseSeparableConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            activation=activation,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states


class SLANetCSPLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        expand_ratio=0.5,
        num_blocks=1,
        activation="hardswish",
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = SLANetConvLayer(in_channels, mid_channels, 1, activation=activation)
        self.short_conv = SLANetConvLayer(in_channels, mid_channels, 1, activation=activation)
        self.final_conv = SLANetConvLayer(2 * mid_channels, out_channels, 1, activation=activation)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                SLANetBottleneck(
                    mid_channels,
                    mid_channels,
                    kernel_size,
                    activation,
                )
            )

    def forward(self, hidden_states):
        hidden_states_short = self.short_conv(hidden_states)

        hidden_states_main = self.main_conv(hidden_states)
        for block in self.blocks:
            hidden_states_main = block(hidden_states_main)

        hidden_states = torch.cat((hidden_states_main, hidden_states_short), dim=1)
        hidden_states = self.final_conv(hidden_states)

        return hidden_states


class SLANetChannelProjector(nn.Module):
    def __init__(self, in_channel_list, out_channels, activation):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(in_channel_list)):
            self.layers.append(
                SLANetConvLayer(
                    in_channels=in_channel_list[i], out_channels=out_channels, kernel_size=1, activation=activation
                )
            )

    def forward(self, hidden_states):
        projected_features = []
        for idx in range(len(self.layers)):
            projected_features.append(self.layers[idx](hidden_states[idx]))
        return projected_features


class SLANetCSPPAN(nn.Module):
    """
    CSP-PAN: Path Aggregation Network with CSP layers
    """

    def __init__(
        self,
        in_channel_list,
        config,
    ):
        super().__init__()
        out_channels = config.post_conv_out_channels
        activation = config.hidden_act
        kernel_size = config.csp_kernel_size
        csp_blocks_num = config.csp_blocks_num

        self.channel_projector = SLANetChannelProjector(in_channel_list, out_channels, activation)

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channel_list) - 1, 0, -1):
            self.top_down_blocks.append(
                SLANetCSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_blocks_num,
                    activation=activation,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channel_list) - 1):
            self.downsamples.append(
                SLANetDepthwiseSeparableConvLayer(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    activation=activation,
                )
            )
            self.bottom_up_blocks.append(
                SLANetCSPLayer(
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_blocks_num,
                    activation=activation,
                )
            )

    def forward(self, hidden_states):
        projected_features = self.channel_projector(hidden_states)

        top_down_features = [projected_features[-1]]
        for top_down_block, low_level_feature in zip(self.top_down_blocks, reversed(projected_features[:-1])):
            high_level_feature = top_down_features[-1]
            upsampled_feature = F.interpolate(
                high_level_feature,
                size=low_level_feature.shape[-2:],
                mode="nearest",
            )
            fused_feature = top_down_block(torch.cat([upsampled_feature, low_level_feature], dim=1))
            top_down_features.append(fused_feature)

        pyramid_features = list(reversed(top_down_features))
        output_feature = pyramid_features[0]
        for downsample_layer, bottom_up_block, high_level_feature in zip(
            self.downsamples, self.bottom_up_blocks, pyramid_features[1:]
        ):
            downsampled_feature = downsample_layer(output_feature)
            output_feature = bottom_up_block(torch.cat([downsampled_feature, high_level_feature], dim=1))

        hidden_states = output_feature.flatten(2).transpose(1, 2)
        return hidden_states


class SLANetModel(SLANetPreTrainedModel):
    def __init__(self, config: SLANetConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = SLANetCSPPAN(self.backbone.num_features[2:], config)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | SLANetForTableRecognitionOutput:
        outputs = self.backbone(hidden_states, **kwargs)
        hidden_states = self.neck(outputs.feature_maps)
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    SLANet Table Recognition model for table recognition tasks. Wraps the core SLANetPreTrainedModel
    and returns outputs compatible with the Transformers table recognition API.
    """
)
class SLANetForTableRecognition(SLANetPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    def __init__(self, config: SLANetConfig):
        super().__init__(config)
        self.model = SLANetModel(config=config)
        self.head = SLANetSLAHead(config=config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | SLANetForTableRecognitionOutput:
        outputs = self.model(pixel_values, **kwargs)
        head_outputs = self.head(outputs.last_hidden_state, **kwargs)
        return SLANetForTableRecognitionOutput(
            last_hidden_state=head_outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            head_hidden_states=head_outputs.hidden_states,
            head_attentions=head_outputs.attentions,
        )


__all__ = ["SLANetConfig", "SLANetForTableRecognition", "SLANetPreTrainedModel", "SLANetSLAHead", "SLANetModel"]
