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
from dataclasses import dataclass

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
from ..auto import AutoConfig
from ..pp_lcnet.modeling_pp_lcnet import PPLCNetConvLayer, PPLCNetDepthwiseSeparableConvLayer
from ..slanext.configuration_slanext import SLANeXtConfig
from ..slanext.modeling_slanext import (
    SLANeXtForTableRecognition,
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
        The kernel size of the Cross Stage Partial (CSP) layer.
    csp_num_blocks (`int`, *optional*, defaults to 1):
        Number of blocks within the Cross Stage Partial (CSP) layer.
    """

    sub_configs = {"backbone_config": AutoConfig}

    vision_config = AttributeError()
    backbone_config: dict | PreTrainedConfig | None = None

    post_conv_in_channels = AttributeError()
    post_conv_out_channels: int = 96
    hidden_size: int = 256

    hidden_act: str = "hardswish"
    csp_kernel_size: int = 5
    csp_num_blocks: int = 1

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


@dataclass
@auto_docstring
class SLANetForTableRecognitionOutput(BaseModelOutputWithNoAttention):
    r"""
    head_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Hidden-states of the SLANetSLAHead at each prediction step, varies up to max `self.config.max_text_length` states (depending on early exits).
    head_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Attentions of the SLANetSLAHead at each prediction step, varies up to max `self.config.max_text_length` attentions (depending on early exits).
    """

    head_hidden_states: torch.FloatTensor | None = None
    head_attentions: torch.FloatTensor | None = None


class SLANetSLAHead(SLANeXtSLAHead):
    pass


class SLANetConvLayer(PPLCNetConvLayer):
    pass


class SLANetDepthwiseSeparableConvLayer(PPLCNetDepthwiseSeparableConvLayer):
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
        config,
    ):
        super().__init__()
        self.squeeze_excitation_module = nn.Identity()


class SLANetBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        config,
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
            config=config,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)

        return hidden_states


class SLANetCSPLayer(nn.Module):
    """
    Cross Stage Partial (CSP) network layer. Similar in structure to DFineCSPRepLayer, but with a different forward computation.
    """

    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        kernel_size=3,
        expansion=0.5,
        num_blocks=1,
        activation="hardswish",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = SLANetConvLayer(in_channels, hidden_channels, 1, activation=activation)
        self.conv2 = SLANetConvLayer(in_channels, hidden_channels, 1, activation=activation)
        self.conv3 = SLANetConvLayer(2 * hidden_channels, out_channels, 1, activation=activation)
        self.bottlenecks = nn.ModuleList(
            [
                SLANetBottleneck(hidden_channels, hidden_channels, kernel_size, activation, config)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        residual = self.conv1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        for bottleneck in self.bottlenecks:
            hidden_states = bottleneck(hidden_states)

        hidden_states = torch.cat((hidden_states, residual), dim=1)
        hidden_states = self.conv3(hidden_states)

        return hidden_states


class SLANetCSPPAN(nn.Module):
    """
    CSP-PAN: Path Aggregation Network with CSP layers
    """

    def __init__(
        self,
        config,
        in_channel_list,
    ):
        super().__init__()
        out_channels = config.post_conv_out_channels
        activation = config.hidden_act
        kernel_size = config.csp_kernel_size
        csp_num_blocks = config.csp_num_blocks

        self.channel_projector = nn.ModuleList(
            [
                SLANetConvLayer(
                    in_channels=in_channel_list[i], out_channels=out_channels, kernel_size=1, activation=activation
                )
                for i in range(len(in_channel_list))
            ]
        )

        # build top-down blocks
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList(
            [
                SLANetCSPLayer(
                    config,
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_num_blocks,
                    activation=activation,
                )
                for _ in range(len(in_channel_list) - 1, 0, -1)
            ]
        )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList(
            [
                SLANetDepthwiseSeparableConvLayer(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    config=config,
                )
                for _ in range(len(in_channel_list) - 1)
            ]
        )
        self.bottom_up_blocks = nn.ModuleList(
            [
                SLANetCSPLayer(
                    config,
                    out_channels * 2,
                    out_channels,
                    kernel_size=kernel_size,
                    num_blocks=csp_num_blocks,
                    activation=activation,
                )
                for _ in range(len(in_channel_list) - 1)
            ]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        projected_features = []
        for idx in range(len(self.channel_projector)):
            projected_features.append(self.channel_projector[idx](hidden_states[idx]))

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


class SLANetBackbone(SLANetPreTrainedModel):
    def __init__(self, config: SLANetConfig):
        super().__init__(config)
        self.vision_backbone = load_backbone(config)
        self.post_csp_pan = SLANetCSPPAN(config, self.vision_backbone.num_features[2:])

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, hidden_states: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | BaseModelOutputWithNoAttention:
        outputs = self.vision_backbone(hidden_states, **kwargs)
        hidden_states = self.post_csp_pan(outputs.feature_maps)
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
class SLANetForTableRecognition(SLANeXtForTableRecognition):
    _keys_to_ignore_on_load_missing = ["num_batches_tracked"]

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[torch.FloatTensor] | SLANetForTableRecognitionOutput:
        outputs = self.backbone(pixel_values, **kwargs)
        head_outputs = self.head(outputs.last_hidden_state, **kwargs)
        # Key difference: no attentions in its vision model
        return SLANetForTableRecognitionOutput(
            last_hidden_state=head_outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            head_hidden_states=head_outputs.hidden_states,
            head_attentions=head_outputs.attentions,
        )


__all__ = ["SLANetConfig", "SLANetForTableRecognition", "SLANetPreTrainedModel", "SLANetSLAHead", "SLANetBackbone"]
