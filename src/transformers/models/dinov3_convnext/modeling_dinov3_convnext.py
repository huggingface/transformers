# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ConvNext model."""

from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ...utils.generic import can_return_tuple
from .configuration_dinov3_convnext import DINOv3ConvNextConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->DINOv3ConvNext
class DINOv3ConvNextDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class DINOv3ConvNextLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, *args, data_format="channels_last", **kwargs):
        super().__init__(*args, **kwargs)
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.data_format = data_format

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, channels, height, width) OR (batch_size, height, width, channels)
        """
        if self.data_format == "channels_first":
            features = features.permute(0, 2, 3, 1)
            features = super().forward(features)
            features = features.permute(0, 3, 1, 2)
        else:
            features = super().forward(features)
        return features


class DINOv3ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations:
     1) DwConv, LayerNorm (channels_first), Conv, GELU, Conv (all in (N, C, H, W) format)
     2) DwConv, Permute, LayerNorm (channels_last), Linear, GELU, Linear, Permute

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`DINOv3ConvNextConfig`]):
            Model config.
        channels (`int`):
            Number of input (and output) channels.
        drop_path (`float`):
            Drop path rate. Default: 0.0.
    """

    def __init__(self, config: DINOv3ConvNextConfig, channels: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.layer_norm = DINOv3ConvNextLayerNorm(channels, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Linear(channels, 4 * channels)  # can be seen as a 1x1 conv
        self.activation_fn = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Linear(4 * channels, channels)  # can be seen as a 1x1 conv
        self.gamma = nn.Parameter(torch.full((channels,), config.layer_scale_init_value), requires_grad=True)
        self.drop_path = DINOv3ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, channels, height, width)
        """
        residual = features
        features = self.depthwise_conv(features)
        features = features.permute(0, 2, 3, 1)  # to channels last
        features = self.layer_norm(features)
        features = self.pointwise_conv1(features)
        features = self.activation_fn(features)
        features = self.pointwise_conv2(features)
        features = features * self.gamma
        features = features.permute(0, 3, 1, 2)  # back to channels first
        features = residual + self.drop_path(features)
        return features


class DINOv3ConvNextStage(nn.Module):
    """ """

    def __init__(self, config: DINOv3ConvNextConfig, stage_idx: int):
        super().__init__()

        in_channels = config.hidden_sizes[stage_idx - 1] if stage_idx > 0 else config.num_channels
        out_channels = config.hidden_sizes[stage_idx]

        if stage_idx == 0:
            self.downsample_layers = nn.ModuleList(
                [
                    nn.Conv2d(config.num_channels, out_channels, kernel_size=4, stride=4),
                    DINOv3ConvNextLayerNorm(out_channels, eps=config.layer_norm_eps, data_format="channels_first"),
                ]
            )
        else:
            self.downsample_layers = nn.ModuleList(
                [
                    DINOv3ConvNextLayerNorm(in_channels, eps=config.layer_norm_eps, data_format="channels_first"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                ]
            )

        num_stage_layers = config.depths[stage_idx]
        num_previous_layers = sum(config.depths[:stage_idx])
        num_total_layers = sum(config.depths)
        drop_path_rates = np.linspace(0, config.drop_path_rate, num_total_layers).tolist()

        self.layers = nn.ModuleList(
            [
                DINOv3ConvNextLayer(config, channels=out_channels, drop_path=drop_path_rates[i])
                for i in range(num_previous_layers, num_previous_layers + num_stage_layers)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, channels, height, width)
        """
        for layer in self.downsample_layers:
            features = layer(features)
        for layer in self.layers:
            features = layer(features)
        return features


@auto_docstring
class DINOv3ConvNextPreTrainedModel(PreTrainedModel):
    config: DINOv3ConvNextConfig
    base_model_prefix = "dinov3_convnext"
    main_input_name = "pixel_values"
    _no_split_modules = ["DINOv3ConvNextLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, DINOv3ConvNextLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DINOv3ConvNextLayer):
            if module.gamma is not None:
                module.gamma.data.fill_(self.config.layer_scale_init_value)


@auto_docstring
class DINOv3ConvNextModel(DINOv3ConvNextPreTrainedModel):
    def __init__(self, config: DINOv3ConvNextConfig):
        super().__init__(config)
        self.config = config
        self.stages = nn.ModuleList([DINOv3ConvNextStage(config, stage_idx) for stage_idx in range(config.num_stages)])
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)  # final norm layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, output_hidden_states: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        hidden_states = pixel_values

        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        all_hidden_states = [hidden_states] if output_hidden_states else []

        for stage in self.stages:
            hidden_states = stage(hidden_states)

            # store intermediate stage outputs
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # make global representation, a.k.a [CLS] token
        pooled_output = self.pool(hidden_states)

        # (batch_size, channels, height, width) -> (batch_size, height * width, channels)
        pooled_output = pooled_output.flatten(2).transpose(1, 2)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # concat "cls" and "patch tokens" as (batch_size, 1 + height * width, channels)
        hidden_states = torch.cat([pooled_output, hidden_states], dim=1)
        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:, 0],
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


__all__ = ["DINOv3ConvNextModel", "DINOv3ConvNextPreTrainedModel"]
