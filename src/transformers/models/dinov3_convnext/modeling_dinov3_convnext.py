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

from typing import Optional, Union

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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Dinov3ConvNext
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape (batch_size, channels, height, width) OR (batch_size, height, width, channels)
        """
        if self.data_format == "channels_first":
            output = input.permute(0, 2, 3, 1)
            output = super().forward(output)
            output = output.permute(0, 3, 1, 2)
        else:
            output = super().forward(input)
        return output


class DINOv3ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: 
     1) DwConv, LayerNorm (channels_first), Conv, GELU, Conv (all in (N, C, H, W) format)
     2) DwConv, Permute, LayerNorm (channels_last), Linear, GELU, Linear, Permute

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`DINOv3ConvNextConfig`]):
            Model config.
        dim (`int`):
            Number of input (and output) channels.
        drop_path (`float`):
            Drop path rate. Default: 0.0.
    """

    def __init__(self, config: DINOv3ConvNextConfig, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layer_norm = DINOv3ConvNextLayerNorm(dim, eps=config.layer_norm_eps)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)  # implemented with linear, but can be seen as a 1x1 conv
        self.activation_fn = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)  # implemented with linear, but can be seen as a 1x1 conv
        self.gamma = nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DINOv3ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, channels, height, width)
        """
        residual = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)  # to channels last
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to channels first
        x = residual + self.drop_path(x)
        return x


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
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(config.num_channels, config.hidden_sizes[0], kernel_size=4, stride=4),
            DINOv3ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                DINOv3ConvNextLayerNorm(config.hidden_sizes[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(
                    config.hidden_sizes[i],
                    config.hidden_sizes[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = np.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    DINOv3ConvNextLayer(
                        config=config,
                        dim=config.hidden_sizes[i],
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(config.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += config.depths[i]

        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)  # final norm layer
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_hidden_states = () if output_hidden_states else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = pixel_values
        for dw_layer, stage_layer in zip(self.downsample_layers, self.stages):
            hidden_states = stage_layer(dw_layer(hidden_states))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = hidden_states.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
        hidden_states = torch.flatten(hidden_states, 2).transpose(1, 2)

        # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
        hidden_states_norm = self.layer_norm(torch.cat([pooled_output.unsqueeze(1), hidden_states], dim=1))

        if not return_dict:
            return (hidden_states_norm, hidden_states_norm[:, 0], all_hidden_states)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=hidden_states_norm,
            pooler_output=hidden_states_norm[:, 0],
            hidden_states=all_hidden_states,
        )


__all__ = ["DINOv3ConvNextModel", "DINOv3ConvNextPreTrainedModel"]
