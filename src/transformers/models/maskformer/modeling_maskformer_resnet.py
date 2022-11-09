# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.s and The HuggingFace Inc. team. All rights reserved.
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

"""MaskFormer ResNet backbone."""

from typing import List, Optional

from torch import Tensor, nn

from ...activations import ACT2FN
from ...backbone import Backbone, ShapeSpec
from ...modeling_outputs import BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..resnet import ResNetConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.resnet.modeling_resnet.ResNetConvLayer with ResNet->MaskFormerResNet
class MaskFormerResNetConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetEmbeddings with ResNetConvLayer->MaskFormerResNetConvLayer
class MaskFormerResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.embedder = MaskFormerResNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


# Copied from transformers.models.resnet.modeling_resnet.ResNetShortCut
class MaskFormerResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetBasicLayer with ResNetShortCut->MaskFormerResNetShortCut, ResNetConvLayer->MaskFormerResNetConvLayer
class MaskFormerResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu"):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = (
            MaskFormerResNetShortCut(in_channels, out_channels, stride=stride)
            if should_apply_shortcut
            else nn.Identity()
        )
        self.layer = nn.Sequential(
            MaskFormerResNetConvLayer(in_channels, out_channels, stride=stride),
            MaskFormerResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetBottleNeckLayer with ResNetShortCut->MaskFormerResNetShortCut, ResNetConvLayer->MaskFormerResNetConvLayer
class MaskFormerResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", reduction: int = 4
    ):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = (
            MaskFormerResNetShortCut(in_channels, out_channels, stride=stride)
            if should_apply_shortcut
            else nn.Identity()
        )
        self.layer = nn.Sequential(
            MaskFormerResNetConvLayer(in_channels, reduces_channels, kernel_size=1),
            MaskFormerResNetConvLayer(reduces_channels, reduces_channels, stride=stride),
            MaskFormerResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetStage with ResNetBottleNeckLayer->MaskFormerResNetBottleNeckLayer, ResNetBasicLayer->MaskFormerResNetBasicLayer
class MaskFormerResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        layer = MaskFormerResNetBottleNeckLayer if config.layer_type == "bottleneck" else MaskFormerResNetBasicLayer

        self.layers = nn.Sequential(
            # downsampling is done in the first layer with stride of 2
            layer(in_channels, out_channels, stride=stride, activation=config.hidden_act),
            *[layer(out_channels, out_channels, activation=config.hidden_act) for _ in range(depth - 1)],
        )

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


# Copied from transformers.models.resnet.modeling_resnet.ResNetEncoder with ResNetStage->MaskFormerResNetStage
class MaskFormerResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            MaskFormerResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(MaskFormerResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# Copied from transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel with ResNetModel->MaskFormerResNetModel
class MaskFormerResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MaskFormerResNetModel):
            module.gradient_checkpointing = value


class MaskFormerResNetModel(MaskFormerResNetPreTrainedModel):
    # Copied from transformers.models.resnet.modeling_resnet.ResNetModel.__init__ with ResNet->MaskFormerResNet
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = MaskFormerResNetEmbeddings(config)
        self.encoder = MaskFormerResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class MaskFormerResNetBackbone(Backbone):
    """
    This class converts [`MaskFormerResNetModel`] into a generic backbone.

    Args:
        config (`ResNetConfig`):
            The configuration used by [`MaskFormerResNetModel`].
    """

    def __init__(self, config: ResNetConfig, out_features):
        super().__init__()
        self.model = MaskFormerResNetModel(config)

        current_stride = self.model.embedder.embedder.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": config.embedding_size}
        self._out_features = out_features

    def forward(self, *args, **kwargs) -> List[Tensor]:
        output = self.model(*args, **kwargs, output_hidden_states=True)
        return output

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }
