# Copyright 2024 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch RTDetr specific ResNet model. The main difference between hugginface ResNet model is that this RTDetrResNet model forces to use shortcut at the first layer in the resnet-18/34 models.
See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/presnet.py#L126 for details.
"""

from torch import Tensor, nn

from ...activations import ACT2FN
from ...backbone_utils import filter_output_hidden_states
from ...modeling_outputs import BackboneOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ..resnet.modeling_resnet import (
    ResNetBackbone,
    ResNetConvLayer,
    ResNetEncoder,
    ResNetPreTrainedModel,
    ResNetShortCut,
)
from .configuration_rt_detr_resnet import RTDetrResNetConfig


logger = logging.get_logger(__name__)


class RTDetrResNetConvLayer(ResNetConvLayer):
    pass


class RTDetrResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a deep aggressive convolution.
    """

    def __init__(self, config: RTDetrResNetConfig):
        super().__init__()
        self.embedder = nn.Sequential(
            *[
                RTDetrResNetConvLayer(
                    config.num_channels,
                    config.embedding_size // 2,
                    kernel_size=3,
                    stride=2,
                    activation=config.hidden_act,
                ),
                RTDetrResNetConvLayer(
                    config.embedding_size // 2,
                    config.embedding_size // 2,
                    kernel_size=3,
                    stride=1,
                    activation=config.hidden_act,
                ),
                RTDetrResNetConvLayer(
                    config.embedding_size // 2,
                    config.embedding_size,
                    kernel_size=3,
                    stride=1,
                    activation=config.hidden_act,
                ),
            ]
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


class RTDetrResNetShortCut(ResNetShortCut):
    pass


class RTDetrResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    See https://github.com/lyuwenyu/RT-DETR/blob/5b628eaa0a2fc25bdafec7e6148d5296b144af85/rtdetr_pytorch/src/nn/backbone/presnet.py#L34.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        should_apply_shortcut: bool = False,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut = (
                nn.Sequential(
                    *[nn.AvgPool2d(2, 2, 0, ceil_mode=True), RTDetrResNetShortCut(in_channels, out_channels, stride=1)]
                )
                if should_apply_shortcut
                else nn.Identity()
            )
        else:
            self.shortcut = (
                RTDetrResNetShortCut(in_channels, out_channels, stride=stride)
                if should_apply_shortcut
                else nn.Identity()
            )
        self.layer = nn.Sequential(
            RTDetrResNetConvLayer(in_channels, out_channels, stride=stride),
            RTDetrResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrResNetBottleNeckLayer(nn.Module):
    """
    A classic RTDetrResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        reduction = 4
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        if stride == 2:
            self.shortcut = nn.Sequential(
                *[
                    nn.AvgPool2d(2, 2, 0, ceil_mode=True),
                    RTDetrResNetShortCut(in_channels, out_channels, stride=1)
                    if should_apply_shortcut
                    else nn.Identity(),
                ]
            )
        else:
            self.shortcut = (
                RTDetrResNetShortCut(in_channels, out_channels, stride=stride)
                if should_apply_shortcut
                else nn.Identity()
            )
        self.layer = nn.Sequential(
            RTDetrResNetConvLayer(
                in_channels, reduces_channels, kernel_size=1, stride=stride if config.downsample_in_bottleneck else 1
            ),
            RTDetrResNetConvLayer(
                reduces_channels, reduces_channels, stride=stride if not config.downsample_in_bottleneck else 1
            ),
            RTDetrResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RTDetrResNetStage(nn.Module):
    """
    A RTDetrResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: RTDetrResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        layer = RTDetrResNetBottleNeckLayer if config.layer_type == "bottleneck" else RTDetrResNetBasicLayer

        if config.layer_type == "bottleneck":
            first_layer = layer(
                config,
                in_channels,
                out_channels,
                stride=stride,
            )
        else:
            first_layer = layer(config, in_channels, out_channels, stride=stride, should_apply_shortcut=True)
        self.layers = nn.Sequential(
            first_layer, *[layer(config, out_channels, out_channels) for _ in range(depth - 1)]
        )

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


@auto_docstring
class RTDetrResNetPreTrainedModel(ResNetPreTrainedModel):
    config: RTDetrResNetConfig
    # Keep "resnet" for checkpoint BC (weight keys under resnet.*)
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["RTDetrResNetConvLayer", "RTDetrResNetShortCut"]


class RTDetrResNetEncoder(ResNetEncoder):
    _can_record_outputs = {"hidden_states": RTDetrResNetStage}

    def __init__(self, config: RTDetrResNetConfig):
        super().__init__(config)
        self.stages = nn.ModuleList([])
        # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
        self.stages.append(
            RTDetrResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RTDetrResNetStage(config, in_channels, out_channels, depth=depth))


@auto_docstring(
    custom_intro="""
    ResNet backbone, to be used with frameworks like RTDETR.
    """
)
class RTDetrResNetBackbone(RTDetrResNetPreTrainedModel, ResNetBackbone):
    has_attentions = False

    def __init__(self, config):
        super().__init__(config)

        self.num_features = [config.embedding_size] + config.hidden_sizes
        self.embedder = RTDetrResNetEmbeddings(config)
        self.encoder = RTDetrResNetEncoder(config)

        # initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(self, pixel_values: Tensor, **kwargs: Unpack[TransformersKwargs]) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import RTDetrResNetConfig, RTDetrResNetBackbone
        >>> import torch

        >>> config = RTDetrResNetConfig()
        >>> model = RTDetrResNetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        embedding_output = self.embedder(pixel_values)
        encoder_outputs = self.encoder(embedding_output, **kwargs)
        hidden_states = encoder_outputs.hidden_states
        feature_maps = tuple(
            hidden_states[idx] for idx, stage in enumerate(self.stage_names) if stage in self.out_features
        )
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states,
            attentions=None,
        )


__all__ = ["RTDetrResNetBackbone", "RTDetrResNetPreTrainedModel"]
