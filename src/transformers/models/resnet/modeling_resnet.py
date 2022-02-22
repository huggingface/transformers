# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ResNet model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_resnet import ResNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"
_FEAT_EXTRACTOR_FOR_DOC = "ConvNextFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = ""
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 14, 14]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = ""
_IMAGE_CLASS_EXPECTED_OUTPUT = "'tabby, tabby cat'"

RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Francesco/resnet50-224-1k",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]


@dataclass
class ResNetEncoderOutput(ModelOutput):
    """
    Class for [`ResNetEncoder`]'s outputs, with potential hidden states (feature maps).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ResNetModelOutput(ModelOutput):
    """
    Class for [`ResNetModel`]'s outputs, with potential hidden states (feature maps).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, config.dim[-1])`):
            Global average pooling of the last feature map followed by a layernorm.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ResNetClassifierOutput(ModelOutput):
    """
    Class for [`ResNetForImageClassification`]'s outputs, with potential hidden states (feature maps).

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at
            the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ResNetConvBnActLayer(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ACT2FN[activation] if activation is not None else nn.Identity()


class ResNetEmbeddings(nn.Sequential):
    """
    ResNet Embedddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, num_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.embedder = ResNetConvBnActLayer(
            num_channels, out_channels, kernel_size=7, stride=2, activation=activation
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


class ResNetEmbeddings3x3(nn.Sequential):
    """
    Modified ResNet Embedddings (stem) proposed in [Bag of Tricks for Image Classification with Convolutional Neural
    Networks](https://arxiv.org/pdf/1812.01187.pdf) The observation is that the computational cost of a convolution is
    quadratic to the kernel width or height. A `7x7` convolution is `5.4` times more expensive than a `3x3`
    convolution. So this tweak replacing the `7x7` convolution in the input stem with three conservative `3x3`
    convolution.
    """

    def __init__(
        self, num_channels: int, out_channels: int, hidden_channels: List[int] = None, activation: str = "relu"
    ):
        super().__init__()
        if hidden_channels is None:
            # default to stemD in the paper
            hidden_channels = [32, 32]

        hidden_channels = hidden_channels + [out_channels]

        self.embedder = nn.Sequential(
            ResNetConvBnActLayer(num_channels, hidden_channels[0], kernel_size=3, stride=2, activation=activation),
            *[
                ResNetConvBnActLayer(in_channels, out_channels, kernel_size=3, activation=activation)
                for in_channels, out_channels in zip(hidden_channels, hidden_channels[1:])
            ],
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


class ResNetShortCut(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)


class ResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by a two `3x3` convolutions.

    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of outputs channels.
        stride (`int`, *optional`, defaults to 1):
            The stride used in the first convolution.
        activation (`int`, *optional*, defaults to `"relu"`):
            The activation used by the layer.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu"):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvBnActLayer(in_channels, out_channels, stride=stride),
            ResNetConvBnActLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, pixel_values):
        residual = pixel_values
        pixel_values = self.layer(pixel_values)
        residual = self.shortcut(residual)
        pixel_values += residual
        pixel_values = self.activation(pixel_values)
        return pixel_values


class ResNetBottleNeckLayer(nn.Module):
    """

    A classic ResNet's bottleneck layer composed by a three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remap the reduced features to `out_channels`.


    Args:
        in_channels (`int`):
            The number of input channels.
        out_channels (`int`):
            The number of outputs channels.
        stride (`int`, *optional`, defaults to 1):
            The stride used in the first convolution.
        activation (`int`, *optional*, defaults to `"relu"`):
            The activation used by the layer.
        reduction (`int`, *optional*, defaults to 4):
            The reduction factor the block applies in the first `1x1` convolution.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", reduction: int = 4
    ):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetConvBnActLayer(in_channels, reduces_channels, kernel_size=1),
            ResNetConvBnActLayer(reduces_channels, reduces_channels, stride=stride),
            ResNetConvBnActLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, pixel_values):
        residual = pixel_values
        pixel_values = self.layer(pixel_values)
        residual = self.shortcut(residual)
        pixel_values += residual
        pixel_values = self.activation(pixel_values)
        return pixel_values


class ResNetStage(nn.Sequential):
    """
    A ResNet stage composed by stacked layers.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.
        stride (`int`, *optional*, defaults to 1):
            The first layer stride, used to downsample the input.
        depth (`int`,*optional*, defaults to 2):
            The number of layers.
        layer_type (`str`, *optional*, defaults to `"basic"`):
            The type of layer, either `"basic"` or `"bottleneck"`.
        activation (`int`, *optional*, defaults to `"relu"`):
            The activation used by all layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
        layer_type: str = "basic",
        activation: str = "relu",
    ):
        super().__init__()

        layer = ResNetBottleNeckLayer if layer_type == "bottleneck" else ResNetBasicLayer

        self.layers = nn.Sequential(
            # downsampling is done in the first layer with stride of 2
            layer(in_channels, out_channels, stride=stride, activation=activation),
            *[layer(out_channels, out_channels, activation=activation) for _ in range(depth - 1)],
        )


class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        # the first block doesn't downsample
        self.stages.append(
            ResNetStage(
                config.hidden_sizes[0],
                config.hidden_sizes[1],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
                layer_type=config.layer_type,
                activation=config.hidden_act,
            )
        )
        remaining_hidden_sizes = config.hidden_sizes[1:]
        in_out_channels = zip(remaining_hidden_sizes, remaining_hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(
                ResNetStage(
                    in_channels, out_channels, depth=depth, layer_type=config.layer_type, activation=config.hidden_act
                )
            )

    def forward(
        self, hidden_states: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> ResNetEncoderOutput:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return ResNetEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ResNetPreTrainedModel(PreTrainedModel):
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
        if isinstance(module, ResNetModel):
            module.gradient_checkpointing = value


RESNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        embeddings = ResNetEmbeddings3x3 if config.embeddings_type == "3x3d" else ResNetEmbeddings
        self.embedder = embeddings(config.num_channels, config.hidden_sizes[0], config.hidden_act)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ResNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return ResNetModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class ResNetForImageClassification(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ResNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ResNetClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs: ResNetModelOutput = self.model(
            pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        we_have_labels = labels is not None

        if we_have_labels:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                criterion = MSELoss()
                if self.num_labels == 1:
                    loss = criterion(logits.squeeze(), labels.squeeze())
                else:
                    loss = criterion(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                criterion = CrossEntropyLoss()
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                criterion = BCEWithLogitsLoss()
                loss = criterion(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ResNetClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
