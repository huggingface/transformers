# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EfficientNet model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import BackboneMixin, PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_efficientnet import EfficientNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/efficientnet-b7"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/efficientnet-b7"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/efficientnet-b7",
    # See all EfficientNet models at https://huggingface.co/models?filter=efficientnet
]


EFFICIENTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def round_filters(config: EfficientNetConfig, num_channels: int):
    """
    Round number of filters based on depth multiplier.
    """
    divisor = config.divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->EfficientNet
class EfficientNetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class EfficientNetEmbeddings(nn.Module):
    def __init__(self, config: EfficientNetConfig, out_channels: int):
        super().__init__()

        self.padding = nn.ZeroPad2d(padding=3)
        self.convolution = nn.Conv2d(
            config.num_channels, round_filters(config, 32), kernel_size=3, stride=2, padding="valid", bias=False
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.padding(pixel_values)
        features = self.convolution(features)
        features = self.batchnorm(features)
        features = self.activation(features)

        return features


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class EfficientNetExpansionLayer(nn.Module):
    """This corresponds to the expansion phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        stride (`int`): Stride size.
        drop_rate (`float`): Dropout rate to be used.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
    ):
        super().__init__()
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expand phase
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)

        return hidden_states


class EfficientNetDepthwiseLayer(nn.Module):
    """This corresponds to the depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        stride (`int`): Stride size.
        drop_rate (`float`): Dropout rate to be used.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        kernel_size: int,
    ):
        super().__init__()
        self.stride = stride
        conv_pad = "valid" if self.stride == 2 else "same"

        self.depthwise_conv_pad = nn.ZeroPad2d(padding=kernel_size)
        self.depthwise_conv = DepthwiseConv2d(
            out_dim, kernel_size=kernel_size, strides=stride, padding=conv_pad, bias=False
        )
        self.depthwise_norm = nn.BatchNorm2d(num_features=out_dim)
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Depthwise convolution
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)
        return hidden_states


class EfficientNetSqueezeExciteLayer(nn.Module):
    """This corresponds to the Squeeze and Excitement phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
    """

    def __init__(self, config, dim, kernel_size):
        super().__init__()
        self.dim_se = max(1, int(dim * config.squeeze_expansion_ratio))

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=kernel_size,
            padding="same",
        )
        self.act_reduce = ACT2FN[config.hidden_act]
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = hidden_states.reshape((1, 1, self.dim))

        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        x = torch.mul(input, hidden_states)
        return x


class EfficientNetFinalLayer(nn.Module):
    """This corresponds to the final phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        stride (`int`): Stride size.
        drop_rate (`float`): Dropout rate to be used.
    """

    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float):
        super().__init__()
        self.apply_dropout = stride == 1 and in_dim == out_dim
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_dim)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, inputs: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.batchnorm(hidden_states)

        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + inputs
        return hidden_states


class EfficientNetBlock(nn.Module):
    """This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        stride (`int`): Stride size.
        drop_rate (`float`): Dropout rate to be used.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
    ):
        super().__init__()
        out_dim = in_dim * expand_ratio
        self.expansion = EfficientNetExpansionLayer(config=config, in_dim=in_dim, out_dim=out_dim, stide=stride)
        self.depthwise_conv = EfficientNetDepthwiseLayer(
            config=config,
            in_dim=in_dim,
            out_dim=out_dim,
            stride=stride,
            kernel_size=kernel_size,
        )
        self.squeeze_excite = EfficientNetSqueezeExciteLayer(config=config, in_dim=in_dim, kernel_size=kernel_size)
        self.projection = EfficientNetFinalLayer(
            config=config,
            in_dim=in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expansion and depthwise convolution phase
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # Squeeze and excite phase
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.conv_block(hidden_states)

        return hidden_states


class EfficientNetEncoder(nn.Module):
    """
    Forward propogates the embeddings through each EfficientNet block.

    Args:
        config ([`EfficientNetConfig`]): Model configuration class.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ):
        super().__init__()
        self.config = config
        self.depth_coefficient = config.depth_coefficient
        self.blocks = []

        def round_repeats(repeats):
            # Round number of block repeats based on depth multiplier.
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]

            for j in range(round_repeats(config.num_block_repeats[i])):
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                block = EfficientNetBlock(
                    config=config,
                    in_dim=in_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                )
                blocks.append(block)
                curr_block_num += 1

        self.blocks.append(block)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


class EfficientNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientNetConfig
    base_model_prefix = "efficientnet"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, EfficientNetBlock):
            module.gradient_checkpointing = value


@add_start_docstrings(
    "The bare EfficientNet model outputting raw features without any specific head on top.",
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetModel(EfficientNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = EfficientNetEmbeddings(config)
        self.encoder = EfficientNetEncoder(config)

        # Final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.drop_rate)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # Global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        pooled_output = self.dropout(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet.
    """,
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetForImageClassification(EfficientNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        num_labels = config.num_labels

        self.efficientnet = EfficientNetModel(config)
        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], num_labels) if num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class EfficientNetBackbone(EfficientNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)

        self.stage_names = config.stage_names
        self.embeddings = EfficientNetEmbeddings(config)
        self.encoder = EfficientNetEncoder(config)

        self.out_features = config.out_features if config.out_features is not None else [self.stage_names[-1]]

        out_feature_channels = {}
        out_feature_channels["stem"] = config.hidden_sizes[0]
        for idx, stage in enumerate(self.stage_names[1:]):
            out_feature_channels[stage] = config.hidden_sizes[idx]

        self.out_feature_channels = out_feature_channels

        # initialize weights and apply final processing
        self.post_init()

    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("google/efficientnet-b7")
        >>> model = AutoBackbone.from_pretrained("google/efficientnet-b7")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        feature_maps = ()
        # we skip the stem
        for idx, (stage, hidden_state) in enumerate(zip(self.stage_names[1:], hidden_states[1:])):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
