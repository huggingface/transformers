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
""" PyTorch ConvNext model."""

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    replace_return_docstrings,
    add_start_docstrings_to_model_forward,
)

from ...modeling_outputs import SemanticSegmentationModelOutput, SequenceClassifierOutput

from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_convnext import ConvNextConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextConfig"
_FEAT_EXTRACTOR_FOR_DOC = "ConvNextFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnext-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "'tabby, tabby cat'"

CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnext-tiny-224",
    # See all ConvNext models at https://huggingface.co/models?filter=convnext
]


@dataclass
class ConvNextEncoderOutput(ModelOutput):
    """
    Class for [`ConvNextEncoder`]'s outputs, with potential hidden states (feature maps).

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
class ConvNextModelOutput(ModelOutput):
    """
    Class for [`ConvNextModel`]'s outputs, with potential hidden states (feature maps).

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
class ConvNextClassifierOutput(ModelOutput):
    """
    Class for [`ConvNextForImageClassification`]'s outputs, with potential hidden states (feature maps).

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


# Stochastic depth implementation
# Taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is the same as the
    DropConnect impl I created for EfficientNet, etc networks, however, the original name is misleading as 'Drop
    Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class ConvNextDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvNextLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNextEmbeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")

    def forward(self, pixel_values):
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextLayer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = ACT2FN[config.hidden_act]
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )
        self.drop_path = ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states):
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states):
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states


class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        cur = 0
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[cur],
            )
            self.stages.append(stage)
            cur += config.depths[i]
            prev_chs = out_chs

        if config.apply_norm_after_stage:
            self.layernorm = nn.ModuleList()
            for i in range(config.num_stages):
                norm_size = config.hidden_sizes[i]
                self.layernorm.append(ConvNextLayerNorm(norm_size, eps=1e-6, data_format="channels_first"))

    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        
        if self.config.apply_norm_after_stage:
            for i, layers in enumerate(zip(self.stages, self.layernorm)):
                layer_module, norm = layers
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                hidden_states = layer_module(hidden_states)
                hidden_states = norm(hidden_states)
        else:
            for i, layer_module in enumerate(self.stages):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return ConvNextEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ConvNextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextConfig
    base_model_prefix = "convnext"
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
        if isinstance(module, ConvNextModel):
            module.gradient_checkpointing = value


CONVNEXT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CONVNEXT_INPUTS_DOCSTRING = r"""
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
    "The bare ConvNext model outputting raw features without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class ConvNextModel(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)

        # final layernorm layer
        if not self.config.apply_norm_after_stage:
            self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ConvNextModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(self, pixel_values=None, output_hidden_states=None, return_dict=None):
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

        # global average pooling, (N, C, H, W) -> (N, C)
        if not self.config.apply_norm_after_stage:
            pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        else:
            pooled_output = last_hidden_state
        
        if not return_dict:
            return (last_hidden_state, None) + encoder_outputs[1:]

        return ConvNextModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXT_START_DOCSTRING,
)
class ConvNextForImageClassification(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ConvNextClassifierOutput,
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

        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

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

        return ConvNextClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class ConvNextConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, dilation=1, in_place=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=in_place)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class ConvNextPyramidPoolingModule(nn.ModuleList):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvNextConvModule(self.in_channels, self.channels, kernel_size=1),
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class ConvNextUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config):
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = config.hidden_sizes
        self.channels = config.auxiliary_channels * 2 # e.g. 512
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = ConvNextPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvNextConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvNextConvModule(in_channels, self.channels, kernel_size=1, in_place=False)
            fpn_conv = ConvNextConvModule(self.channels, self.channels, kernel_size=3, padding=1, in_place=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvNextConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states):
        # build laterals
        laterals = []
        for i, conv in enumerate(self.lateral_convs):
            print(f"Pre-lateral shape - {i}", encoder_hidden_states[i].shape)
            op = conv(encoder_hidden_states[i])
            print(f"Post-lateral shape - {i}", op.shape)
            laterals.append(op)
        # laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class ConvNextFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (ConvNextConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config, in_index=2, kernel_size=3, dilation=1):
        super().__init__()
        self.in_channels = config.hidden_sizes[-2]
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvNextConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                ConvNextConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvNextConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states):
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    ConvNext Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    CONVNEXT_START_DOCSTRING,
)
class ConvNextForSemanticSegmentation(ConvNextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convnext = ConvNextModel(config)

        # Semantic segmentation head(s)
        self.decode_head = ConvNextUperHead(config)
        self.auxiliary_head = ConvNextFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, upsampled_logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # compute weighted loss
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmentationModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
        legacy_output=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
        legacy_output (`bool`, *optional*):
            Whether to return the legacy outputs or not (with logits of shape `height / 4 , width / 4`). Will default
            to `self.config.legacy_output`.

            This argument is only present for backward compatibility reasons and will be removed in v5 of Transformers.

        Returns:

        Examples:

        ```python
        >>> from transformers import ConvNextFeatureExtractor, ConvNextForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> feature_extractor = ConvNextFeatureExtractor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
        >>> model = ConvNextForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        legacy_output = legacy_output if legacy_output is not None else self.config.legacy_output
        if not legacy_output:
            warnings.warn(
                "The output of this model has changed in v4.17.0 and the logits now have the same size as the inputs. "
                "You can activate the previous behavior by passing `legacy_output=True` to this call or the "
                "configuration of this model (only until v5, then that argument will be removed).",
                FutureWarning,
            )

        outputs = self.convnext(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[2]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]

        logits = self.decode_head(features)

        upsampled_logits = nn.functional.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.compute_loss(upsampled_logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits if legacy_output else upsampled_logits,) + outputs[2:]
            else:
                output = (logits if legacy_output else upsampled_logits,) + outputs[3:]
            return ((loss,) + output) if loss is not None else output

        if legacy_output:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=None,
            )
        else:
            return SemanticSegmentationModelOutput(
                loss=loss,
                logits=upsampled_logits,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=None,
            )