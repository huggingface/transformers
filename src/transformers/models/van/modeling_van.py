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
""" PyTorch Van model."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
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
from .configuration_van import VanConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "VanConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "van-base"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "van-base"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

VAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "van-base",
    # See all van models at https://huggingface.co/models?filter=van
]


@dataclass
# Copied from transformers.models.convnext.modeling_convnext.ConvNextEncoderOutput with ConvNext->Van
class VanEncoderOutput(ModelOutput):
    """
    Class for [`VanEncoder`]'s outputs, with potential hidden states (feature maps).

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
# Copied from transformers.models.convnext.modeling_convnext.ConvNextModelOutput with ConvNext->Van
class VanModelOutput(ModelOutput):
    """
    Class for [`VanModel`]'s outputs, with potential hidden states (feature maps).

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
# Copied from transformers.models.convnext.modeling_convnext.ConvNextClassifierOutput with ConvNext->Van
class VanClassifierOutput(ModelOutput):
    """
    Class for [`VanForImageClassification`]'s outputs, with potential hidden states (feature maps).

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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Van
class VanDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# Copied from transformers.models.convnext.modeling_convnext.ConvNextEmbeddings with ConvNext->Van
class VanOverlappingPatchEmbedder(nn.Sequential):
    """
    Downsamples the input using a patchfy operation with a `stride` of 4 by default. From [PVTv2: Improved Baselines
    with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.embeddings = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.norm = nn.BatchNorm2d(hidden_size)


class VanMlpLayer(nn.Sequential):
    """
    MLP from [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        hidden_act: str = "gelu",
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.fc2 = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.norm = nn.Dropout(dropout_rate)


class VanLargeKernelAttention(nn.Sequential):
    def __init__(self, hidden_size):
        super().__init__()
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        self.depth_wise_dilated = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=7, dilation=3, padding=9, groups=hidden_size
        )
        self.point_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)


class VanLargeKernelAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = VanLargeKernelAttention(hidden_size)

    def forward(self, hidden_state):
        attention = self.attention(hidden_state)
        attended = hidden_state * attention
        return attended


class VanSpatialAttentionLayer(nn.Module):
    """
    VAN spatial attention layer composed projection (conv) -> act -> LKA attention -> projection (conv) + residual
    connetion.

    """

    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.pre_projection = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(hidden_size, hidden_size, kernel_size=1)),
                    ("act", ACT2FN[hidden_act]),
                ]
            )
        )
        self.attention_layer = VanLargeKernelAttentionLayer(hidden_size)
        self.post_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.pre_projection(hidden_state)
        hidden_state = self.attention_layer(hidden_state)
        hidden_state = self.post_projection(hidden_state)

        return residual + hidden_state


class VanLayerScaling(nn.Module):
    def __init__(self, hidden_size: int, initial_value: float = 1e-2):
        super().__init__()
        self.weight = nn.Parameter(initial_value * torch.ones((hidden_size)), requires_grad=True)

    def forward(self, hidden_state):
        # unsqueeze for broadcasting
        hidden_state = self.weight.unsqueeze(-1).unsqueeze(-1) * hidden_state
        return hidden_state


class VanLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_expansion: int = 4,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.5,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.drop_path = VanDropPath(drop_path) if drop_path_rate > 0.0 else nn.Identity()
        self.pre_norm = nn.BatchNorm2d(hidden_size)
        self.attention = VanSpatialAttentionLayer(hidden_size, hidden_act)
        self.attention_scaling = VanLayerScaling(hidden_size)
        self.post_norm = nn.BatchNorm2d(hidden_size)
        self.mlp = VanMlpLayer(hidden_size, hidden_size * mlp_expansion, hidden_size, hidden_act, dropout_rate)
        self.mlp_scaling = VanLayerScaling(hidden_size)

    def forward(self, hidden_state):
        residual = hidden_state
        # attention
        hidden_state = self.pre_norm(hidden_state)
        hidden_state = self.attention(hidden_state)
        hidden_state = self.attention_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        # residual connection
        hidden_state = residual + hidden_state
        residual = hidden_state
        # mlp
        hidden_state = self.post_norm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.mlp_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        hidden_state = residual + hidden_state
        # residual connection
        hidden_state = residual + hidden_state
        return hidden_state


class VanStage(nn.Sequential):
    """VanStage stage, consisting of an optional downsampling layer + multiple layers.

    Args:
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        stride: int,
        depth: int,
        mlp_expansion: int = 4,
        drop_path_rate: float = 0.0,
        dropout_rate: float = 0.0,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.embeddings = VanOverlappingPatchEmbedder(in_channels, hidden_size, patch_size, stride)
        self.layers = nn.Sequential(
            *[
                VanLayer(
                    hidden_size,
                    mlp_expansion=mlp_expansion,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    hidden_act=hidden_act,
                )
                for _ in range(depth)
            ]
        )


# Copied from transformers.models.convnext.modeling_convnext.ConvNextEncoder with ConvNext->Van
class VanEncoder(nn.Module):
    """_summary_"""

    def __init__(self, config: VanConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        patch_sizes = config.patch_sizes
        strides = config.strides
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        mlp_expansions = config.mlp_expansions
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        for num_stage, (patch_size, stride, hidden_size, depth, mlp_expantion, drop_path_rate) in enumerate(
            zip(patch_sizes, strides, hidden_sizes, depths, mlp_expansions, drop_path_rates)
        ):
            is_first_stage = num_stage == 0
            in_channels = hidden_sizes[num_stage - 1]
            if is_first_stage:
                in_channels = config.num_channels
            self.stages.append(
                VanStage(
                    in_channels,
                    hidden_size,
                    patch_size=patch_size,
                    stride=stride,
                    depth=depth,
                    mlp_expansion=mlp_expantion,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=config.dropout_rate,
                )
            )

    def forward(self, hidden_state, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, stage_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)

        return VanEncoderOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
        )


# Copied from transformers.models.convnext.modeling_convnext.ConvNextPreTrainedModel with ConvNext->Van,convnext->van
class VanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VanConfig
    base_model_prefix = "van"
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
        if isinstance(module, VanModel):
            module.gradient_checkpointing = value


VAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VanConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VAN_INPUTS_DOCSTRING = r"""
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
    "The bare Van model outputting raw features without any specific head on top.",
    VAN_START_DOCSTRING,
)
# Copied from transformers.models.convnext.modeling_convnext.ConvNextModel with CONVNEXT->VAN,ConvNext->Van
class VanModel(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = VanEncoder(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=VanModelOutput,
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
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return VanModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    Van Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    VAN_START_DOCSTRING,
)
# Copied from transformers.models.convnext.modeling_convnext.ConvNextForImageClassification with CONVNEXT->VAN,ConvNext->Van,convnext->van
class VanForImageClassification(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.van = VanModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=VanClassifierOutput,
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

        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

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

        return VanClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
