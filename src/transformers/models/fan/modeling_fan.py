# coding=utf-8
# Copyright 2022 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Fan model."""

# Transformers implementation of the following paper: https://arxiv.org/abs/2204.12451
# Based on the following repository https://github.com/NVlabs/FAN

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2CLS
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_fan import FanConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ksmcg/fan_base_18_p16_224"
_CONFIG_FOR_DOC = "FanConfig"
_FEAT_EXTRACTOR_FOR_DOC = "FanFeatureExtractor"

FAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ksmcg/fan_tiny_12_p16_224",
    "ksmcg/fan_small_12_p16_224",
    "ksmcg/fan_base_18_p16_224",
    "ksmcg/fan_large_24_p16_224"
    # "nvidia/fan",
    # See all Fan models at https://huggingface.co/models?filter=fan
]


@dataclass
class FanModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        backbone_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (FanConvNeXt).
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FanSemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        backbone_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (FanConvNeXt).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FanImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        backbone_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (FanConvNeXt).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class FanIdentityMultiple(nn.Module):
    r"""A placeholder identity operator that can take multiple arguments in the forward pass."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        return args


# BELOW: utilities copied from
# https://github.com/NVlabs/FAN/blob/master/models/fan.py
class FanPositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all of Need" paper.
    """

    def __init__(self, config: FanConfig):
        super().__init__()
        self.temperature = 10_000
        self.hidden_dim = 32
        self.eps = 1e-6
        self.scale = 2 * math.pi
        self.token_projection = nn.Conv2d(self.hidden_dim * 2, config.hidden_size, kernel_size=1)
        self.rounding_mode = config.rounding_mode  # Uses Floor for Classifier and None for Segmentation
        # Segmentation Positional Encoder link https://github.com/NVlabs/FAN/blob/master/segmentation/mmseg/models/backbones/fan.py
        # Classifier Positional Encoder link https://github.com/NVlabs/FAN/blob/master/models/fan.py

    def forward(self, batch_size: int, height: int, width: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, height + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, width)
        x_embed = torch.arange(1, width + 1, dtype=torch.float32, device=device).repeat(1, height, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode=self.rounding_mode) / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(batch_size, 1, 1, 1)  # (batch_size, num_channels, height, width)


class FanSqueezeExcite(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        se_ratio = 0.25
        reduced_channels = int(input_channels * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(input_channels, reduced_channels, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, input_channels, 1, bias=True)

    def forward(self, hidden_state):
        squeeze_excited = self.avg_pool(hidden_state)
        squeeze_excited = self.conv_reduce(squeeze_excited)
        squeeze_excited = self.act1(squeeze_excited)
        squeeze_excited = self.conv_expand(squeeze_excited)
        hidden_state = hidden_state * squeeze_excited.sigmoid()
        return hidden_state


class FanSqueezeExciteMLP(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        in_features = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        hidden_features = int(in_features * config.mlp_ratio)
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = FanDepthwiseConv(config, index)
        self.weight = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = ACT2CLS[config.hidden_act]()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.se = FanSqueezeExcite(in_features)

    def forward(self, hidden_state, height, width):
        batch_size, seq_len, num_channels = hidden_state.shape
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.drop(self.weight * self.dwconv(hidden_state, height, width)) + hidden_state
        hidden_state = self.fc2(hidden_state)
        hidden_state = self.drop(hidden_state)
        hidden_state = (
            self.se(hidden_state.permute(0, 2, 1).reshape(batch_size, num_channels, height, width))
            .reshape(batch_size, num_channels, seq_len)
            .permute(0, 2, 1)
        )
        return hidden_state, height, width


class FanMlp(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        in_features = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        hidden_features = int(in_features * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = FanDepthwiseConv(config, index)
        self.weight = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = ACT2CLS[config.hidden_act]()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.drop(self.weight * self.dwconv(hidden_states, height, width)) + hidden_states
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


class FanConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, config: FanConfig):
        super().__init__()
        act_layer = ACT2CLS[config.hidden_act] if config.hidden_act else nn.GELU
        hidden_size = config.hidden_size
        self.proj = nn.ModuleList()
        if config.patch_size == 16:
            self.proj.append(
                nn.Conv2d(config.num_channels, hidden_size // 8, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 8))
            self.proj.append(act_layer())
            self.proj.append(
                nn.Conv2d(hidden_size // 8, hidden_size // 4, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 4))
            self.proj.append(act_layer())
            self.proj.append(
                nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 2))
            self.proj.append(act_layer())
            self.proj.append(nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1, bias=False))
            self.proj.append(nn.BatchNorm2d(hidden_size))
        elif config.patch_size == 8:
            self.proj.append(
                nn.Conv2d(config.num_channels, hidden_size // 4, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 4))
            self.proj.append(act_layer())
            self.proj.append(
                nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 2))
            self.proj.append(act_layer())
            self.proj.append(nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1, bias=False))
            self.proj.append(nn.BatchNorm2d(hidden_size))
        elif config.patch_size == 4:
            self.proj.append(
                nn.Conv2d(config.num_channels, hidden_size // 2, kernel_size=3, stride=2, padding=1, bias=False)
            )
            self.proj.append(nn.BatchNorm2d(hidden_size // 2))
            self.proj.append(act_layer())
            self.proj.append(nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1, bias=False))
            self.proj.append(nn.BatchNorm2d(hidden_size))
        else:
            raise ValueError(f"For convolutional projection, patch size has to be in [8, 16] not {config.patch_size}")

    def forward(self, pixel_values: torch.Tensor):
        output = pixel_values
        for block in self.proj:
            output = block(output)
        height_patches, width_patches = output.shape[2], output.shape[3]
        output = output.flatten(2).transpose(1, 2)  # (batch_size, seq_len, num_channels)
        return output, (height_patches, width_patches)


class FanDepthwiseConv(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        in_features = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        hidden_features = int(in_features * config.mlp_ratio)
        kernel_size = 3
        padding = kernel_size // 2
        self.conv1 = torch.nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_features,
        )
        self.act = ACT2CLS[config.hidden_act]()
        self.bn = nn.BatchNorm2d(hidden_features)
        self.conv2 = torch.nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_features,
        )

    def forward(self, hidden_states, height: int, width: int):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.bn(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, num_channels, seq_len).permute(0, 2, 1)
        return hidden_states


# Copied from timm.models.layers.drop
class FanDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.

    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, hidden_states):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        shape = (hidden_states.shape[0],) + (1,) * (
            hidden_states.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = hidden_states.new_empty(shape).bernoulli_(self.keep_prob)
        if self.keep_prob > 0.0:
            random_tensor.div_(self.keep_prob)
        return hidden_states * random_tensor


# Copied from timm.models.layers.mlp
class FanMlpOriginal(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size if config.channel_dims is None else config.channel_dims[-1]
        hidden_features = int(hidden_size * config.mlp_ratio) or hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.act = ACT2CLS[config.hidden_act]()
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)
        self.fc2 = nn.Linear(hidden_features, hidden_size)
        self.drop2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop2(hidden_states)
        return hidden_states


# Copied from timm.models.cait
class FanClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size if config.channel_dims is None else config.channel_dims[-1]
        self.num_attention_heads = (
            config.num_attention_heads
            if not isinstance(config.num_attention_heads, list)
            else config.num_attention_heads[-1]
        )
        head_dim = dim // self.num_attention_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.key = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.value = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = (
            self.query(hidden_states[:, 0])
            .unsqueeze(1)
            .reshape(batch_size, 1, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )
        key_layer = (
            self.key(hidden_states)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        query_layer = query_layer * self.scale
        value_layer = (
            self.value(hidden_states)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        attn = query_layer @ key_layer.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        hidden_states_cls = (attn @ value_layer).transpose(1, 2).reshape(batch_size, 1, num_channels)
        hidden_states_cls = self.proj(hidden_states_cls)
        hidden_states_cls = self.proj_drop(hidden_states_cls)

        return hidden_states_cls


class FanClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size if config.channel_dims is None else config.channel_dims[-1]
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attn = FanClassAttn(config)
        self.drop_path = FanDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = FanMlpOriginal(config)

        if config.eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.weight1 = nn.Parameter(config.eta * torch.ones(hidden_size), requires_grad=True)
            self.weight2 = nn.Parameter(config.eta * torch.ones(hidden_size), requires_grad=True)
        else:
            self.weight1, self.weight2 = 1.0, 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        normalized_attn = self.attn(hidden_states_norm)
        hidden_states_attn = torch.cat([normalized_attn, hidden_states_norm[:, 1:]], dim=1)
        hidden_states = hidden_states + self.drop_path(self.weight1 * hidden_states_attn)
        if self.config.tokens_norm:
            hidden_states = self.norm2(hidden_states)
        else:
            hidden_states = torch.cat([self.norm2(hidden_states[:, 0:1]), hidden_states[:, 1:]], dim=1)
        x_res = hidden_states
        cls_token = hidden_states[:, 0:1]
        cls_token = self.weight2 * self.mlp(cls_token)
        hidden_states = torch.cat([cls_token, hidden_states[:, 1:]], dim=1)
        hidden_states = x_res + self.drop_path(hidden_states)
        return hidden_states


class FanTokenMixing(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        dim = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        num_attention_heads = (
            config.num_attention_heads
            if not isinstance(config.num_attention_heads, list)
            else config.num_attention_heads[index]
        )

        assert (
            dim % num_attention_heads == 0
        ), f"dim {dim} should be divided by num_attention_heads {num_attention_heads}."

        self.num_attention_heads = num_attention_heads
        head_dim = dim // num_attention_heads
        self.scale = head_dim**-0.5
        self.query = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, num_channels = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        key_value_layer = (
            self.key_value(hidden_states)
            .reshape(batch_size, -1, 2, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(2, 0, 3, 1, 4)
        )

        key_layer, value_layer = key_value_layer[0], key_value_layer[1]
        attn = query_layer * self.scale @ key_layer.transpose(-2, -1)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        hidden_states = (attn @ value_layer).transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.proj_drop(hidden_states)
        return hidden_states, attn


class FanHybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, config: FanConfig):
        super().__init__()
        backbone = FanConvNeXt(config)
        patch_size = config.hybrid_patch_size
        hidden_size = config.hidden_size
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.backbone = backbone
        feature_dim = config.hybrid_in_channels[len(config.depths) - 1]
        downsample = (
            4 * 2 * (len(config.depths) - 1)
        )  # Stem has stride 4, First layer has stride 1, remaining have stride 2
        feature_size = [config.img_size[0] // downsample, config.img_size[1] // downsample]
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor):
        output, out_list = self.backbone(pixel_values)
        batch_size, num_channels, height, width = output.shape
        output = self.proj(output).flatten(2).transpose(1, 2)
        return output, (height // self.patch_size[0], width // self.patch_size[1]), out_list


class FanChannelProcessing(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        dim = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        num_attention_heads = (
            config.num_attention_heads
            if not isinstance(config.num_attention_heads, list)
            else config.num_attention_heads[index]
        )
        assert (
            dim % num_attention_heads == 0
        ), f"dim {dim} should be divided by num_attention_heads {num_attention_heads}."

        self.dim = dim
        # num_attention_heads = c_head_num or num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.temperature = nn.Parameter(torch.ones(num_attention_heads, 1, 1))

        # config of mlp for value_layer processing
        self.drop_path = FanDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.mlp_v = FanMlp(config=config, index=index)
        self.norm_v = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.query = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        value_layer = hidden_states.reshape(
            batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads
        ).permute(0, 2, 1, 3)

        query_layer = (
            self.query(hidden_states)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )
        key_layer = hidden_states.reshape(
            batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads
        ).permute(0, 2, 1, 3)

        query_layer = query_layer.softmax(-2).transpose(-1, -2)
        _, _, seq_len, _ = key_layer.shape
        key_layer = torch.nn.functional.adaptive_avg_pool2d(key_layer.softmax(-2), (seq_len, 1))

        attn = (query_layer @ key_layer).sigmoid()
        attn = attn * self.temperature
        attn = self.attn_drop(attn)

        value_layer = value_layer.transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        value_layer = self.mlp_v(value_layer, height, width)
        value_layer = self.norm_v(value_layer)
        value_layer = value_layer.reshape(
            batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads
        ).transpose(1, 2)

        repeat_time = seq_len // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        hidden_states = (
            (attn * value_layer.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_channels)
        )
        return hidden_states, (attn * value_layer.transpose(-1, -2)).transpose(-1, -2)  # attn


class FanBlock_SE(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        dim = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = FanTokenMixing(config, index)
        self.drop_path = FanDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = FanSqueezeExciteMLP(config, index)

        self.weight1 = nn.Parameter(config.eta * torch.ones(dim), requires_grad=True)
        self.weight2 = nn.Parameter(config.eta * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_state, height_patches: int, width_patches: int):
        hidden_state_new, attn_s = self.attn(self.norm1(hidden_state))
        hidden_state = hidden_state + self.drop_path(self.weight1 * hidden_state_new)
        hidden_state_new, height_patches, width_patches = self.mlp(
            self.norm2(hidden_state), height_patches, width_patches
        )
        hidden_state = hidden_state + self.drop_path(self.weight2 * hidden_state_new)
        return hidden_state, height_patches, width_patches, attn_s


class FanBlock(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()
        dim = config.hidden_size if config.channel_dims is None else config.channel_dims[index]
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = FanTokenMixing(config, index)
        self.drop_path = FanDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = FanChannelProcessing(config, index)
        self.weight1 = nn.Parameter(config.eta * torch.ones(dim), requires_grad=True)
        self.weight2 = nn.Parameter(config.eta * torch.ones(dim), requires_grad=True)
        create_downsample = (config.channel_dims is not None) and (index < config.num_hidden_layers - 1)
        create_downsample = create_downsample and config.channel_dims[index] != config.channel_dims[index + 1]
        self.downsample = FanOverlapPatchEmbed(config, index) if create_downsample else FanIdentityMultiple()

    def forward(self, hidden_state, height_patches, width_patches):

        hidden_state_new, attn_s = self.attn(self.norm1(hidden_state))
        hidden_state = hidden_state + self.drop_path(self.weight1 * hidden_state_new)

        hidden_state_new, attn_c = self.mlp(self.norm2(hidden_state), height_patches, width_patches)
        hidden_state = hidden_state + self.drop_path(self.weight2 * hidden_state_new)

        hidden_state, height_patches, width_patches = self.downsample(hidden_state, height_patches, width_patches)
        return hidden_state, height_patches, width_patches, attn_s


class FanOverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config: FanConfig, index: int):
        super().__init__()

        img_size = config.img_size
        patch_size = (3, 3)
        self.height, self.width = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(
            config.channel_dims[index],
            config.channel_dims[index + 1],
            kernel_size=patch_size,
            stride=2,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(config.channel_dims[index + 1])

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, num_channels, height, width)
        hidden_states = self.proj(hidden_states)
        _, _, height, width = hidden_states.shape

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.norm(hidden_states)

        return hidden_states, height, width


class FanLayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie seq_len, num_channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return F.layer_norm(
                input_tensor.permute(0, 2, 3, 1),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).permute(0, 3, 1, 2)
        else:
            std, mean = torch.var_mean(input_tensor, dim=1, keepdim=True)
            normalized_tensor = (input_tensor - mean) * torch.rsqrt(std + self.eps)
            normalized_tensor = normalized_tensor * self.weight[:, None, None] + self.bias[:, None, None]
            return normalized_tensor


class FanConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FanConvNeXtBlock(nn.Module):
    """FanConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (seq_len, num_channels, height,
      width) (2) DwConv -> Permute to (seq_len, height, width, num_channels); LayerNorm (channels_last) -> Linear ->
      GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate choice of
    LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear is a better choice.
    This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & width/ different HW.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        config: FanConfig,
        stage_index: int,
        block_index: int,
    ):
        super().__init__()
        mlp_ratio = 4
        dim = config.hybrid_in_channels[stage_index]
        droppath_rate = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ][stage_index][block_index]
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = FanLayerNorm2d(dim, eps=config.layer_norm_eps)
        self.mlp = FanConvMlp(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.weight = (
            nn.Parameter(config.initializer_range * torch.ones(dim)) if config.initializer_range > 0 else None
        )
        self.drop_path = FanDropPath(droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        # Added This initialization to pass initialization Test
        self.weight.data = nn.init.trunc_normal_(
            self.weight.data,
            std=config.initializer_range,
            a=-2 * config.initializer_range,
            b=2 * config.initializer_range,
        )

    def forward(self, hidden_states):
        shortcut = hidden_states
        hidden_states = self.conv_dw(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.weight is not None:
            hidden_states = hidden_states.mul(self.weight.reshape(1, -1, 1, 1))
        hidden_states = self.drop_path(hidden_states) + shortcut
        return hidden_states


class FanConvNeXtStage(nn.Module):
    def __init__(self, config: FanConfig, index: int):
        super().__init__()

        in_channels = config.hybrid_in_channels[0] if index == 0 else config.hybrid_in_channels[index - 1]
        out_channels = config.hybrid_in_channels[index]
        stride = 2 if index > 0 else 1
        depth = config.depths[index]
        do_downsample = in_channels != out_channels or stride > 1
        self.downsample = nn.ModuleList()

        if do_downsample:
            self.downsample.append(FanLayerNorm2d(in_channels, eps=config.layer_norm_eps))
            self.downsample.append(nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride))
        else:
            self.downsample.append(nn.Identity())
        self.blocks = nn.ModuleList()
        for j in range(depth):
            self.blocks.append(FanConvNeXtBlock(config, stage_index=index, block_index=j))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.downsample:
            hidden_states = layer(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class FanConvNeXt(nn.Module):
    r"""FanConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_labels (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, config: FanConfig):
        super().__init__()
        patch_size = 4
        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer width/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(config.num_channels, config.hybrid_in_channels[0], kernel_size=patch_size, stride=patch_size),
            FanLayerNorm2d(config.hybrid_in_channels[0], eps=config.layer_norm_eps),
        )

        self.stages = nn.ModuleList()
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for index in range(len(config.depths)):
            self.stages.append(FanConvNeXtStage(config, index))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.stem(hidden_states)
        intermediate_states = []
        for stage in self.stages:
            hidden_states = stage(hidden_states)
            intermediate_states.append(hidden_states)

        return hidden_states, intermediate_states


class FanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FanConfig
    base_model_prefix = "fan"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Added Lower and Upper bound to pass initialization test
            lower, upper = -2 * self.config.initializer_range, 2 * self.config.initializer_range
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=self.config.initializer_range, a=lower, b=upper
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FanEncoder):
            module.gradient_checkpointing = value


FAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~FanConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FAN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`FanImageProcessor`]. See [`FanImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FanEmbeddings(nn.Module):
    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        img_size = (
            config.img_size
            if isinstance(config.img_size, collections.abc.Iterable)
            else (config.img_size, config.img_size)
        )
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[1] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        if config.backbone is None:
            self.patch_embeddings = FanConvPatchEmbed(config)
        elif config.backbone == "hybrid":
            self.patch_embeddings = FanHybridEmbed(config)
        else:
            raise ValueError(f"{config.backbone} has to be either hybrid or None")
        if config.use_pos_embed:
            self.pos_embed = FanPositionalEncodingFourier(config)
        self.pos_drop = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values=None,
        output_hidden_states=None,
    ):
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                input to the layer of shape `(batch, height, width, input_channels(3))`

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        batch_size = pixel_values.shape[0]
        encoder_states = () if output_hidden_states else None
        if isinstance(self.patch_embeddings, FanHybridEmbed):
            hidden_states, (height_patches, width_patches), intermediate_states = self.patch_embeddings(pixel_values)
            if output_hidden_states:
                encoder_states = encoder_states + tuple(intermediate_states)
        else:
            hidden_states, (height_patches, width_patches) = self.patch_embeddings(pixel_values)

        if self.config.use_pos_embed:
            pos_encoding = (
                self.pos_embed(batch_size, height_patches, width_patches)
                .reshape(batch_size, -1, hidden_states.shape[1])
                .permute(0, 2, 1)
            )
            hidden_states = hidden_states + pos_encoding

        hidden_states = self.pos_drop(hidden_states)
        if output_hidden_states:
            return hidden_states, (height_patches, width_patches), encoder_states

        return hidden_states, (height_patches, width_patches), encoder_states


class FanEncoderLayer(nn.Module):
    def __init__(self, config: FanConfig, index=0):
        super().__init__()
        self.config = config

        img_size = (
            config.img_size
            if isinstance(config.img_size, collections.abc.Iterable)
            else (config.img_size, config.img_size)
        )
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        if config.se_mlp:
            self.block = FanBlock_SE(config=config, index=index)
        else:
            self.block = FanBlock(config=config, index=index)

    def forward(self, hidden_state, height_patches, width_patches):
        hidden_state, height_patches, width_patches, attn = self.block(hidden_state, height_patches, width_patches)
        return hidden_state, height_patches, width_patches, attn


class FanEncoder(nn.Module):
    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        img_size = (
            config.img_size
            if isinstance(config.img_size, collections.abc.Iterable)
            else (config.img_size, config.img_size)
        )
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        channel_dims = (
            [config.hidden_size] * config.num_hidden_layers if config.channel_dims is None else config.channel_dims
        )
        self.blocks = nn.ModuleList([FanEncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[-1]))
        self.cls_attn_blocks = nn.ModuleList([FanClassAttentionBlock(config) for _ in range(config.cls_attn_layers)])

    def forward(
        self,
        inputs_embeds=None,
        height_patches=None,
        width_patches=None,
        embedding_hidden_states=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        batch_size = inputs_embeds.shape[0]
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        current_hidden_state = inputs_embeds
        for blk in self.blocks:

            if self.gradient_checkpointing:
                current_hidden_state, height_patches, width_patches, attn = torch.utils.checkpoint.checkpoint(
                    blk, current_hidden_state, height_patches, width_patches
                )
            else:
                (current_hidden_state, height_patches, width_patches, attn) = blk(
                    current_hidden_state, height_patches, width_patches
                )

            if output_attentions:
                all_attentions = all_attentions + (attn,)

            if output_hidden_states:
                encoder_states = encoder_states + (current_hidden_state,)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        current_hidden_state = torch.cat((cls_tokens, current_hidden_state), dim=1)

        for blk in self.cls_attn_blocks:
            current_hidden_state = blk(current_hidden_state)

        if output_hidden_states:
            encoder_states = encoder_states + (current_hidden_state[:, 1:, :],)

        if not return_dict:
            return tuple(
                elem
                for elem in [current_hidden_state, encoder_states, all_attentions, embedding_hidden_states]
                if elem is not None
            )
        return FanModelOutput(
            last_hidden_state=current_hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
            backbone_hidden_states=embedding_hidden_states,
        )


@add_start_docstrings(
    "The bare Fan Model transformer outputting raw hidden-states without any specific head on top.",
    FAN_START_DOCSTRING,
)
class FanModel(FanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = FanEmbeddings(config)
        self.encoder = FanEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=FanModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape

        # Prepare head mask if needed
        # First, sent pixel_values through Backbone to obtain the features if needed
        # pixel_values should be of shape (batch_size, num_channels, height, width)

        hidden_states, (height_patches, width_patches), embeddings_encoder_states = self.embeddings(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )
        encoder_outputs = self.encoder(
            hidden_states,
            height_patches,
            width_patches,
            embeddings_encoder_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return FanModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
            backbone_hidden_states=embeddings_encoder_states if output_hidden_states else None,
        )


class FanClassificationHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the image classes logits

    """

    def __init__(self, config: FanConfig):
        super().__init__()
        num_features = config.hidden_size if config.channel_dims is None else config.channel_dims[-1]
        self.norm = nn.LayerNorm(num_features, eps=config.layer_norm_eps)
        self.head = nn.Linear(num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, last_hidden_state):
        pooled_output = self.norm(last_hidden_state)[:, 0]  # Extracts the First Token
        output = self.head(pooled_output)
        return output


@add_start_docstrings(
    """
    Fan Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    FAN_START_DOCSTRING,
)
class FanForImageClassification(FanPreTrainedModel):
    def __init__(self, config: FanConfig):
        super().__init__(config)

        # Fan encoder model
        self.fan = FanModel(config)
        # Image clasification head
        self.head = FanClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FanImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values,
        labels: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ) -> Union[Tuple, FanImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import FanForImageClassification, FanImageProcessor

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = FanImageProcessor.from_pretrained("ksmcg/fan_base_18_p16_224")
        >>> model = FanForImageClassification.from_pretrained("ksmcg/fan_base_18_p16_224")
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: tabby, tabby cat
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.fan(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return tuple(
                elem
                for elem in [loss, logits, outputs.hidden_states, outputs.attentions, outputs.backbone_hidden_states]
                if elem is not None
            )

        return FanImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            backbone_hidden_states=outputs.backbone_hidden_states if output_hidden_states else None,
        )


# Copied from modeling_segformer.py, Since Fan Model uses the segformer head
class FanSegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: FanConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class FanDecodeHead(nn.Module):
    def __init__(self, config: FanConfig):
        super().__init__()
        self.config = config
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for in_channels in config.segmentation_in_channels:
            mlp = FanSegformerMLP(config, input_dim=in_channels)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * len(config.segmentation_in_channels),
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.decoder_dropout)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states, backbone_hidden_states=None):
        batch_size = encoder_hidden_states[-1].shape[0]
        is_backbone_hybrid = self.config.backbone == "hybrid"

        def reshape_hidden_state(hidden_state):
            height_patches = self.config.img_size[0] // self.config.patch_size
            width_patches = self.config.img_size[1] // self.config.patch_size

            hidden_state_reshaped = (
                hidden_state.reshape(batch_size, height_patches, width_patches, -1).permute(0, 3, 1, 2).contiguous()
            )
            return hidden_state_reshaped

        out_index = [4, 7, 11]
        # TODO: Upsample first 2 states to match expected output
        if is_backbone_hybrid:
            encoder_states = backbone_hidden_states + (
                reshape_hidden_state(encoder_hidden_states[self.config.out_index]),
                reshape_hidden_state(encoder_hidden_states[-1]),
            )
        else:
            encoder_states = tuple(reshape_hidden_state(encoder_hidden_states[idx]) for idx in out_index) + (
                reshape_hidden_state(encoder_hidden_states[-1]),
            )

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_states, self.linear_c):
            if encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


@add_start_docstrings(
    """Fan Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.""",
    FAN_START_DOCSTRING,
)
class FanForSemanticSegmentation(FanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fan = FanModel(config)
        self.decode_head = FanDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FanSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, FanSemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import FanForSemanticSegmentation, FanImageProcessor
        >>> from PIL import Image
        >>> import requests

        >>> feature_extractor = FanImageProcessor.from_pretrained("ksmcg/fan_base_16_p4_hybrid")
        >>> # note: we are loading a FanForSemanticSegmentation from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = FanForSemanticSegmentation.from_pretrained("ksmcg/fan_base_16_p4_hybrid")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 1000, 56, 56]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.fan(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        backbone_hidden_states = outputs.backbone_hidden_states

        logits = self.decode_head(encoder_hidden_states, backbone_hidden_states)

        loss = None
        if labels is not None:
            if not self.config.num_labels > 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        if not return_dict:
            return tuple(
                elem
                for elem in [
                    loss,
                    logits,
                    outputs.hidden_states if output_hidden_states else None,
                    outputs.attentions if output_attentions else None,
                    outputs.backbone_hidden_states if output_hidden_states else None,
                ]
                if elem is not None
            )

        return FanSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            backbone_hidden_states=outputs.backbone_hidden_states if output_hidden_states else None,
        )
