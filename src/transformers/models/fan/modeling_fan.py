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
""" PyTorch FAN model."""

# Transformers implementation of the following paper: https://arxiv.org/abs/2204.12451
# Based on the following repository https://github.com/NVlabs/FAN

import collections.abc
import math
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2CLS
from ...modeling_outputs import ModelOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_fan import FANConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "ksmcg/fan_base_18_p16_224"
_CONFIG_FOR_DOC = "FANConfig"
_FEAT_EXTRACTOR_FOR_DOC = "FANFeatureExtractor"

FAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ksmcg/fan_tiny_12_p16_224",
    "ksmcg/fan_small_12_p16_224",
    "ksmcg/fan_base_18_p16_224",
    "ksmcg/fan_large_24_p16_224"
    # "nvidia/fan",
    # See all FAN models at https://huggingface.co/models?filter=fan
]


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


@dataclass
class FANModelOutput(ModelOutput):
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
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (ConvNeXt).
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FANSemanticSegmenterOutput(ModelOutput):
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
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (ConvNeXt).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FANImageClassifierOutput(ModelOutput):
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
            Tuple of `torch.FloatTensor` only available when backbone is hybrid (ConvNeXt).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    backbone_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# BELOW: utilities copied from
# https://github.com/NVlabs/FAN/blob/master/models/fan.py
class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all of Need" paper.
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000, rounding_mode=None):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6
        self.rounding_mode = rounding_mode  # Uses Floor for Classifier and None for Segmentation
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=torch.sigmoid,
        divisor=1,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class SEMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
        use_se=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.weight = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()

    def forward(self, x, height, width):
        batch_size, seq_len, num_channels = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # import pdb; pdb.set_trace()
        x = self.drop(self.weight * self.dwconv(x, height, width)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = (
            self.se(x.permute(0, 2, 1).reshape(batch_size, num_channels, height, width))
            .reshape(batch_size, num_channels, seq_len)
            .permute(0, 2, 1)
        )
        return x, height, width


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        linear=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.weight = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, height, width):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.weight * self.dwconv(x, height, width)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, hidden_size=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        #         import pdb; pdb.set_trace()

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, hidden_size // 8, 2),
                act_layer(),
                conv3x3(hidden_size // 8, hidden_size // 4, 2),
                act_layer(),
                conv3x3(hidden_size // 4, hidden_size // 2, 2),
                act_layer(),
                conv3x3(hidden_size // 2, hidden_size, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, hidden_size // 4, 2),
                act_layer(),
                conv3x3(hidden_size // 4, hidden_size // 2, 2),
                act_layer(),
                conv3x3(hidden_size // 2, hidden_size, 2),
            )
        elif patch_size == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, hidden_size // 4, 2),
                act_layer(),
                conv3x3(hidden_size // 4, hidden_size // 1, 2),
            )
        else:
            raise ValueError(f"For convolutional projection, patch size has to be in [8, 16] not {patch_size}")

    def forward(self, x, return_feat=False):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (batch_size, seq_len, num_channels)
        if return_feat:
            return x, (Hp, Wp), None
        return x, (Hp, Wp)


class DWConv(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features,
            in_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_features,
        )
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
        )

    def forward(self, x, height: int, width: int):
        batch_size, seq_len, num_channels = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)

        x = self.conv2(x)
        x = x.reshape(batch_size, num_channels, seq_len).permute(0, 2, 1)
        return x


# Copied from timm.models.layers.drop
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# Copied from timm.models.layers.mlp
class MlpOri(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# Copied from timm.models.cait
class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_attention_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        head_dim = dim // num_attention_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, num_channels = x.shape
        q = (
            self.q(x[:, 0])
            .unsqueeze(1)
            .reshape(batch_size, 1, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(batch_size, 1, num_channels)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        if return_attention:
            return x_cls, attn
        return x_cls


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
        self,
        dim,
        num_attention_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        tokens_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = ClassAttn(
            dim,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpOri(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.weight1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.weight2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.weight1, self.weight2 = 1.0, 1.0
        self.tokens_norm = tokens_norm

    def forward(self, x, return_attention=False):
        x_norm1 = self.norm1(x)
        if return_attention:
            x1, attn = self.attn(x_norm1, return_attention=return_attention)
        else:
            x1 = self.attn(x_norm1)
        x_attn = torch.cat([x1, x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.weight1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.weight2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        if return_attention:
            return x, attn
        return x


class TokenMixing(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # sr_ratio=1,
        linear=False,
        share_atten=False,
        drop_path=0.0,
        emlp=False,
        sharpen_attn=False,
        mlp_hidden_dim=None,
        act_layer=nn.GELU,
        drop=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert (
            dim % num_attention_heads == 0
        ), f"dim {dim} should be divided by num_attention_heads {num_attention_heads}."

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        head_dim = dim // num_attention_heads
        self.scale = qk_scale or head_dim**-0.5

        self.share_atten = share_atten
        self.emlp = emlp

        cha_sr = 1
        self.q = nn.Linear(dim, dim // cha_sr, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 // cha_sr, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear
        # self.sr_ratio = sr_ratio

    def forward(self, x, height, width, atten=None, return_attention=False):
        batch_size, seq_len, num_channels = x.shape
        q = (
            self.q(x)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )

        kv = (
            self.kv(x)
            .reshape(batch_size, -1, 2, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(2, 0, 3, 1, 4)
        )

        k, v = kv[0], kv[1]
        attn = q * self.scale @ k.transpose(-2, -1)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=2,
        feature_size=None,
        in_chans=3,
        hidden_size=384,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (
            feature_size[0] // patch_size[0],
            feature_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, return_feat=False):
        x, out_list = self.backbone(x, return_feat=return_feat)
        batch_size, num_channels, height, width = x.shape
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        if return_feat:
            return x, (height // self.patch_size[0], width // self.patch_size[1]), out_list
        else:
            return x, (height // self.patch_size[0], width // self.patch_size[1])


class ChannelProcessing(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        linear=False,
        drop_path=0.0,
        mlp_hidden_dim=None,
        act_layer=nn.GELU,
        drop=None,
        norm_layer=nn.LayerNorm,
        cha_sr_ratio=1,
        # c_head_num=None,
    ):
        super().__init__()
        assert (
            dim % num_attention_heads == 0
        ), f"dim {dim} should be divided by num_attention_heads {num_attention_heads}."

        self.dim = dim
        # num_attention_heads = c_head_num or num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.temperature = nn.Parameter(torch.ones(num_attention_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_attention_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_v = Mlp(
            in_features=dim // self.cha_sr_ratio,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear,
        )
        self.norm_v = norm_layer(dim // self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1, -2)
        _, _, seq_len, _ = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (seq_len, 1))

        attn = torch.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, height, width, atten=None):
        batch_size, seq_len, num_channels = x.shape
        v = x.reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads).permute(
            0, 2, 1, 3
        )

        q = (
            self.q(x)
            .reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads)
            .permute(0, 2, 1, 3)
        )
        k = x.reshape(batch_size, seq_len, self.num_attention_heads, num_channels // self.num_attention_heads).permute(
            0, 2, 1, 3
        )

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = (
            self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), height, width))
            .reshape(Bv, Nv, Hd, Cv)
            .transpose(1, 2)
        )

        repeat_time = seq_len // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(batch_size, seq_len, num_channels)
        return x, (attn * v.transpose(-1, -2)).transpose(-1, -2)  # attn

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class FANBlock_SE(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        linear=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        # sr_ratio=1.0,
        use_se=False,
        qk_scale=None,
        downsample=None,
        # c_head_num=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            act_layer=act_layer,
            # sr_ratio=sr_ratio,
            linear=linear,
            emlp=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = SEMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.weight1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.weight2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, Hp: int, Wp: int, attn=None):
        x_new, attn_s = self.attn(self.norm1(x), Hp, Wp)
        x = x + self.drop_path(self.weight1 * x_new)
        x_new, Hp, Wp = self.mlp(self.norm2(x), Hp, Wp)
        x = x + self.drop_path(self.weight2 * x_new)
        return x, Hp, Wp, attn_s


class FANBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        # sr_ratio=1.0,
        downsample=None,
        # c_head_num=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_attention_heads=num_attention_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            act_layer=act_layer,
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            # sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ChannelProcessing(
            dim,
            num_attention_heads=num_attention_heads,
            act_layer=act_layer,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop=drop,
            mlp_hidden_dim=int(dim * mlp_ratio),
            # c_head_num=c_head_num,
        )

        self.weight1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.weight2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.downsample = downsample

    def forward(self, x, Hp, Wp, attn=None, return_attention=False):

        x_new, attn_s = self.attn(self.norm1(x), Hp, Wp)
        x = x + self.drop_path(self.weight1 * x_new)

        x_new, attn_c = self.mlp(self.norm2(x), Hp, Wp, atten=attn)
        x = x + self.drop_path(self.weight2 * x_new)
        if return_attention:
            return x, attn_s

        if self.downsample is not None:
            x, Hp, Wp = self.downsample(x, Hp, Wp)
        return x, Hp, Wp, attn_s


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, hidden_size=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.height, self.width = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(
            in_chans,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, height, width):
        batch_size, seq_len, num_channels = x.shape
        x = x.transpose(-1, -2).reshape(batch_size, num_channels, height, width)
        x = self.proj(x)
        _, _, height, width = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, height, width


# ConvNext Utils for Hybrid Backbones
def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie seq_len, num_channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class ConvMlp(nn.Module):
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block
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
        dim,
        drop_path=0.0,
        ls_init_value=1e-6,
        conv_mlp=True,
        mlp_ratio=4,
        norm_layer=None,
    ):
        super().__init__()
        if not norm_layer:
            norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.weight = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Added This initialization to pass initialization Test
        self.weight.data = nn.init.trunc_normal_(
            self.weight.data, std=ls_init_value, a=-2 * ls_init_value, b=2 * ls_init_value
        )

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.weight is not None:
            x = x.mul(self.weight.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        depth=2,
        dp_rates=None,
        ls_init_value=1.0,
        conv_mlp=True,
        norm_layer=None,
        cl_norm_layer=None,
        cross_stage=False,
    ):
        super().__init__()

        if in_chs != out_chs or stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=stride, stride=stride),
            )
        else:
            self.downsample = nn.Identity()

        dp_rates = dp_rates or [0.0] * depth
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=out_chs,
                    drop_path=dp_rates[j],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer if conv_mlp else cl_norm_layer,
                )
                for j in range(depth)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
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

    def __init__(
        self,
        in_chans=3,
        num_labels=1000,
        global_pool="avg",
        output_stride=32,
        patch_size=4,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        ls_init_value=1e-6,
        conv_mlp=True,
        use_head=True,
        head_init_scale=1.0,
        head_norm_first=False,
        norm_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        assert output_stride == 32
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert (
                conv_mlp
            ), "If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input"
            cl_norm_layer = norm_layer

        self.num_labels = num_labels
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer width/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(dims[0]),
        )

        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        curr_stride = patch_size
        prev_chs = dims[0]
        self.stages = nn.ModuleList()
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(len(depths)):
            stride = 2 if i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            out_chs = dims[i]
            self.stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    stride=stride,
                    depth=depths[i],
                    dp_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer,
                    cl_norm_layer=cl_norm_layer,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f"stages.{i}")]
        self.num_features = prev_chs

    def forward(self, x, return_feat=False):
        x = self.stem(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)

        return x, out_list if return_feat else x


class FANPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FANConfig
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
        if isinstance(module, FANEncoder):
            module.gradient_checkpointing = value


FAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~FANConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FAN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`FANImageProcessor`]. See [`FANImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class FANEmbeddings(FANPreTrainedModel):
    def __init__(self, config: FANConfig):
        super().__init__(config)

        img_size = to_2tuple(config.img_size)
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        act_layer = ACT2CLS[config.hidden_act] if config.hidden_act else nn.GELU

        if config.backbone is None:
            self.patch_embeddings = ConvPatchEmbed(
                img_size=img_size,
                patch_size=config.patch_size,
                in_chans=config.num_channels,
                hidden_size=config.hidden_size,
                act_layer=act_layer,
            )
        elif config.backbone == "hybrid":
            backbone = ConvNeXt(
                depths=self.config.depths,
                dims=self.config.dims,
                use_head=False,
                ls_init_value=self.config.initializer_range,
            )
            self.patch_embeddings = HybridEmbed(
                backbone=backbone, patch_size=config.hybrid_patch_size, hidden_size=config.hidden_size
            )
        else:
            raise ValueError(f"{config.backbone} has to be either hybrid or None")
        if config.use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=config.hidden_size, rounding_mode=self.config.rounding_mode)
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
        if isinstance(self.patch_embeddings, HybridEmbed):
            hidden_states, (Hp, Wp), out_list = self.patch_embeddings(pixel_values, return_feat=True)
            if output_hidden_states:
                encoder_states = encoder_states + tuple(out_list)
        else:
            hidden_states, (Hp, Wp) = self.patch_embeddings(pixel_values)

        if self.config.use_pos_embed:
            pos_encoding = (
                self.pos_embed(batch_size, Hp, Wp).reshape(batch_size, -1, hidden_states.shape[1]).permute(0, 2, 1)
            )
            hidden_states = hidden_states + pos_encoding

        hidden_states = self.pos_drop(hidden_states)
        if output_hidden_states:
            return hidden_states, (Hp, Wp), encoder_states

        return hidden_states, (Hp, Wp), encoder_states


class FANEncoderLayer(FANPreTrainedModel):
    def __init__(self, config: FANConfig, index=0):
        super().__init__(config)

        img_size = to_2tuple(config.img_size)
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        num_attention_heads = (
            [config.num_attention_heads] * config.num_hidden_layers
            if not isinstance(config.num_attention_heads, list)
            else config.num_attention_heads
        )
        channel_dims = (
            [config.hidden_size] * config.num_hidden_layers if config.channel_dims is None else config.channel_dims
        )
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        act_layer = ACT2CLS[config.hidden_act] if config.hidden_act else nn.GELU

        downsample = None

        if config.se_mlp:
            build_block = FANBlock_SE
        else:
            build_block = FANBlock
        if index < config.num_hidden_layers - 1 and channel_dims[index] != channel_dims[index + 1]:
            downsample = OverlapPatchEmbed(
                img_size=img_size,
                patch_size=3,
                stride=2,
                in_chans=channel_dims[index],
                hidden_size=channel_dims[index + 1],
            )

        self.block = build_block(
            dim=channel_dims[index],
            num_attention_heads=num_attention_heads[index],
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop=config.hidden_dropout_prob,
            # sr_ratio=config.sr_ratio[index], # Unused
            attn_drop=config.attention_probs_dropout_prob,
            drop_path=config.drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eta=config.eta,
            downsample=downsample,
            # c_head_num=config.c_head_num[index] if config.c_head_num is not None else None,
        )

    def forward(self, hidden_state, Hp, Wp):
        hidden_state, Hp, Wp, attn = self.block(hidden_state, Hp, Wp)
        return hidden_state, Hp, Wp, attn


class FANEncoder(FANPreTrainedModel):
    def __init__(self, config: FANConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        img_size = to_2tuple(config.img_size)
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        num_attention_heads = (
            [config.num_attention_heads] * config.num_hidden_layers
            if not isinstance(config.num_attention_heads, list)
            else config.num_attention_heads
        )
        channel_dims = (
            [config.hidden_size] * config.num_hidden_layers if config.channel_dims is None else config.channel_dims
        )
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        act_layer = ACT2CLS[config.hidden_act] if config.hidden_act else nn.GELU
        self.blocks = nn.ModuleList([FANEncoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.num_features = self.hidden_size = channel_dims[-1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[-1]))
        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=channel_dims[-1],
                    num_attention_heads=num_attention_heads[-1],
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.hidden_dropout_prob,
                    attn_drop=config.attention_probs_dropout_prob,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=config.eta,
                    tokens_norm=config.tokens_norm,
                )
                for _ in range(config.cls_attn_layers)
            ]
        )
        if self.config.backbone == "hybrid" and self.config.feat_downsample:
            self.learnable_downsample = nn.Conv2d(
                in_channels=self.config.hidden_size,
                out_channels=768,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
            )

    def forward(
        self,
        inputs_embeds=None,
        Hp=None,
        Wp=None,
        embedding_hidden_states=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        batch_size = inputs_embeds.shape[0]
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        is_backbone_hybrid = self.config.backbone == "hybrid"

        current_hidden_state = inputs_embeds
        for idx, blk in enumerate(self.blocks):

            if self.gradient_checkpointing:
                current_hidden_state, Hp, Wp, attn = torch.utils.checkpoint.checkpoint(
                    blk, current_hidden_state, Hp, Wp
                )
            else:
                (current_hidden_state, Hp, Wp, attn) = blk(current_hidden_state, Hp, Wp)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

            if output_hidden_states:
                encoder_states = encoder_states + (current_hidden_state,)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        current_hidden_state = torch.cat((cls_tokens, current_hidden_state), dim=1)

        for blk in self.cls_attn_blocks:
            if output_attentions:
                current_hidden_state, attn = blk(current_hidden_state, output_attentions)
            else:
                current_hidden_state = blk(current_hidden_state)

        if output_hidden_states:
            if is_backbone_hybrid and self.config.feat_downsample:
                tmp = current_hidden_state[:, 1:, :].reshape(batch_size, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
                tmp = self.learnable_downsample(tmp)
                tmp = tmp.reshape(batch_size, Hp * Wp, -1).permute(0, 2, 1).contiguous()
                encoder_states + (tmp,)
            else:
                encoder_states = encoder_states + (current_hidden_state[:, 1:, :],)

        if not return_dict:
            return tuple(
                v
                for v in [current_hidden_state, encoder_states, all_attentions, embedding_hidden_states]
                if v is not None
            )
        return FANModelOutput(
            last_hidden_state=current_hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
            backbone_hidden_states=embedding_hidden_states,
        )


@add_start_docstrings(
    "The bare FAN Model transformer outputting raw hidden-states without any specific head on top.",
    FAN_START_DOCSTRING,
)
class FANModel(FANPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = FANEmbeddings(config)
        self.encoder = FANEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=FANModelOutput,
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

        hidden_states, (Hp, Wp), embeddings_encoder_states = self.embeddings(
            pixel_values=pixel_values, output_hidden_states=output_hidden_states
        )
        encoder_outputs = self.encoder(
            hidden_states,
            Hp,
            Wp,
            embeddings_encoder_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return FANModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
            backbone_hidden_states=embeddings_encoder_states if output_hidden_states else None,
        )


class FANClassificationHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the image classes logits

    """

    def __init__(self, num_labels, num_features, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_labels) if num_labels > 0 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)[:, 0]  # Extracts the First Token
        x = self.head(x)
        return x


@add_start_docstrings(
    """
    FAN Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    FAN_START_DOCSTRING,
)
class FANForImageClassification(FANPreTrainedModel):
    def __init__(self, config: FANConfig):
        super().__init__(config)

        # FAN encoder model
        self.fan = FANModel(config)

        num_features = config.hidden_size if config.channel_dims is None else config.channel_dims[-1]
        # Image clasification head
        norm_layer = partial(nn.LayerNorm, eps=config.layer_norm_eps)
        self.head = FANClassificationHead(config.num_labels, num_features, norm_layer)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FANImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values,
        labels: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import FANForImageClassification, FANImageProcessor

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = FANImageProcessor.from_pretrained("ksmcg/fan_base_18_p16_224")
        >>> model = FANForImageClassification.from_pretrained("ksmcg/fan_base_18_p16_224")
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
                v
                for v in [loss, logits, outputs.hidden_states, outputs.attentions, outputs.backbone_hidden_states]
                if v is not None
            )

        return FANImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            backbone_hidden_states=outputs.backbone_hidden_states if output_hidden_states else None,
        )


# Copied from modeling_segformer.py, Since FAN Model uses the segformer head
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: FANConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class FANDecodeHead(FANPreTrainedModel):
    def __init__(self, config: FANConfig):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for in_channels in config.segmentation_in_channels:
            mlp = SegformerMLP(config, input_dim=in_channels)
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
            Hp = self.config.img_size[0] // self.config.patch_size
            Wp = self.config.img_size[1] // self.config.patch_size

            hidden_state_reshaped = hidden_state.reshape(batch_size, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
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
    """FAN Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.""",
    FAN_START_DOCSTRING,
)
class FANForSemanticSegmentation(FANPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fan = FANModel(config)
        self.decode_head = FANDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FANSemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import FANForSemanticSegmentation, FANImageProcessor
        >>> from PIL import Image
        >>> import requests

        >>> feature_extractor = FANImageProcessor.from_pretrained("ksmcg/fan_base_16_p4_hybrid")
        >>> # note: we are loading a FANForSemanticSegmentation from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = FANForSemanticSegmentation.from_pretrained("ksmcg/fan_base_16_p4_hybrid")
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
                v
                for v in [
                    loss,
                    logits,
                    outputs.hidden_states if output_hidden_states else None,
                    outputs.attentions if output_attentions else None,
                    outputs.backbone_hidden_states if output_hidden_states else None,
                ]
                if v is not None
            )

        return FANSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            backbone_hidden_states=outputs.backbone_hidden_states if output_hidden_states else None,
        )
