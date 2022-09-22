# coding=utf-8
# Copyright 2022 kiansierra90@gmail.com The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FAN model. """


import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    is_torch_greater_than_1_6,
    prune_linear_layer,
)
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    logging,
    replace_return_docstrings,
)
from .configuration_fan import FANConfig


if is_timm_available():
    from timm import create_model
    from timm.models.cait import ClassAttn
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    from timm.models.registry import register_model
    from timm.models.vision_transformer import Mlp as MlpOri


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "nvidia/fan"
_CONFIG_FOR_DOC = "FANConfig"
_TOKENIZER_FOR_DOC = "FANTokenizer"

FAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/fan",
    # See all FAN models at https://huggingface.co/models?filter=fan
]

# BELOW: utilities copied from
# https://github.com/NVlabs/FAN/blob/master/models/fan.py


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all of Need" paper.
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()


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
        gate_fn=sigmoid,
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
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # import pdb; pdb.set_trace()
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = self.se(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, N).permute(0, 2, 1)
        return x, H, W


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
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        #         import pdb; pdb.set_trace()

        if patch_size == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 1, 2),
                # act_layer(),
                # conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise ("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
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

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)

        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttentionBlock(nn.Module):
    """Class Attention Layer as in CaiT https://arxiv.org/abs/2103.17239"""

    def __init__(
        self,
        dim,
        num_heads,
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
            num_heads=num_heads,
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
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0
        self.tokens_norm = tokens_norm

    def forward(self, x, return_attention=False):
        x_norm1 = self.norm1(x)
        if return_attention:
            x1, attn = self.attn(x_norm1, use_attn=return_attention)
        else:
            x1 = self.attn(x_norm1)
        x_attn = torch.cat([x1, x_norm1[:, 1:]], dim=1)
        x = x + self.drop_path(self.gamma1 * x_attn)
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = torch.cat([self.norm2(x[:, 0:1]), x[:, 1:]], dim=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        if return_attention:
            return attn
        return x


class TokenMixing(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
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
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
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
        self.sr_ratio = sr_ratio
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, atten=None, return_attention=False):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # import pdb;pdb.set_trace()
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = q * self.scale @ k.transpose(-2, -1)  # * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn @ v


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
        embed_dim=384,
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
                o = self.backbone.forward_features(torch.zeros(1, in_chans, img_size[0], img_size[1]))
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
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        B, C, H, W = x.shape
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x, (H // self.patch_size[0], W // self.patch_size[1])


class ChannelProcessing(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        linear=False,
        drop_path=0.0,
        mlp_hidden_dim=None,
        act_layer=nn.GELU,
        drop=None,
        norm_layer=nn.LayerNorm,
        cha_sr_ratio=1,
        c_head_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1, -2)
        _, _, N, _ = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))

        attn = torch.nn.functional.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        Bv, Hd, Nv, Cv = v.shape
        v = (
            self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W))
            .reshape(Bv, Nv, Hd, Cv)
            .transpose(1, 2)
        )

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x, (attn * v.transpose(-1, -2)).transpose(-1, -2)  # attn

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature"}


class FANBlock_SE(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        use_se=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        sr_ratio=1.0,
        qk_scale=None,
        linear=False,
        downsample=None,
        c_head_num=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            sr_ratio=sr_ratio,
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

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H: int, W: int, attn=None):
        x_new, _ = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)
        x_new, H, W = self.mlp(self.norm2(x), H, W)
        x = x + self.drop_path(self.gamma2 * x_new)
        return x, H, W


class FANBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        sharpen_attn=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eta=1.0,
        sr_ratio=1.0,
        downsample=None,
        c_head_num=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenMixing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_hidden_dim=int(dim * mlp_ratio),
            sharpen_attn=sharpen_attn,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop=drop,
            drop_path=drop_path,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = ChannelProcessing(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop=drop,
            mlp_hidden_dim=int(dim * mlp_ratio),
            c_head_num=c_head_num,
        )

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

        self.downsample = downsample
        self.H = None
        self.W = None

    def forward(self, x, attn=None, return_attention=False):
        H, W = self.H, self.W

        x_new, attn_s = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.gamma1 * x_new)

        x_new, attn_c = self.mlp(self.norm2(x), H, W, atten=attn)
        x = x + self.drop_path(self.gamma2 * x_new)
        if return_attention:
            return x, attn_s

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        self.H, self.W = H, W
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-1, -2).reshape(B, C, H, W)
        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class FAN(nn.Module):
    """
    Based on timm code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        sharpen_attn=False,
        channel_dims=None,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        sr_ratio=None,
        backbone=None,
        use_checkpoint=False,
        act_layer=None,
        norm_layer=None,
        se_mlp=False,
        cls_attn_layers=2,
        use_pos_embed=True,
        eta=1.0,
        tokens_norm=False,
        c_head_num=None,
        hybrid_patch_size=2,
        head_init_scale=1.0,
    ):

        super().__init__()
        img_size = to_2tuple(img_size)
        self.use_checkpoint = use_checkpoint
        assert (img_size[0] % patch_size == 0) and (
            img_size[0] % patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        self.num_classes = num_classes
        num_heads = [num_heads] * depth if not isinstance(num_heads, list) else num_heads

        channel_dims = [embed_dim] * depth if channel_dims is None else channel_dims
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        if backbone == None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                act_layer=act_layer,
            )
        else:
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=hybrid_patch_size, embed_dim=embed_dim)

        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if se_mlp:
            build_block = FANBlock_SE
        else:
            build_block = FANBlock
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i < depth - 1 and channel_dims[i] != channel_dims[i + 1]:
                downsample = OverlapPatchEmbed(
                    img_size=img_size,
                    patch_size=3,
                    stride=2,
                    in_chans=channel_dims[i],
                    embed_dim=channel_dims[i + 1],
                )
            else:
                downsample = None
            self.blocks.append(
                build_block(
                    dim=channel_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    sr_ratio=sr_ratio[i],
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=eta,
                    downsample=downsample,
                    c_head_num=c_head_num[i] if c_head_num is not None else None,
                )
            )
        self.num_features = self.embed_dim = channel_dims[i]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[i]))
        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=channel_dims[-1],
                    num_heads=num_heads[-1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=eta,
                    tokens_norm=tokens_norm,
                )
                for _ in range(cls_attn_layers)
            ]
        )

        # Classifier head
        self.norm = norm_layer(channel_dims[i])
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Init weights
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}  # , 'patch_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)
        H, W = Hp, Wp
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            H, W = blk.H, blk.W

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_last_selfattention(self, x, use_cls_attn=False, layer_idx=11):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.pos_drop(x)

        return_idx = layer_idx or len(self.blocks) - 1

        for i, blk in enumerate(self.blocks):
            if i == return_idx:
                x, attn = blk(x, Hp, Wp, return_attention=True)
            else:
                x, Hp, Wp = blk(x, Hp, Wp)

        if use_cls_attn:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for i, blk in enumerate(self.cls_attn_blocks):
                if i < len(self.cls_attn_blocks) - 1:
                    x = blk(x)
                else:
                    attn = blk(x, return_attention=True)
                    return attn
        else:
            return attn


class FANPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = FANConfig
    base_model_prefix = "fan"  # TODO: FAN or model?
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FANEncoder):
            module.gradient_checkpointing = value


# TODO: Update FAN Start Docstring
FAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~FANConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# TODO: Update FAN Inputs Docstring
FAN_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`FANTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
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
        self.use_checkpoint = config.use_checkpoint
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        act_layer = config.act_layer or nn.GELU

        if config.backbone == None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
                act_layer=act_layer,
            )
        else:
            self.patch_embed = HybridEmbed(
                backbone=config.backbone, patch_size=config.hybrid_patch_size, embed_dim=config.embed_dim
            )

        if config.use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=config.embed_dim)
        self.pos_drop = nn.Dropout(p=config.drop_rate)

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        B = pixel_values.shape[0]
        inputs_embeds, (Hp, Wp) = self.patch_embed(pixel_values)
        hidden_states = inputs_embeds

        if self.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, hidden_states.shape[1]).permute(0, 2, 1)
            hidden_states = hidden_states + pos_encoding

        hidden_states = self.pos_drop(hidden_states)
        H, W = Hp, Wp
        return hidden_states


class FANEncoder(FANPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - position_embeddings are added to the forward pass.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: FANConfig):
        super().__init__(config)

        img_size = to_2tuple(config.img_size)
        self.use_checkpoint = config.use_checkpoint
        assert (img_size[0] % config.patch_size == 0) and (
            img_size[0] % config.patch_size == 0
        ), "`patch_size` should divide image dimensions evenly"

        num_heads = [config.num_heads] * config.depth if not isinstance(config.num_heads, list) else config.num_heads
        channel_dims = [config.embed_dim] * config.depth if config.channel_dims is None else config.channel_dims
        norm_layer = config.norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = config.act_layer or nn.GELU

        if config.backbone == None:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
                act_layer=act_layer,
            )
        else:
            self.patch_embed = HybridEmbed(
                backbone=config.backbone, patch_size=config.hybrid_patch_size, embed_dim=config.embed_dim
            )

        if config.use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=config.embed_dim)
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        if config.se_mlp:
            build_block = FANBlock_SE
        else:
            build_block = FANBlock
        self.blocks = nn.ModuleList([])
        for i in range(config.depth):
            if i < config.depth - 1 and channel_dims[i] != channel_dims[i + 1]:
                downsample = OverlapPatchEmbed(
                    img_size=img_size,
                    patch_size=3,
                    stride=2,
                    in_chans=channel_dims[i],
                    embed_dim=channel_dims[i + 1],
                )
            else:
                downsample = None
            self.blocks.append(
                build_block(
                    dim=channel_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.drop_rate,
                    sr_ratio=config.sr_ratio[i],
                    attn_drop=config.attn_drop_rate,
                    drop_path=config.drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=config.eta,
                    downsample=downsample,
                    c_head_num=config.c_head_num[i] if config.c_head_num is not None else None,
                )
            )
        self.num_features = self.embed_dim = channel_dims[i]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel_dims[i]))
        self.cls_attn_blocks = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim=channel_dims[-1],
                    num_heads=num_heads[-1],
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    eta=config.eta,
                    tokens_norm=config.tokens_norm,
                )
                for _ in range(config.cls_attn_layers)
            ]
        )

        # in the original DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        B = inputs_embeds.shape[0]
        inputs_embeds, (Hp, Wp) = self.patch_embed(inputs_embeds)
        hidden_states = inputs_embeds

        if self.config.use_pos_embed:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, hidden_states.shape[1]).permute(0, 2, 1)
            hidden_states = hidden_states + pos_encoding

        hidden_states = self.pos_drop(hidden_states)
        H, W = Hp, Wp

        for blk in self.blocks:
            blk.H, blk.W = H, W

            if self.use_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(blk, hidden_states)
            else:
                hidden_states = blk(hidden_states)
            H, W = blk.H, blk.W

        cls_tokens = self.cls_token.expand(B, -1, -1)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for blk in self.cls_attn_blocks:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states = blk(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


@add_start_docstrings(
    "The bare FAN Model transformer outputting raw hidden-states without any specific head on top.",
    FAN_START_DOCSTRING,
)
class FANModel(FANPreTrainedModel):
    # TODO: Update  Docstring
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # self.embeddings = FANEmbeddings(config)
        self.encoder = FANEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape `(batch_size, 1)`
            instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up
            decoding (see `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        # Prepare head mask if needed

        # embedding_output = self.embeddings(pixel_values=pixel_values)
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FANClassificationHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, num_classes, num_features, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(num_features)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.norm(x)[:, 0]
        x = self.head(x)
        return x


# TODO: Update Image Classification Docstring
@add_start_docstrings(
    """
    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    FAN_START_DOCSTRING,
)
class FANForImageClassification(FANPreTrainedModel):
    def __init__(self, config: FANConfig):
        super().__init__(config)

        # DETR encoder-decoder model
        self.model = FANModel(config)

        num_features = config.embed_dim if config.channel_dims is None else config.channel_dims[-1]
        # Object detection heads
        self.head = FANClassificationHead(config.num_classes, num_features, config.norm_layer)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FAN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        labels: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # TODO: Update Docstring appropiately
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import DeiTFeatureExtractor, DeiTForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a DeiTForImageClassificationWithTeacher from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        >>> model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: maillot
        ```"""

        outputs = self.model(
            pixel_values, pixel_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None
        )
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
