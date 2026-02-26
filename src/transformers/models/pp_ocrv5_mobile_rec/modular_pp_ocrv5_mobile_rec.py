# Copyright 2025 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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


import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    ChannelDimension,
    ImageInput,
)
from ...modeling_utils import PreTrainedModel, logging
from ...utils import ModelOutput, auto_docstring
from ...utils.generic import TensorType


logger = logging.get_logger(__name__)


class PPOCRV5MobileRecLearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), scale_value, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((1,), bias_value, dtype=torch.float32))

    def forward(self, x):
        return self.scale * x + self.bias


class PPOCRV5MobileRecConvBNLayer_PPLCNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, lr_mult=1.0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PPOCRV5MobileRecAct(nn.Module):
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = PPOCRV5MobileRecLearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))


class PPOCRV5MobileRecLearnableRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        num_conv_branches=1,
        lr_mult=1.0,
        lab_lr=0.1,
    ):
        super().__init__()

        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

        self.conv_kxk = nn.ModuleList(
            [
                PPOCRV5MobileRecConvBNLayer_PPLCNet(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    lr_mult=lr_mult,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            PPOCRV5MobileRecConvBNLayer_PPLCNet(in_channels, out_channels, 1, stride, groups=groups, lr_mult=lr_mult)
            if kernel_size > 1
            else None
        )

        self.lab = PPOCRV5MobileRecLearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = PPOCRV5MobileRecAct(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        # for export
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out

        out = 0
        if self.identity is not None:
            out = out + self.identity(x)

        if self.conv_1x1 is not None:
            out = out + self.conv_1x1(x)

        for conv in self.conv_kxk:
            out = out + conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

    def rep(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True,
        )
        with torch.no_grad():
            self.reparam_conv.weight.copy_(kernel)
            self.reparam_conv.bias.copy_(bias)
        self.is_repped = True

    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        if not isinstance(kernel1x1, torch.Tensor):
            return 0
        else:
            # paddings: (left, right, top, bottom)
            return F.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self):
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1, self.kernel_size // 2)

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk = kernel_conv_kxk + kernel
            bias_conv_kxk = bias_conv_kxk + bias

        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(self, branch):
        if not branch:
            return 0, 0
        elif isinstance(branch, PPOCRV5MobileRecConvBNLayer_PPLCNet):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class PPOCRV5MobileRecSELayer(nn.Module):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = identity * x
        return x


class PPOCRV5MobileRecLCNetV3Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        conv_kxk_num=4,
        lr_mult=1.0,
        lab_lr=0.1,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = PPOCRV5MobileRecLearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )
        if use_se:
            self.se = PPOCRV5MobileRecSELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = PPOCRV5MobileRecLearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPOCRV5MobileRecBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        prenorm=True,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_class = eval(norm_layer.replace("nn.", "nn."))
            self.norm1 = norm_class(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = PPOCRV5MobileRecAttention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = nn.Identity()
        if isinstance(norm_layer, str):
            norm_class = eval(norm_layer.replace("nn.", "nn."))
            self.norm2 = norm_class(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = PPOCRV5MobileRecMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PPOCRV5MobileRecPPLCNetV3(nn.Module):
    def __init__(
        self,
        net_config,
        scale=1.0,
        conv_kxk_num=4,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        lab_lr=0.1,
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.lr_mult_list = lr_mult_list
        self.net_config = net_config

        assert isinstance(self.lr_mult_list, (list, tuple)), (
            f"lr_mult_list should be in (list, tuple) but got {type(self.lr_mult_list)}"
        )
        assert len(self.lr_mult_list) == 6, f"lr_mult_list length should be 6 but got {len(self.lr_mult_list)}"

        self.conv1 = PPOCRV5MobileRecConvBNLayer_PPLCNet(
            in_channels=3,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=2,
            lr_mult=self.lr_mult_list[0],
        )

        self.blocks2 = nn.Sequential(
            *[
                PPOCRV5MobileRecLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[1],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                PPOCRV5MobileRecLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[2],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                PPOCRV5MobileRecLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[3],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                PPOCRV5MobileRecLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[4],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                PPOCRV5MobileRecLCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[5],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )
        self.out_channels = make_divisible(512 * scale)

    def forward(self, x):
        out_list = []
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class PPOCRV5MobileRecFCTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()

        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        if self.only_transpose:
            return x.transpose(1, 2)
        else:
            return self.fc(x.transpose(1, 2))


def zeros_(x):
    return nn.init.constant_(x, 0.0)


class PPOCRV5MobileRecAddPos(nn.Module):
    def __init__(self, dim, w):
        super().__init__()

        self.dec_pos_embed = nn.Parameter(zeros_((1, w, dim)))

    def forward(self, x):
        x = x + self.dec_pos_embed[:, : x.shape[1], :]
        return x


class PPOCRV5MobileRecAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1], dtype=torch.float32)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_paddle = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].flatten(1)
            mask_inf = torch.full([H * W, H * W], float("-inf"), dtype=torch.float32)
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(1))
        self.mixer = mixer

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, -1, 3, self.num_heads, self.head_dim)).permute((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).permute((0, 2, 1, 3)).reshape((B, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PPOCRV5MobileRecConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias_attr=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()

        if isinstance(padding, list):
            padding = tuple(padding)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class PPOCRV5MobileRecMultiHead(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        super().__init__()

        self.head_list = kwargs.pop("head_list")
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)
        self.in_channels = in_channels
        if self.use_pool:
            self.pool = nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2), padding=0)
        self.gtc_head = "sar"
        assert len(self.head_list) >= 2
        for idx, head_name in enumerate(self.head_list):
            name = list(head_name)[0]
            if name == "NRTRHead":
                gtc_args = self.head_list[idx][name]
                max_text_length = gtc_args.get("max_text_length", 25)
                nrtr_dim = gtc_args.get("nrtr_dim", 256)
                num_decoder_layers = gtc_args.get("num_decoder_layers", 4)
                if self.use_pos:
                    self.before_gtc = nn.Sequential(
                        nn.Flatten(2),
                        PPOCRV5MobileRecFCTranspose(in_channels, nrtr_dim),
                        PPOCRV5MobileRecAddPos(nrtr_dim, 80),
                    )
                else:
                    self.before_gtc = nn.Sequential(nn.Flatten(2), PPOCRV5MobileRecFCTranspose(in_channels, nrtr_dim))

                self.gtc_head = PPOCRV5MobileRecTransformer(
                    d_model=nrtr_dim,
                    nhead=nrtr_dim // 32,
                    num_encoder_layers=-1,
                    beam_size=-1,
                    num_decoder_layers=num_decoder_layers,
                    max_len=max_text_length,
                    dim_feedforward=nrtr_dim * 4,
                    out_channels=out_channels_list["NRTRLabelDecode"],
                )
            elif name == "CTCHead":
                # ctc neck
                self.encoder_reshape = PPOCRV5MobileRecIm2Seq(in_channels)
                neck_args = copy.deepcopy(self.head_list[idx][name]["Neck"])
                encoder_type = neck_args.pop("name")
                self.ctc_encoder = PPOCRV5MobileRecSequenceEncoder(
                    in_channels=in_channels, encoder_type=encoder_type, **neck_args
                )
                # ctc head
                head_args = self.head_list[idx][name]["Head"]
                self.ctc_head = PPOCRV5MobileRecCTCHead(
                    in_channels=self.ctc_encoder.out_channels,
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **head_args,
                )
            else:
                raise NotImplementedError(f"{name} is not supported in MultiHead yet")

    def forward(self, x, targets=None):
        if self.use_pool:
            x = self.pool(x.reshape(x.shape[0], 3, -1, self.in_channels).permute(0, 3, 1, 2))
        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder, targets)
        head_out = {}
        head_out["ctc"] = ctc_out
        head_out["ctc_neck"] = ctc_encoder
        if not self.training:
            return ctc_out
        if self.gtc_head == "sar":
            sar_out = self.sar_head(x, targets[1:])
            head_out["sar"] = sar_out
        else:
            gtc_out = self.gtc_head(self.before_gtc(x), targets[1:])
            head_out["gtc"] = gtc_out
        return head_out


class PPOCRV5MobileRecCTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.in_channels = in_channels
        self.fc_decay = fc_decay

        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result


class PPOCRV5MobileRecMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PPOCRV5MobileRecTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        beam_size=0,
        num_decoder_layers=6,
        max_len=25,
        dim_feedforward=1024,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        in_channels=0,
        out_channels=0,
        scale_embedding=True,
    ):
        super().__init__()

        self.out_channels = out_channels + 1
        self.max_len = max_len
        self.embedding = PPOCRV5MobileRecEmbeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PPOCRV5MobileRecPositionalEncoding(dropout=residual_dropout_rate, dim=d_model)

        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList(
                [
                    PPOCRV5MobileRecTransformerBlock(
                        d_model,
                        nhead,
                        dim_feedforward,
                        attention_dropout_rate,
                        residual_dropout_rate,
                        with_self_attn=True,
                        with_cross_attn=False,
                    )
                    for i in range(num_encoder_layers)
                ]
            )
        else:
            self.encoder = None

        self.decoder = nn.ModuleList(
            [
                PPOCRV5MobileRecTransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for i in range(num_decoder_layers)
            ]
        )

        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(self.out_channels, d_model, bias=False)

    def forward_train(self, src, tgt):
        tgt = tgt[:, :-1]

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)

        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src
        else:
            memory = src
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
        output = tgt
        logit = self.tgt_word_prj(output)
        return logit

    def forward(self, src, targets=None):
        if self.training:
            max_len = targets[1].max()
            tgt = targets[0][:, : 2 + max_len]
            return self.forward_train(src, tgt)
        else:
            if self.beam_size > 0:
                return self.forward_beam(src)
            else:
                return self.forward_test(src)

    def forward_test(self, src):
        bs = src.shape[0]
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
            memory = src
        else:
            memory = src
        dec_seq = torch.full((bs, 1), 2, dtype=torch.long, device=src.device)
        dec_prob = torch.full((bs, 1), 1.0, dtype=torch.float32, device=src.device)
        for len_dec_seq in range(1, self.max_len):
            dec_seq_embed = self.embedding(dec_seq)
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(dec_seq_embed.shape[1]).to(dec_seq_embed.device)
            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt = decoder_layer(tgt, memory, self_mask=tgt_mask)
            dec_output = tgt
            dec_output = dec_output[:, -1, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            preds_idx = torch.argmax(word_prob, dim=-1)
            if torch.equal(preds_idx, torch.full_like(preds_idx, 3)):
                break
            preds_prob = torch.max(word_prob, dim=-1)[0]
            dec_seq = torch.cat([dec_seq, preds_idx.view(-1, 1)], dim=1)
            dec_prob = torch.cat([dec_prob, preds_prob.view(-1, 1)], dim=1)
        return [dec_seq, dec_prob]

    def forward_beam(self, images):
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            beamed_tensor_shape = beamed_tensor.shape
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (
                n_curr_active_inst * n_bm,
                beamed_tensor_shape[1],
                beamed_tensor_shape[2],
            )

            beamed_tensor = beamed_tensor.reshape(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(
                0, torch.tensor(curr_active_inst_idx, dtype=torch.long, device=beamed_tensor.device)
            )
            beamed_tensor = beamed_tensor.reshape(new_shape)

            return beamed_tensor

        def collate_active_info(src_enc, inst_idx_to_position_map, active_inst_idx_list):
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_src_enc = collect_active_part(
                src_enc.permute(1, 0, 2), active_inst_idx, n_prev_active_inst, n_bm
            ).permute(1, 0, 2)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq)
                dec_partial_seq = dec_partial_seq.reshape(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
                dec_seq = self.embedding(dec_seq)
                dec_seq = self.positional_encoding(dec_seq)
                tgt_mask = self.generate_square_subsequent_mask(dec_seq.shape[1]).to(dec_seq.device)
                tgt = dec_seq
                for decoder_layer in self.decoder:
                    tgt = decoder_layer(tgt, enc_output, self_mask=tgt_mask)
                dec_output = tgt
                dec_output = dec_output[:, -1, :]
                word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.reshape(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)
            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            if self.encoder is not None:
                src = self.positional_encoding(images)
                src_enc = src
                for encoder_layer in self.encoder:
                    src_enc = encoder_layer(src_enc)
            else:
                src_enc = images

            n_bm = self.beam_size
            inst_dec_beams = [PPOCRV5MobileRecBeam(n_bm) for _ in range(1)]
            active_inst_idx_list = list(range(1))
            src_enc = src_enc.repeat(1, n_bm, 1)
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            for len_dec_seq in range(1, self.max_len):
                src_enc_copy = src_enc.clone()
                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_enc_copy,
                    inst_idx_to_position_map,
                    n_bm,
                )
                if not active_inst_idx_list:
                    break
                src_enc, inst_idx_to_position_map = collate_active_info(
                    src_enc_copy, inst_idx_to_position_map, active_inst_idx_list
                )
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
        result_hyp = []
        hyp_scores = []
        for bs_hyp, score in zip(batch_hyp, batch_scores):
            l = len(bs_hyp[0])
            bs_hyp_pad = bs_hyp[0] + [3] * (25 - l)
            result_hyp.append(bs_hyp_pad)
            score = float(score[0]) / l
            hyp_score = [score for _ in range(25)]
            hyp_scores.append(hyp_score)
        return [
            torch.tensor(np.array(result_hyp), dtype=torch.long),
            torch.tensor(hyp_scores, dtype=torch.float32),
        ]

    def generate_square_subsequent_mask(self, sz):
        mask = torch.zeros(sz, sz, dtype=torch.float32)
        mask_inf = torch.triu(torch.full((sz, sz), float("-inf"), dtype=torch.float32), diagonal=1)
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0)


class PPOCRV5MobileRecMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim**-0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        qN = query.shape[1]

        if self.self_attn:
            qkv = self.qkv(query).reshape(query.shape[0], qN, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            kN = key.shape[1]
            q = self.q(query).reshape(query.shape[0], qN, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kv = self.kv(key).reshape(key.shape[0], kN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape(query.shape[0], qN, self.embed_dim)
        x = self.out_proj(x)

        return x


class PPOCRV5MobileRecTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super().__init__()

        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = PPOCRV5MobileRecMultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate, self_attn=with_self_attn
            )
            self.norm1 = LayerNorm(d_model, eps=epsilon)
            self.dropout1 = Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = PPOCRV5MobileRecMultiheadAttention(d_model, nhead, dropout=attention_dropout_rate)
            self.norm2 = LayerNorm(d_model, eps=epsilon)
            self.dropout2 = Dropout(residual_dropout_rate)

        self.mlp = PPOCRV5MobileRecMlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )

        self.norm3 = LayerNorm(d_model, eps=epsilon)
        self.dropout3 = Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))
        return tgt


class PPOCRV5MobileRecPositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.permute(1, 0, 2)
        self.register_buffer("pe", pe.contiguous())

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x).permute(1, 0, 2)


class PPOCRV5MobileRecEmbeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model
        self.vocab = vocab
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


class PPOCRV5MobileRecBeam:
    def __init__(self, size, device=False):
        self.size = size
        self._done = False
        self.scores = torch.zeros((size,), dtype=torch.float32)
        self.all_scores = []
        self.prev_ks = []
        self.next_ys = [torch.full((size,), 0, dtype=torch.long)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        num_words = word_prob.shape[1]
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.reshape(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        if self.next_ys[-1][0] == 3:
            self._done = True
            self.all_scores.append(self.scores)
        return self._done

    def sort_scores(self):
        return self.scores, torch.arange(self.scores.shape[0], dtype=torch.int32)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[2] + h for h in hyps]
            dec_seq = torch.tensor(hyps, dtype=torch.long)
        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return [x.item() for x in hyp[::-1]]


class PPOCRV5MobileRecIm2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # (batch, width, channels)
        return x


def get_para_bias_attr(l2_decay, k):
    stdv = 1.0 / math.sqrt(k * 1.0)

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -stdv, stdv)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -stdv, stdv)

    return weight_init


class PPOCRV5MobileRecEncoderWithFC(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()

        self.out_channels = hidden_size
        weight_attr, bias_attr = get_para_bias_attr(l2_decay=0.00001, k=in_channels)
        self.fc = nn.Linear(
            in_channels,
            hidden_size,
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PPOCRV5MobileRecEncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels,
        dims=64,
        depth=2,
        hidden_dims=120,
        use_guide=False,
        num_heads=8,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path=0.0,
        kernel_size=[3, 3],
        qk_scale=None,
    ):
        super().__init__()

        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = PPOCRV5MobileRecConvBNLayer(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=nn.SiLU,
        )
        self.conv2 = PPOCRV5MobileRecConvBNLayer(in_channels // 8, hidden_dims, kernel_size=1, act=nn.SiLU)

        self.svtr_block = nn.ModuleList(
            [
                PPOCRV5MobileRecBlock(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    epsilon=1e-05,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = PPOCRV5MobileRecConvBNLayer(hidden_dims, in_channels, kernel_size=1, act=nn.SiLU)
        self.conv4 = PPOCRV5MobileRecConvBNLayer(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=[kernel_size[0] // 2, kernel_size[1] // 2],
            act=nn.SiLU,
        )

        self.conv1x1 = PPOCRV5MobileRecConvBNLayer(in_channels // 8, dims, kernel_size=1, act=nn.SiLU)
        self.out_channels = dims

    def forward(self, x):
        if self.use_guide:
            z = x.clone().detach()
        else:
            z = x
        # for short cut
        h = z
        # reduce dim
        z = self.conv1(z)
        z = self.conv2(z)
        # SVTR global block
        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)
        # last stage
        z = z.view(B, H, W, C).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))

        return z


class PPOCRV5MobileRecSequenceEncoder(nn.Module):
    def __init__(self, in_channels, encoder_type, hidden_size=48, **kwargs):
        super().__init__()

        self.encoder_reshape = PPOCRV5MobileRecIm2Seq(in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder_type = encoder_type
        if encoder_type == "reshape":
            self.only_reshape = True
        else:
            support_encoder_dict = {"svtr": PPOCRV5MobileRecEncoderWithSVTR}
            assert encoder_type in support_encoder_dict, f"{encoder_type} must in {support_encoder_dict.keys()}"
            if encoder_type == "svtr":
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, **kwargs)
            elif encoder_type == "cascadernn":
                self.encoder = support_encoder_dict[encoder_type](
                    self.encoder_reshape.out_channels, hidden_size, **kwargs
                )
            else:
                self.encoder = support_encoder_dict[encoder_type](self.encoder_reshape.out_channels, hidden_size)
            self.out_channels = self.encoder.out_channels
            self.only_reshape = False

    def forward(self, x):
        if self.encoder_type != "svtr":
            x = self.encoder_reshape(x)
            if not self.only_reshape:
                x = self.encoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.encoder_reshape(x)
            return x


def trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)


@auto_docstring(custom_intro="ImageProcessor for the PP-OCRv5_mobile_rec model.")
class PPOCRV5MobileRecImageProcessor(BaseImageProcessor):
    r"""
    Constructs a PPOCRV5MobileRec image processor.

    Args:
        rec_image_shape (`List[int]`, *optional*, defaults to `[3, 48, 320]`):
            The target image shape for recognition in format [channels, height, width].
        max_img_width (`int`, *optional*, defaults to `3200`):
            The maximum width allowed for the resized image.
        character_list (`List[str]` or `str`, *optional*, defaults to `None`):
            The list of characters for text recognition decoding. If `None`, defaults to
            "0123456789abcdefghijklmnopqrstuvwxyz".
        use_space_char (`bool`, *optional*, defaults to `True`):
            Whether to include space character in the character list.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image pixel values to [0, 1] by dividing by 255.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image with mean=0.5 and std=0.5.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        rec_image_shape: list[int] = [3, 48, 320],
        max_img_width: int = 3200,
        character_list: list[str] | str | None = None,
        use_space_char: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rec_image_shape = rec_image_shape if rec_image_shape is not None else [3, 48, 320]
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

        # Initialize character list for decoding
        self._init_character_list(character_list, use_space_char)

    def _init_character_list(
        self,
        character_list: list[str] | str | None,
        use_space_char: bool,
    ) -> None:
        """
        Initialize the character list and character-to-index mapping for CTC decoding.

        Args:
            character_list (`List[str]` or `str`, *optional*):
                The list of characters or a string of characters. If `None`, defaults to
                "0123456789abcdefghijklmnopqrstuvwxyz".
            use_space_char (`bool`):
                Whether to include space character in the character list.
        """
        if character_list is None:
            characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
        elif isinstance(character_list, str):
            characters = list(character_list)
        else:
            characters = list(character_list)

        if use_space_char:
            characters.append(" ")

        # Add CTC blank token at the beginning
        characters = ["blank"] + characters

        self.character = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}

    def _resize_norm_img(
        self,
        img: np.ndarray,
        max_wh_ratio: float,
        data_format: ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Resize and normalize a single image while maintaining aspect ratio.

        Args:
            img (`np.ndarray`):
                The input image in HWC format.
            max_wh_ratio (`float`):
                The maximum width-to-height ratio for resizing.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image.

        Returns:
            `np.ndarray`: The processed image in CHW format with padding.
        """
        img_c, img_h, img_w = self.rec_image_shape

        # Calculate target width based on max_wh_ratio
        target_w = int(img_h * max_wh_ratio)

        if target_w > self.max_img_width:
            # If target width exceeds max, resize to max width
            resized_image = self._ocr_resize(img, (self.max_img_width, img_h))
            resized_w = self.max_img_width
            target_w = self.max_img_width
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(img_h * ratio) > target_w:
                resized_w = target_w
            else:
                resized_w = int(math.ceil(img_h * ratio))
            resized_image = self._ocr_resize(img, (resized_w, img_h))

        # Convert to float32
        resized_image = resized_image.astype(np.float32)

        # Transpose to CHW format
        resized_image = resized_image.transpose((2, 0, 1))

        # Rescale to [0, 1]
        if self.do_rescale:
            resized_image = resized_image / 255.0

        # Normalize with mean=0.5, std=0.5
        if self.do_normalize:
            resized_image = (resized_image - 0.5) / 0.5

        # Create padded image
        padding_im = np.zeros((img_c, img_h, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def preprocess(
        self,
        img: ImageInput,
        rec_image_shape: list[int] | None = None,
        max_img_width: int | None = None,
        do_rescale: bool | None = None,
        do_normalize: bool | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image for PPOCRV5MobileRec text recognition.

        Args:
            img (`ImageInput`):
                The input image to preprocess. Can be a PIL Image, numpy array, or torch tensor.
            rec_image_shape (`List[int]`, *optional*):
                The target image shape [channels, height, width]. Defaults to `self.rec_image_shape`.
            max_img_width (`int`, *optional*):
                The maximum width for the resized image. Defaults to `self.max_img_width`.
            do_rescale (`bool`, *optional*):
                Whether to rescale pixel values to [0, 1]. Defaults to `self.do_rescale`.
            do_normalize (`bool`, *optional*):
                Whether to normalize with mean=0.5 and std=0.5. Defaults to `self.do_normalize`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be "pt", "tf", "np", or None.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format of the output image.

        Returns:
            `BatchFeature`: A BatchFeature containing the processed `pixel_values`.
        """
        # Use instance defaults if not specified
        rec_image_shape = rec_image_shape if rec_image_shape is not None else self.rec_image_shape
        max_img_width = max_img_width if max_img_width is not None else self.max_img_width
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize

        # Store original values and temporarily update for processing
        original_rec_image_shape = self.rec_image_shape
        original_max_img_width = self.max_img_width
        original_do_rescale = self.do_rescale
        original_do_normalize = self.do_normalize

        self.rec_image_shape = rec_image_shape
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize

        try:
            # Convert to numpy array
            img = np.array(img)

            # Get image dimensions
            img_c, img_h, img_w = self.rec_image_shape
            h, w = img.shape[:2]

            # Calculate max_wh_ratio dynamically
            base_wh_ratio = img_w / img_h
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(base_wh_ratio, wh_ratio)

            # Process the image
            processed_img = self._resize_norm_img(img, max_wh_ratio)

            # Add batch dimension
            processed_img = np.expand_dims(processed_img, axis=0)

            data = {"pixel_values": processed_img}
            return BatchFeature(data=data, tensor_type=return_tensors)

        finally:
            # Restore original values
            self.rec_image_shape = original_rec_image_shape
            self.max_img_width = original_max_img_width
            self.do_rescale = original_do_rescale
            self.do_normalize = original_do_normalize

    def _ocr_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to exactly match cv2.resize behavior using fixed-point arithmetic.

        This implementation uses fixed-point arithmetic (matching OpenCV's internal approach)
        with float32 precision for coordinate calculations to achieve maximum compatibility.

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV uses fixed-point arithmetic with these constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scaling factors using float32 (like cv2)
        scale_x = np.float32(w) / np.float32(target_w)
        scale_y = np.float32(h) / np.float32(target_h)

        # Create coordinate grids for output image using float32
        out_y, out_x = np.meshgrid(
            np.arange(target_h, dtype=np.float32), np.arange(target_w, dtype=np.float32), indexing="ij"
        )

        # Map output coordinates to input coordinates (pixel-center alignment) using float32
        src_x = (out_x + np.float32(0.5)) * scale_x - np.float32(0.5)
        src_y = (out_y + np.float32(0.5)) * scale_y - np.float32(0.5)

        # Clip to valid range
        src_x = np.clip(src_x, np.float32(0), np.float32(w - 1))
        src_y = np.clip(src_y, np.float32(0), np.float32(h - 1))

        # Get integer parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        # Calculate fractional parts (keep in float32)
        fx = src_x - x0.astype(np.float32)
        fy = src_y - y0.astype(np.float32)

        # Convert to fixed-point (with rounding, matching cv2's behavior)
        wx = np.round(fx * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        wy = np.round(fy * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)

        # Clamp to valid range
        wx = np.minimum(wx, INTER_RESIZE_COEF_SCALE)
        wy = np.minimum(wy, INTER_RESIZE_COEF_SCALE)

        # Calculate the four interpolation weights in fixed-point
        w0 = (INTER_RESIZE_COEF_SCALE - wx) * (INTER_RESIZE_COEF_SCALE - wy)
        w1 = wx * (INTER_RESIZE_COEF_SCALE - wy)
        w2 = (INTER_RESIZE_COEF_SCALE - wx) * wy
        w3 = wx * wy

        # Perform bilinear interpolation for each channel using fixed-point arithmetic
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)

        for c in range(img.shape[2]):
            # Get the four corner pixel values (as int32 for fixed-point math)
            Ia = img[y0, x0, c].astype(np.int32)
            Ib = img[y0, x1, c].astype(np.int32)
            Ic = img[y1, x0, c].astype(np.int32)
            Id = img[y1, x1, c].astype(np.int32)

            # Fixed-point interpolation
            val = w0 * Ia + w1 * Ib + w2 * Ic + w3 * Id

            # Divide by INTER_RESIZE_COEF_SCALE^2 with rounding
            # This is equivalent to: (val + (1 << 21)) >> 22
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            # Clamp to [0, 255]
            output[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def _ctc_decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray,
        is_remove_duplicate: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Decode CTC output indices to text.

        Args:
            text_index (`np.ndarray`):
                The predicted character indices with shape (batch_size, sequence_length).
            text_prob (`np.ndarray`):
                The predicted character probabilities with shape (batch_size, sequence_length).
            is_remove_duplicate (`bool`, *optional*, defaults to `True`):
                Whether to remove duplicate consecutive characters.

        Returns:
            `List[Tuple[str, float]]`: A list of tuples containing (decoded_text, confidence_score).
        """
        result_list = []
        ignored_tokens = [0]  # CTC blank token
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def post_process_text_recognition(
        self,
        pred: np.ndarray,
    ) -> tuple[list[str], list[float]]:
        """
        Post-process the model output to decode text recognition results.

        Args:
            pred (`np.ndarray`):
                The model output predictions. Expected shape is (batch_size, sequence_length, num_classes)
                or a list/tuple containing such an array.

        Returns:
            `Tuple[List[str], List[float]]`: A tuple containing:
                - texts: List of decoded text strings.
                - scores: List of confidence scores for each decoded text.
        """
        preds = np.array(pred[0].detach().cpu())
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)

        text = self._ctc_decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
        )

        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])

        return texts, scores


@auto_docstring(custom_intro="FastImageProcessor for the PP-OCRv5_mobile_rec model.")
class PPOCRV5MobileRecImageProcessorFast(BaseImageProcessorFast):
    r"""
    Constructs a fast PPOCRV5MobileRec image processor that supports batch processing.

    This processor is designed to handle multiple images efficiently while maintaining
    strict compatibility with [`PPOCRV5MobileRecImageProcessor`]. The preprocessing
    results are guaranteed to be identical to the non-fast version.

    Args:
        rec_image_shape (`List[int]`, *optional*, defaults to `[3, 48, 320]`):
            The target image shape for recognition in format [channels, height, width].
        max_img_width (`int`, *optional*, defaults to `3200`):
            The maximum width allowed for the resized image.
        character_list (`List[str]` or `str`, *optional*, defaults to `None`):
            The list of characters for text recognition decoding. If `None`, defaults to
            "0123456789abcdefghijklmnopqrstuvwxyz".
        use_space_char (`bool`, *optional*, defaults to `True`):
            Whether to include space character in the character list.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image pixel values to [0, 1] by dividing by 255.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image with mean=0.5 and std=0.5.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The mean values for image normalization. Used for validation but actual
            normalization uses fixed value 0.5 in `_resize_norm_img`.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The standard deviation values for image normalization. Used for validation
            but actual normalization uses fixed value 0.5 in `_resize_norm_img`.

    Examples:

    ```python
    >>> from PIL import Image
    >>> from transformers import PPOCRV5MobileRecImageProcessorFast

    >>> processor = PPOCRV5MobileRecImageProcessorFast()

    >>> # Process a single image
    >>> image = Image.open("text_image.png")
    >>> inputs = processor(image, return_tensors="pt")

    >>> # Process multiple images in batch
    >>> images = [Image.open(f"text_image_{i}.png") for i in range(4)]
    >>> batch_inputs = processor(images, return_tensors="pt")
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        rec_image_shape: list[int] | None = None,
        max_img_width: int = 3200,
        character_list: list[str] | str | None = None,
        use_space_char: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rec_image_shape = rec_image_shape if rec_image_shape is not None else [3, 48, 320]
        self.max_img_width = max_img_width
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        # Set default image_mean and image_std for normalization (mean=0.5, std=0.5)
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        # Initialize character list for decoding
        self._init_character_list(character_list, use_space_char)

    def _init_character_list(
        self,
        character_list: list[str] | str | None,
        use_space_char: bool,
    ) -> None:
        """
        Initialize the character list and character-to-index mapping for CTC decoding.

        Args:
            character_list (`List[str]` or `str`, *optional*):
                The list of characters or a string of characters. If `None`, defaults to
                "0123456789abcdefghijklmnopqrstuvwxyz".
            use_space_char (`bool`):
                Whether to include space character in the character list.
        """
        if character_list is None:
            characters = list("0123456789abcdefghijklmnopqrstuvwxyz")
        elif isinstance(character_list, str):
            characters = list(character_list)
        else:
            characters = list(character_list)

        if use_space_char:
            characters.append(" ")

        # Add CTC blank token at the beginning
        characters = ["blank"] + characters

        self.character = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(characters)}

    def _ocr_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to exactly match cv2.resize behavior using fixed-point arithmetic.

        This implementation uses fixed-point arithmetic (matching OpenCV's internal approach)
        with float32 precision for coordinate calculations to achieve maximum compatibility.

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV uses fixed-point arithmetic with these constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scaling factors using float32 (like cv2)
        scale_x = np.float32(w) / np.float32(target_w)
        scale_y = np.float32(h) / np.float32(target_h)

        # Create coordinate grids for output image using float32
        out_y, out_x = np.meshgrid(
            np.arange(target_h, dtype=np.float32), np.arange(target_w, dtype=np.float32), indexing="ij"
        )

        # Map output coordinates to input coordinates (pixel-center alignment) using float32
        src_x = (out_x + np.float32(0.5)) * scale_x - np.float32(0.5)
        src_y = (out_y + np.float32(0.5)) * scale_y - np.float32(0.5)

        # Clip to valid range
        src_x = np.clip(src_x, np.float32(0), np.float32(w - 1))
        src_y = np.clip(src_y, np.float32(0), np.float32(h - 1))

        # Get integer parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        # Calculate fractional parts (keep in float32)
        fx = src_x - x0.astype(np.float32)
        fy = src_y - y0.astype(np.float32)

        # Convert to fixed-point (with rounding, matching cv2's behavior)
        wx = np.round(fx * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        wy = np.round(fy * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)

        # Clamp to valid range
        wx = np.minimum(wx, INTER_RESIZE_COEF_SCALE)
        wy = np.minimum(wy, INTER_RESIZE_COEF_SCALE)

        # Calculate the four interpolation weights in fixed-point
        w0 = (INTER_RESIZE_COEF_SCALE - wx) * (INTER_RESIZE_COEF_SCALE - wy)
        w1 = wx * (INTER_RESIZE_COEF_SCALE - wy)
        w2 = (INTER_RESIZE_COEF_SCALE - wx) * wy
        w3 = wx * wy

        # Perform bilinear interpolation for each channel using fixed-point arithmetic
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)

        for c in range(img.shape[2]):
            # Get the four corner pixel values (as int32 for fixed-point math)
            Ia = img[y0, x0, c].astype(np.int32)
            Ib = img[y0, x1, c].astype(np.int32)
            Ic = img[y1, x0, c].astype(np.int32)
            Id = img[y1, x1, c].astype(np.int32)

            # Fixed-point interpolation
            val = w0 * Ia + w1 * Ib + w2 * Ic + w3 * Id

            # Divide by INTER_RESIZE_COEF_SCALE^2 with rounding
            # This is equivalent to: (val + (1 << 21)) >> 22
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            # Clamp to [0, 255]
            output[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def _resize_norm_img(
        self,
        img: np.ndarray,
        max_wh_ratio: float,
        data_format: ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Resize and normalize a single image while maintaining aspect ratio.

        This method uses PIL-based resizing instead of cv2 while maintaining
        consistent preprocessing results with [`PPOCRV5MobileRecImageProcessor`].

        Args:
            img (`np.ndarray`):
                The input image in HWC format.
            max_wh_ratio (`float`):
                The maximum width-to-height ratio for resizing.
            data_format (`ChannelDimension`, *optional*):
                The channel dimension format of the output image.

        Returns:
            `np.ndarray`: The processed image in CHW format with padding.
        """
        img_c, img_h, img_w = self.rec_image_shape

        # Calculate target width based on max_wh_ratio
        target_w = int(img_h * max_wh_ratio)

        if target_w > self.max_img_width:
            # If target width exceeds max, resize to max width
            resized_image = self._ocr_resize(img, (self.max_img_width, img_h))
            resized_w = self.max_img_width
            target_w = self.max_img_width
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(img_h * ratio) > target_w:
                resized_w = target_w
            else:
                resized_w = int(math.ceil(img_h * ratio))
            resized_image = self._ocr_resize(img, (resized_w, img_h))

        # Convert to float32
        resized_image = resized_image.astype(np.float32)

        # Transpose to CHW format
        resized_image = resized_image.transpose((2, 0, 1))

        # Rescale to [0, 1]
        if self.do_rescale:
            resized_image = resized_image / 255.0

        # Normalize with mean=0.5, std=0.5
        if self.do_normalize:
            resized_image = (resized_image - 0.5) / 0.5

        # Create padded image
        padding_im = np.zeros((img_c, img_h, target_w), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess a batch of images for text recognition.

        Args:
            images (`List[torch.Tensor]`):
                List of images to preprocess.
            **kwargs:
                Additional keyword arguments.

        Returns:
            `BatchFeature`: A dictionary containing the processed pixel values.
        """
        # Convert torch tensors to numpy arrays in HWC format
        np_images = []
        for img in images:
            # img is a torch.Tensor in CHW format, convert to HWC numpy array
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).numpy()
            else:
                img_np = np.array(img)
            np_images.append(img_np)

        # Calculate max width-to-height ratio across all images
        for img in np_images:
            imgC, imgH, imgW = self.rec_image_shape
            max_wh_ratio = imgW / imgH
            h, w = img.shape[:2]
            wh_ratio = w / float(h)
            max_wh_ratio = max(max_wh_ratio, wh_ratio)

        # Process each image
        processed_images = []
        for img in np_images:
            processed_img = self._resize_norm_img(
                img,
                max_wh_ratio=max_wh_ratio,
            )
            processed_images.append(processed_img)

        # Stack into batch tensor
        pixel_values = np.stack(processed_images, axis=0)
        pixel_values = torch.from_numpy(pixel_values)

        return BatchFeature(data={"pixel_values": pixel_values})

    def _ctc_decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray,
        is_remove_duplicate: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Decode CTC output indices to text.

        This method is identical to the one in [`PPOCRV5MobileRecImageProcessor`] to ensure
        consistent decoding results.

        Args:
            text_index (`np.ndarray`):
                The predicted character indices with shape (batch_size, sequence_length).
            text_prob (`np.ndarray`):
                The predicted character probabilities with shape (batch_size, sequence_length).
            is_remove_duplicate (`bool`, *optional*, defaults to `True`):
                Whether to remove duplicate consecutive characters.

        Returns:
            `List[Tuple[str, float]]`: A list of tuples containing (decoded_text, confidence_score).
        """
        result_list = []
        ignored_tokens = [0]  # CTC blank token
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)

            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def post_process_text_recognition(
        self,
        pred: np.ndarray,
    ) -> tuple[list[str], list[float]]:
        """
        Post-process the model output to decode text recognition results.

        This method is identical to the one in [`PPOCRV5MobileRecImageProcessor`] to ensure
        consistent post-processing behavior.

        Args:
            pred (`np.ndarray`):
                The model output predictions. Expected shape is (batch_size, sequence_length, num_classes)
                or a list/tuple containing such an array.

        Returns:
            `Tuple[List[str], List[float]]`: A tuple containing:
                - texts: List of decoded text strings.
                - scores: List of confidence scores for each decoded text.
        """
        preds = np.array(pred[0].detach().cpu())
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)

        text = self._ctc_decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
        )

        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])

        return texts, scores


@auto_docstring(custom_intro="Configuration for the PP-OCRv5_mobile_rec model.")
class PPOCRV5MobileRecConfig(PreTrainedConfig):
    model_type = "pp_ocrv5_mobile_rec"
    """
    This is the configuration class to store the configuration of a [`PPOCRV5MobileRec`]. It is used to instantiate a
    PPOCRV5 Mobile text recognition model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the PPOCRV5 Mobile Rec
    [PaddlePaddle/PP-OCRv5-mobile-rec](https://huggingface.co/PaddlePaddle/PP-OCRv5-mobile-rec) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    def __init__(
        self,
        scale: float = 0.95,
        conv_kxk_num: int = 4,
        lr_mult_list: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        lab_lr: float = 0.1,
        net_config: dict | None = None,
        head_list: list | None = None,
        decode_list: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.net_config = net_config
        self.scale = scale
        self.conv_kxk_num = conv_kxk_num
        self.lr_mult_list = lr_mult_list
        self.lab_lr = lab_lr
        self.head_list = head_list
        self.decode_list = decode_list


class PPOCRV5MobileRecPreTrainedModel(PreTrainedModel):
    """
    Base class for all PP-OCRv5_mobile_rec pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: PPOCRV5MobileRecConfig
    _keys_to_ignore_on_load_missing = [r".*num_batches_tracked.*"]
    base_model_prefix = "pp_ocrv5_mobile_rec"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)

        # PPOCRV5MobileRecConvBNLayer_PPLCNet: kaiming_normal for conv layers
        if isinstance(module, PPOCRV5MobileRecConvBNLayer_PPLCNet):
            init.kaiming_normal_(module.conv.weight)

        # PPOCRV5MobileRecCTCHead: uniform initialization based on fc_decay
        if isinstance(module, PPOCRV5MobileRecCTCHead):
            if module.mid_channels is None:
                stdv = 1.0 / math.sqrt(module.in_channels * 1.0)
                init.uniform_(module.fc.weight, -stdv, stdv)
                if module.fc.bias is not None:
                    init.uniform_(module.fc.bias, -stdv, stdv)
            else:
                stdv1 = 1.0 / math.sqrt(module.in_channels * 1.0)
                init.uniform_(module.fc1.weight, -stdv1, stdv1)
                if module.fc1.bias is not None:
                    init.uniform_(module.fc1.bias, -stdv1, stdv1)

                stdv2 = 1.0 / math.sqrt(module.mid_channels * 1.0)
                init.uniform_(module.fc2.weight, -stdv2, stdv2)
                if module.fc2.bias is not None:
                    init.uniform_(module.fc2.bias, -stdv2, stdv2)

        # PPOCRV5MobileRecTransformer: xavier_normal for linear layers, zeros for bias
        if isinstance(module, PPOCRV5MobileRecTransformer):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.0)
            # Special initialization for tgt_word_prj
            w0 = np.random.normal(0.0, module.d_model**-0.5, (module.d_model, module.out_channels)).astype(np.float32)
            module.tgt_word_prj.weight.copy_(torch.from_numpy(w0))

        # PPOCRV5MobileRecAddPos: trunc_normal for dec_pos_embed
        if isinstance(module, PPOCRV5MobileRecAddPos):
            init.trunc_normal_(module.dec_pos_embed, std=0.02)

        # PPOCRV5MobileRecEmbeddings: normal initialization with std based on d_model
        if isinstance(module, PPOCRV5MobileRecEmbeddings):
            w0 = np.random.normal(0.0, module.d_model**-0.5, (module.vocab, module.d_model)).astype(np.float32)
            module.embedding.weight.copy_(torch.from_numpy(w0))

        # PPOCRV5MobileRecEncoderWithSVTR: trunc_normal for linear weights, ones/zeros for LayerNorm
        if isinstance(module, PPOCRV5MobileRecEncoderWithSVTR):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.LayerNorm):
                    init.constant_(m.bias, 0.0)
                    init.constant_(m.weight, 1.0)


@auto_docstring(custom_intro="The PP-OCRv5_mobile_rec model.")
class PPOCRV5MobileRecModel(PPOCRV5MobileRecPreTrainedModel):
    """
    Core PP-OCRv5_mobile_rec model, consisting of Backbone and Head networks.
    Generates structure probs for table recognition tasks.
    """

    def __init__(self, config: PPOCRV5MobileRecConfig):
        super().__init__(config)
        self.backbone = PPOCRV5MobileRecPPLCNetV3(
            scale=config.scale,
            net_config=config.net_config,
            conv_kxk_num=config.conv_kxk_num,
            lr_mult_list=config.lr_mult_list,
            lab_lr=config.lab_lr,
        )
        self.head = PPOCRV5MobileRecMultiHead(
            in_channels=self.backbone.out_channels,
            out_channels_list=config.decode_list,
            head_list=config.head_list,
        )
        self.post_init()

    def forward(self, pixel_values, **kwargs):
        x = self.backbone(pixel_values)
        x = self.head(x)

        return x


@auto_docstring(custom_intro="TextRecognition for the PP-OCRv5_mobile_rec model.")
class PPOCRV5MobileRecForTextRecognition(PPOCRV5MobileRecPreTrainedModel):
    """
    PPOCRV5MobileRec model for table recognition tasks.
    """

    def __init__(self, config: PPOCRV5MobileRecConfig):
        super().__init__(config)
        self.model = PPOCRV5MobileRecModel(config)
        self.post_init()

    def forward(self, pixel_values, return_dict: bool | None = None, **kwargs):
        x = self.model(pixel_values)
        if not return_dict:
            return ((x),)
        else:
            return PPOCRV5MobileRecOutput(text_probs=x)


@dataclass
class PPOCRV5MobileRecOutput(ModelOutput):
    """
    Output class for PPOCRV5MobileRecForTextRecognition. Extends ModelOutput
    to include table recognition probs.
    """

    text_probs: torch.FloatTensor | None = None


__all__ = [
    "PPOCRV5MobileRecForTextRecognition",
    "PPOCRV5MobileRecImageProcessor",
    "PPOCRV5MobileRecImageProcessorFast",
    "PPOCRV5MobileRecConfig",
    "PPOCRV5MobileRecModel",
    "PPOCRV5MobileRecPreTrainedModel",
]
