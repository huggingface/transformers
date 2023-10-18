# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViT model."""


import math
from functools import reduce
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.modules.utils import _pair, _single

from ...modeling_outputs import (
    ProPainterFrameModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_propainter import ProPainterConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ProPainterConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "shauray/ProPainter-hf"

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "shauray/ProPainter-hf"


VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shauray/ProPainter-hf",
]


class P3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_residual=0, bias=True):
        super().__init__()
        config = ProPainterConfig()
        self.Conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=bias,
            ),
            nn.LeakyReLU(config.threshold, inplace=True),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                bias=bias,
            )
        )
        self.use_residual = use_residual

    def forward(self, pixel_values):
        feat1 = self.Conv1(pixel_values)
        feat2 = self.Conv2(feat1)
        if self.use_residual:
            output = feat1 + feat2
        else:
            output = feat2
        return output


class EdgeDetection(nn.Module):
    def __init__(self, config: ProPainterConfig, in_channels, out_channels, hidden_channels):
        super().__init__()

        self.config = config
        self.edge_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1), nn.LeakyReLU(config.threshold, inplace=True)
        )

        self.edge_layer_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1), nn.LeakyReLU(config.threshold, inplace=True)
        )

        self.edge_layer_2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1))

        self.edge_act = nn.LeakyReLU(0.01, inplace=True)

        self.edge_out = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)

    def forward(self, flow):
        flow = self.edge_projection(flow)
        edge = self.edge_layer_1(flow)
        edge = self.edge_layer_2(edge)
        edge = self.edge_act(flow + edge)
        edge = self.edge_out(edge)
        return torch.sigmoid(edge)


class ReccurrentFlowCompleteNet(nn.Module):
    def __init__(self, config: ProPainterConfig):
        super().__init__()

        self.Downsample = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), padding_mode="replicate"),
            nn.LeakyReLU(config.threshold, inplace=True),
        )

        self.Encoder1 = nn.Sequential(
            P3DBlock(32, 32, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            P3DBlock(32, 64, 3, 2, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
        )

        self.Encoder2 = nn.Sequential(
            P3DBlock(64, 64, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            P3DBlock(64, 128, 3, 2, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
        )

        self.MidDilation = nn.Sequential(
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv3d(128, 128, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(config.threshold, inplace=True),
        )

        self.feat_prop_module = BidirectionalPropagation(128)

        self.Decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(64, 32, 3, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
        )  # 2x

        self.Decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(128, 64, 3, 1),
            nn.LeakyReLU(config.threshold, inplace=True),
        )  # 4x

        self.Upsample = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(config.threshold, inplace=True), deconv(32, 2, 3, 1)
        )

        self.EdgeDetector = EdgeDetection(config, in_channels=2, out_channels=1, hidden_channels=16)

        # for m in self.modules():
        #    if isinstance(m, SecondOrderDeformableAlignment):
        #        m.init_offset()

    def forward(self, masked_flows, masks):
        b, t, _, h, w = masked_flows.size()
        masked_flows = masked_flows.permute(0, 2, 1, 3, 4)
        masks = masks.permute(0, 2, 1, 3, 4)

        inputs = torch.cat((masked_flows, masks), dim=1)

        x = self.Downsample(inputs)

        feat_e1 = self.Encoder1(x)
        feat_e2 = self.Encoder2(feat_e1)  # b c t h w
        feat_mid = self.MidDilation(feat_e2)  # b c t h w
        feat_mid = feat_mid.permute(0, 2, 1, 3, 4)  # b t c h w

        feat_prop = self.feat_prop_module(feat_mid)
        feat_prop = feat_prop.view(-1, 128, h // 8, w // 8)  # b*t c h w

        _, c, _, h_f, w_f = feat_e1.shape
        feat_e1 = feat_e1.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)  # b*t c h w
        feat_d2 = self.Decoder2(feat_prop) + feat_e1

        _, c, _, h_f, w_f = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)  # b*t c h w

        feat_d1 = self.Decoder1(feat_d2)

        flow = self.Upsample(feat_d1)
        if self.training:
            edge = self.EdgeDetector(flow)
            edge = edge.view(b, t, 1, h, w)
        else:
            edge = None

        flow = flow.view(b, t, 2, h, w)

        return flow, edge

    def forward_bidirect_flow(self, masked_flows_bi, masks):
        """
        Args:
            masked_flows_bi: [masked_flows_f, masked_flows_b] | (b t-1 2 h w), (b t-1 2 h w)
            masks: b t 1 h w
        """
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        # mask flow
        masked_flows_forward = masked_flows_bi[0] * (1 - masks_forward)
        masked_flows_backward = masked_flows_bi[1] * (1 - masks_backward)

        # -- completion --
        # forward
        pred_flows_forward, pred_edges_forward = self.forward(masked_flows_forward, masks_forward)

        # backward
        masked_flows_backward = torch.flip(masked_flows_backward, dims=[1])
        masks_backward = torch.flip(masks_backward, dims=[1])
        pred_flows_backward, pred_edges_backward = self.forward(masked_flows_backward, masks_backward)
        pred_flows_backward = torch.flip(pred_flows_backward, dims=[1])
        if self.training:
            pred_edges_backward = torch.flip(pred_edges_backward, dims=[1])

        return [pred_flows_forward, pred_flows_backward], [pred_edges_forward, pred_edges_backward]

    def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
        masks_forward = masks[:, :-1, ...].contiguous()
        masks_backward = masks[:, 1:, ...].contiguous()

        pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
        pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)

        return pred_flows_forward, pred_flows_backward


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.Conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.Conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.Relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.Norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.Norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.Norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.Norm1 = nn.BatchNorm2d(planes)
            self.Norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.Norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.Norm1 = nn.InstanceNorm2d(planes)
            self.Norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.Norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.Norm1 = nn.Sequential()
            self.Norm2 = nn.Sequential()
            if not stride == 1:
                self.Norm3 = nn.Sequential()

        if stride == 1:
            self.Downsample = None

        else:
            self.Downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.Norm3)

    def forward(self, x):
        y = x
        y = self.Relu(self.Norm1(self.Conv1(y)))
        y = self.Relu(self.Norm2(self.Conv2(y)))

        if self.Downsample is not None:
            x = self.Downsample(x)

        return self.Relu(x + y)


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.Conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.Conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.Conv2(self.Relu(self.Conv1(x)))


class FlowMotionEncoder(nn.Module):
    def __init__(self, args):
        super(FlowMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.Conv_c1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.Conv_c2 = nn.Conv2d(256, 192, 3, padding=1)
        self.Conv_f1 = nn.Conv2d(2, 128, 7, padding=3)
        self.Conv_f2 = nn.Conv2d(128, 64, 3, padding=1)
        self.Conv_ = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.Conv_c1(corr))
        cor = F.relu(self.Conv_c2(cor))
        flo = F.relu(self.Conv_f1(flow))
        flo = F.relu(self.Conv_f2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.Conv_(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.Conv_z1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.Conv_r1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.Conv_q1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.Conv_z2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.Conv_r2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.Conv_q2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.Conv_z1(hx))
        r = torch.sigmoid(self.Conv_r1(hx))
        q = torch.tanh(self.Conv_q1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.Conv_z2(hx))
        r = torch.sigmoid(self.Conv_r2(hx))
        q = torch.tanh(self.Conv_q2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ModulatedDeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True,
    ):
        super(ModulatedDeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset, mask):
        pass


class FlowUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(FlowUpdateBlock, self).__init__()
        self.args = args
        self.Encoder = FlowMotionEncoder(args)
        self.GRU = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.FlowHead = FlowHead(hidden_dim, hidden_dim=256)

        self.Mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0)
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.Encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.GRU(net, inp)
        delta_flow = self.FlowHead(net)

        # scale mask to balence gradients
        mask = 0.25 * self.Mask(net)
        return net, mask, delta_flow


class FlowEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(FlowEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.Norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.Norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.Norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.Norm1 = nn.Sequential()

        self.Conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.Relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.Conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.Dropout = nn.Dropout2d(p=dropout)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #        if m.weight is not None:
        #            nn.init.constant_(m.weight, 1)
        #        if m.bias is not None:
        #            nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.Conv1(x)
        x = self.Norm1(x)
        x = self.Relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.Conv2(x)

        if self.training and self.dropout is not None:
            x = self.Dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        fmap1 = fmap1.contiguous()
        fmap2 = fmap2.contiguous()
        coords = coords.contiguous()
        ctx.save_for_backward(fmap1, fmap2, coords)
        ctx.r = r
        (corr,) = correlation_cudaz.forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = correlation_cudaz.backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def bilinear_sampler(self, img, coords, mode="bilinear", mask=False):
        """Wrapper for grid_sample, uses pixel coordinates"""
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask

        return img

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim))


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.ConvOffset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, x, cond_feat, flow):
        out = self.ConvOffset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.Layers = nn.ModuleList(
            [
                nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.Layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.Conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.Conv(x)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 5)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.ConvOffset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

    def forward(self, x, extra_feat):
        out = self.ConvOffset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel):
        super(BidirectionalPropagation, self).__init__()
        modules = ["backward_", "forward_"]
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16
            )

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats["spatial"] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ["backward_", "forward_"]:
            feats[module_name] = []

            frame_idx = range(0, t)
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]

            if "backward" in module_name:
                frame_idx = frame_idx[::-1]

            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                if i > 0:
                    cond_n1 = feat_prop

                    # initialize second-order features
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:  # second-order features
                        feat_n2 = feats[module_name][-2]
                        cond_n2 = feat_n2

                    cond = torch.cat(
                        [cond_n1, feat_current, cond_n2], dim=1
                    )  # condition information, cond(flow warped 1st/2nd feature)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)  # two order feat_prop -1 & -2
                    feat_prop = self.deform_align[module_name](feat_prop, cond)

                # fuse current features
                feat = (
                    [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]
                )

                feat = torch.cat(feat, dim=1)
                # embed current features
                feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != "spatial"]
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


class ProPainterBidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(ProPainterBidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ["backward_1", "forward_1"]
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

    def flow_warp(self, x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True):
        """Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        """
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(
                f"The spatial sizes of input ({x.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same."
            )
        _, _, h, w = x.size()
        # create mesh grid
        device = flow.device
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(
            x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners
        )
        return output

    def binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        # return mask.float()
        return mask.to(mask)

    def fbConsistencyCheck(self, flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
        def length_sq(x):
            return torch.sum(torch.square(x), dim=1, keepdim=True)

        flow_bw_warped = self.flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
        flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

        mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
        occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

        # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
        fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
        return fb_valid_fw

    def forward(self, x, flows_forward, flows_backward, mask, interpolation="bilinear"):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats["input"] = [x[:, i, :, :, :] for i in range(0, t)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if "backward" in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = self.fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = self.flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = self.flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                        mask_prop_valid = self.binary_mask(mask_prop_valid)

                        union_vaild_mask = self.binary_mask(mask_current * flow_vaild_mask * (1 - mask_prop_valid))
                        feat_prop = union_vaild_mask * feat_warped + (1 - union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = self.binary_mask(mask_current * (1 - (flow_vaild_mask * (1 - mask_prop_valid))))

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            # end for
            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats["forward_1"], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            _, masks_f = None, None
            outputs = self.fuse(torch.cat([outputs_b, outputs_f, mask_in], dim=1)) + x.view(-1, c, h, w)
        else:
            torch.stack(masks["backward_1"], dim=1)
            masks_f = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), outputs.view(b, -1, c, h, w), masks_f


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(
            feat, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        feat = self.bias_conv(feat)
        return feat


class SparseWindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        window_size,
        pool_size=(4, 4),
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pooling_token=True,
    ):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1.0 / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[: -self.expand_size[0], : -self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[: -self.expand_size[0], self.expand_size[1] :] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0] :, : -self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0] :, self.expand_size[1] :] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1))

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def window_partition(self, x, window_size, n_head):
        """
        Args:
            x: shape is (B, T, H, W, C)
            window_size (tuple[int]): window size
        Returns:
            windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
        """
        B, T, H, W, C = x.shape
        x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C // n_head)
        windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return windows

    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        b, t, h, w, c = x.shape  # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0]  # 20
        new_w = n_ww * self.window_size[1]  # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
            mask = F.pad(mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        win_q = self.window_partition(q.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        win_k = self.window_partition(k.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        win_v = self.window_partition(v.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            (k_tl, v_tl) = (
                torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)) for a in (k, v)
            )
            (k_tr, v_tr) = (
                torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)) for a in (k, v)
            )
            (k_bl, v_bl) = (
                torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)) for a in (k, v)
            )
            (k_br, v_br) = (
                torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)) for a in (k, v)
            )

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = (
                self.window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
                )
                for a in (k_tl, k_tr, k_bl, k_br)
            )
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = (
                self.window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
                )
                for a in (v_tl, v_tr, v_bl, v_br)
            )
            rool_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 4).contiguous()
            rool_v = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 4
            ).contiguous()  # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            rool_v = rool_v.view(b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head)
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)
        else:
            win_k = win_k
            win_v = win_v

        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
            # pool_k
            pool_k = self.key(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = (
                self.value(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            )  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh * n_ww)
        mask = torch.sum(mask, dim=1)  # [b, n_wh*n_ww]
        for i in range(win_q.shape[0]):
            ### For masked windows
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.n_head, t * w_h * w_w, c_head)
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]
                # mask out key and value
                if T_ind is not None:
                    # key [n_wh*n_ww, n_head, t, w_h*w_w, c_head]
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                    # value
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(mask_n, self.n_head, -1, c_head)
                else:
                    win_k_t = win_k_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)
                    win_v_t = win_v_t.view(n_wh * n_ww, self.n_head, t * w_h * w_w, c_head)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h * w_w, c_head)

            ### For unmasked windows
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, : w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, : w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))

        return out


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce((lambda x, y: x * y), t2t_params["kernel_size"])  # 49

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params["kernel_size"]):
            n_vecs *= int(
                (output_size[i] + 2 * self.t2t_params["padding"][i] - (d - 1) - 1) / self.t2t_params["stride"][i] + 1
            )

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(
            normalizer,
            output_size=output_size,
            kernel_size=self.t2t_params["kernel_size"],
            padding=self.t2t_params["padding"],
            stride=self.t2t_params["stride"],
        )

        x = F.fold(
            x.view(-1, n_vecs, c).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.t2t_params["kernel_size"],
            padding=self.t2t_params["padding"],
            stride=self.t2t_params["stride"],
        )

        x = (
            F.unfold(
                x / normalizer,
                kernel_size=self.t2t_params["kernel_size"],
                padding=self.t2t_params["padding"],
                stride=self.t2t_params["stride"],
            )
            .permute(0, 2, 1)
            .contiguous()
            .view(b, n, c)
        )
        x = self.fc2(x)
        return x


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, norm_layer=nn.LayerNorm, t2t_params=None):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, x, fold_x_size, mask=None, T_ind=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape  # 20 36

        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind)

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
            blocks.append(TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t_params=t2t_params))
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, "wrong t_dilation input."
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (self.depths // t_dilation)
        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i])
        return x


class OpticalFlow(nn.Module):
    def __init__(self, config: ProPainterConfig, hidden_dim=128, context_dim=128):
        super(OpticalFlow, self).__init__()
        self.config = config

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        config.corr_levels = 4
        config.corr_radius = 4

        # feature network, context network, and update block
        self.FeatureNet = FlowEncoder(output_dim=256, norm_fn="instance", dropout=config.dropout)
        self.ContextNet = FlowEncoder(output_dim=hidden_dim + context_dim, norm_fn="batch", dropout=config.dropout)
        self.UpdateBlock = FlowUpdateBlock(self.config, hidden_dim=hidden_dim)

        self.l1_criterion = nn.L1Loss()

    def coords_grid(self, batch, ht, wd):
        coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = self.coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = self.coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def upflow8(self, flow, mode="bilinear"):
        new_size = (8 * flow.shape[2], 8 * flow.shape[3])
        return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def OpticalPairs(self, image1, image2, iters=12, flow_init=None, is_training=True):
        """Estimate optical flow between pair of frames"""

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.FeatureNet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.config.corr_radius)

        # run the context network
        cnet = self.ContextNet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            flow = flow.to(image1.dtype)
            net, up_mask, delta_flow = self.UpdateBlock(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = self.upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if not is_training:
            return coords1 - coords0, flow_up

        return flow_predictions

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()
        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            _, gt_flows_forward = self.OpticalPairs(gtlf_1, gtlf_2, iters=iters, is_training=False)
            _, gt_flows_backward = self.OpticalPairs(gtlf_2, gtlf_1, iters=iters, is_training=False)

        gt_flows_forward = gt_flows_forward.view(b, l_t - 1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t - 1, 2, h, w)

        return gt_flows_forward, gt_flows_backward


class ProPainterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ProPainterConfig
    base_model_prefix = "propainter"
    #main_input_name = "frames"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["OpticalFlow.ContextNet.layer2.0","OpticalFlow.ContextNet.layer3.0"]

  
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # elif isinstance(module, ViTEmbeddings):
        #     module.position_embeddings.data = nn.init.trunc_normal_(
        #         module.position_embeddings.data.to(torch.float32),
        #         mean=0.0,
        #         std=self.config.initializer_range,
        #     ).to(module.position_embeddings.dtype)

        #     module.cls_token.data = nn.init.trunc_normal_(
        #         module.cls_token.data.to(torch.float32),
        #         mean=0.0,
        #         std=self.config.initializer_range,
        #     ).to(module.cls_token.dtype)

        def _set_gradient_checkpointing(self, module: ProPainterModel, value: bool = False) -> None:
            if isinstance(module, ProPainterModel):
                module.gradient_checkpointing = value


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        #pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class ProPainterModel(ProPainterPreTrainedModel):
    def __init__(self, config: ProPainterConfig):
        super(ProPainterModel, self).__init__(config)

        self.config = config
        self.OpticalFlow = OpticalFlow(config)
        self.FlowComplete = ReccurrentFlowCompleteNet(config)

        channel = config.num_channels
        hidden = config.hidden_size

        self.encoder = Encoder()

        self.decoder = nn.Sequential(
            deconv(channel, channel, kernel_size=3, padding=1),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(config.threshold, inplace=True),
            deconv(channel // 2, channel // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(config.threshold, inplace=True),
            nn.Conv2d(channel // 2, 3, kernel_size=3, stride=1, padding=1),
        )

        t2t_params = {"kernel_size": config.kernel_size, "stride": config.stride, "padding": config.padding}

        self.ss = SoftSplit(channel, hidden, config.kernel_size, config.stride, config.padding)
        self.sc = SoftComp(channel, hidden, config.kernel_size, config.stride, config.padding)
        self.max_pool = nn.MaxPool2d(config.kernel_size, config.stride, config.padding)

        # feature propagation module
        self.img_prop_module = ProPainterBidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = ProPainterBidirectionalPropagation(128, learnable=True)

        depths = config.transformer_depth
        num_heads = config.transformer_heads
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(
            dim=hidden,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params,
        )
        self.post_init()

    def img_propagation(self, masked_frames, completed_flows, masks, interpolation="nearest"):
        _, _, prop_frames, updated_masks = self.img_prop_module(
            masked_frames, completed_flows[0], completed_flows[1], masks, interpolation
        )
        return prop_frames, updated_masks

    def _forward(
        self,
        masked_frames,
        completed_flows,
        masks_in,
        masks_updated,
        num_local_frames,
        interpolation="bilinear",
        t_dilation=2,
    ):
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()

        # extracting features
        enc_feat = self.encoder(
            torch.cat(
                [
                    masked_frames.view(b * t, 3, ori_h, ori_w),
                    masks_in.view(b * t, 1, ori_h, ori_w),
                    masks_updated.view(b * t, 1, ori_h, ori_w),
                ],
                dim=1,
            )
        )
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        ds_flows_f = (
            F.interpolate(
                completed_flows[0].view(-1, 2, ori_h, ori_w), scale_factor=1 / 4, mode="bilinear", align_corners=False
            ).view(b, l_t - 1, 2, h, w)
            / 4.0
        )
        ds_flows_b = (
            F.interpolate(
                completed_flows[1].view(-1, 2, ori_h, ori_w), scale_factor=1 / 4, mode="bilinear", align_corners=False
            ).view(b, l_t - 1, 2, h, w)
            / 4.0
        )
        ds_mask_in = F.interpolate(masks_in.reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4, mode="nearest").view(
            b, t, 1, h, w
        )
        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4, mode="nearest"
        ).view(b, l_t, 1, h, w)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_feat, _ = self.feat_prop_module(local_feat, ds_flows_f, ds_flows_b, prop_mask_in, interpolation)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        batch_size, time_steps, channels, height, width = mask_pool_l.shape

        mask_pool_l = mask_pool_l.view(batch_size, time_steps, height, width, channels)
        mask_pool_l = mask_pool_l.contiguous()

        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            output = self.decoder(enc_feat[:, :l_t].view(-1, c, h, w))
            output = torch.tanh(output).view(b, l_t, 3, ori_h, ori_w)

        return output

    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        frames_inp: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ProPainterFrameModelingOutput]:

        video_length = frames.size(1)
        h, w = frames.shape[-2], frames.shape[-1]

        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) < 720:
            short_clip_len = 8
        elif frames.size(-1) < 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(frames.size(1), f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = self.OpticalFlow(frames[:, f:end_f], iters=self.config.raft_iter)
                else:
                    flows_f, flows_b = self.OpticalFlow(frames[:, f - 1 : end_f], iters=self.config.raft_iter)

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.OpticalFlow(frames, iters=self.config.raft_iter)

        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.config.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.config.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.config.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.config.subvideo_length)
                pred_flows_bi_sub, _ = self.FlowComplete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), flow_masks[:, s_f : e_f + 1]
                )
                pred_flows_bi_sub = self.FlowComplete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f : e_f + 1],
                )

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = self.FlowComplete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.FlowComplete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(
            100, self.config.subvideo_length
        )  # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f : e_f - 1], pred_flows_bi[1][:, s_f : e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated[:, s_f:e_f], "nearest"
                )
                updated_frames_sub = (
                    frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                    + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                )
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e])

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = self.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest"
            )
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = self.config.neighbor_length // 2
        if video_length > self.config.subvideo_length:
            ref_num = self.config.subvideo_length // self.config.ref_stride
        else:
            ref_num = -1

        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length, self.config.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )

            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            # pred_img = selected_imgs # results of image propagation
            pred_img = self._forward(
                selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t
            )

            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).detach().numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (
                    1 - binary_masks[i]
                )
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        return ProPainterFrameModelingOutput(
            reconstructed_frames=torch.tensor(comp_frames),
            loss=None,
        )

    def get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index
        


@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)

class ProPainterForImageInPainting(ProPainterPreTrainedModel):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__(config)

        self.model = ProPainterModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProPainterFrameModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        frames_inp: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ProPainterFrameModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        frames_inp = frames_inp.cpu().tolist()
        
        out = self.model(
            frames,
            flow_masks,
            masks_dilated,
            frames_inp,
            output_attentions,
            output_hidden_states,
            interpolate_pos_encoding,
            return_dict,
        )

        return ProPainterFrameModelingOutput(
            reconstructed_frames=out,
        )




@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ProPainterForImageOutPainting(ProPainterPreTrainedModel):
    def __init__(self, config: ProPainterConfig) -> None:
        super().__init__(config)

        self.InPainting = ProPainterModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ProPainterFrameModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        flow_masks: Optional[torch.BoolTensor] = None,
        masks_dilated: Optional[torch.Tensor] = None,
        frames_inp: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ProPainterFrameModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        import numpy as np

        video_length = frames.size(1)
        h, w = frames.shape[-2], frames.shape[-1]

        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) < 720:
            short_clip_len = 8
        elif frames.size(-1) < 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(frames.size(1), f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = self.OpticalFlow(frames[:, f:end_f], iters=self.config.raft_iter)
                else:
                    flows_f, flows_b = self.OpticalFlow(frames[:, f - 1 : end_f], iters=self.config.raft_iter)

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = self.OpticalFlow(frames, iters=self.config.raft_iter)

        flow_length = gt_flows_bi[0].size(1)
        if flow_length > self.config.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, self.config.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + self.config.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + self.config.subvideo_length)
                pred_flows_bi_sub, _ = self.FlowComplete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), flow_masks[:, s_f : e_f + 1]
                )
                pred_flows_bi_sub = self.FlowComplete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f : e_f + 1],
                )

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = self.FlowComplete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = self.FlowComplete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(
            100, self.config.subvideo_length
        )  # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f : e_f - 1], pred_flows_bi[1][:, s_f : e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = self.InPainting.img_propagation(
                    masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated[:, s_f:e_f], "nearest"
                )
                updated_frames_sub = (
                    frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                    + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                )
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e])

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = self.InPainting.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest"
            )
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = self.config.neighbor_length // 2
        if video_length > self.config.subvideo_length:
            ref_num = self.config.subvideo_length // self.config.ref_stride
        else:
            ref_num = -1

        for f in range(0, video_length, neighbor_stride):
            neighbor_ids = list(range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)))
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length, self.config.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )

            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            # pred_img = selected_imgs # results of image propagation
            pred_img = self.InPainting(
                selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t
            )

            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (
                    1 - binary_masks[i]
                )
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        return ProPainterFrameModelingOutput(
            reconstructed_frames=comp_frames,
        )


