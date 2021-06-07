# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from .batch_norm import get_norm
from .blocks import DepthwiseSeparableConv2d
from .wrappers import Conv2d


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
        use_depthwise_separable_conv=False,
    ):
        """
        Args:
            in_channels (int): number of input channels for ASPP.
            out_channels (int): number of output channels.
            dilations (list): a list of 3 dilations in ASPP.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format. norm is
                applied to all conv layers except the conv following
                global average pooling.
            activation (callable): activation function.
            pool_kernel_size (tuple, list): the average pooling size (kh, kw)
                for image pooling layer in ASPP. If set to None, it always
                performs global average pooling. If not None, it must be
                divisible by the shape of inputs in forward(). It is recommended
                to use a fixed input feature size in training, and set this
                option to match this size, so that it performs global average
                pooling in training, and the size of the pooling window stays
                consistent in inference.
            dropout (float): apply dropout on the output of ASPP. It is used in
                the official DeepLab implementation with a rate of 0.1:
                https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/model.py#L532  # noqa
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                for 3x3 convs in ASPP, proposed in :paper:`DeepLabV3+`.
        """
        super(ASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        use_bias = norm == ""
        self.convs = nn.ModuleList()
        # conv 1x1
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=deepcopy(activation),
            )
        )
        weight_init.c2_xavier_fill(self.convs[-1])
        # atrous convs
        for dilation in dilations:
            if use_depthwise_separable_conv:
                self.convs.append(
                    DepthwiseSeparableConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        norm1=norm,
                        activation1=deepcopy(activation),
                        norm2=norm,
                        activation2=deepcopy(activation),
                    )
                )
            else:
                self.convs.append(
                    Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                        activation=deepcopy(activation),
                    )
                )
                weight_init.c2_xavier_fill(self.convs[-1])
        # image pooling
        # We do not add BatchNorm because the spatial resolution is 1x1,
        # the original TF implementation has BatchNorm.
        if pool_kernel_size is None:
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        else:
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        weight_init.c2_xavier_fill(image_pooling[1])
        self.convs.append(image_pooling)

        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        size = x.shape[-2:]
        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res
