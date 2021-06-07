# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Implementation of RegNet models from :paper:`dds` and :paper:`scaling`.

This code is adapted from https://github.com/facebookresearch/pycls with minimal modifications.
Some code duplication exists between RegNet and ResNets (e.g., ResStem) in order to simplify
model loading.
"""

import numpy as np
from torch import nn

from detectron2.layers import CNNBlockBase, ShapeSpec, get_norm

from .backbone import Backbone

__all__ = [
    "AnyNet",
    "RegNet",
    "ResStem",
    "SimpleStem",
    "VanillaBlock",
    "ResBasicBlock",
    "ResBottleneckBlock",
]


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def gap2d():
    """Helper for building a global average pooling layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def pool2d(k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


class ResStem(CNNBlockBase):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out, norm, activation_class):
        super().__init__(w_in, w_out, 4)
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = get_norm(norm, w_out)
        self.af = activation_class()
        self.pool = pool2d(3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStem(CNNBlockBase):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm, activation_class):
        super().__init__(w_in, w_out, 2)
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = get_norm(norm, w_out)
        self.af = activation_class()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se, activation_class):
        super().__init__()
        self.avg_pool = gap2d()
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation_class(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class VanillaBlock(CNNBlockBase):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, norm, activation_class, _params):
        super().__init__(w_in, w_out, stride)
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = get_norm(norm, w_out)
        self.a_af = activation_class()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = get_norm(norm, w_out)
        self.b_af = activation_class()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, norm, activation_class, _params):
        super().__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = get_norm(norm, w_out)
        self.a_af = activation_class()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = get_norm(norm, w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(CNNBlockBase):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, norm, activation_class, params):
        super().__init__(w_in, w_out, stride)
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = get_norm(norm, w_out)
        self.f = BasicTransform(w_in, w_out, stride, norm, activation_class, params)
        self.af = activation_class()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, norm, activation_class, params):
        super().__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = get_norm(norm, w_b)
        self.a_af = activation_class()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = get_norm(norm, w_b)
        self.b_af = activation_class()
        self.se = SE(w_b, w_se, activation_class) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = get_norm(norm, w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(CNNBlockBase):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, norm, activation_class, params):
        super().__init__(w_in, w_out, stride)
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = get_norm(norm, w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, norm, activation_class, params)
        self.af = activation_class()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_class, norm, activation_class, params):
        super().__init__()
        for i in range(d):
            block = block_class(w_in, w_out, stride, norm, activation_class, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(Backbone):
    """AnyNet model. See :paper:`dds`."""

    def __init__(
        self,
        *,
        stem_class,
        stem_width,
        block_class,
        depths,
        widths,
        group_widths,
        strides,
        bottleneck_ratios,
        se_ratio,
        activation_class,
        freeze_at=0,
        norm="BN",
        out_features=None,
    ):
        """
        Args:
            stem_class (callable): A callable taking 4 arguments (channels in, channels out,
                normalization, callable returning an activation function) that returns another
                callable implementing the stem module.
            stem_width (int): The number of output channels that the stem produces.
            block_class (callable): A callable taking 6 arguments (channels in, channels out,
                stride, normalization, callable returning an activation function, a dict of
                block-specific parameters) that returns another callable implementing the repeated
                block module.
            depths (list[int]): Number of blocks in each stage.
            widths (list[int]): For each stage, the number of output channels of each block.
            group_widths (list[int]): For each stage, the number of channels per group in group
                convolution, if the block uses group convolution.
            strides (list[int]): The stride that each network stage applies to its input.
            bottleneck_ratios (list[float]): For each stage, the ratio of the number of bottleneck
                channels to the number of block input channels (or, equivalently, output channels),
                if the block uses a bottleneck.
            se_ratio (float): The ratio of the number of channels used inside the squeeze-excitation
                (SE) module to it number of input channels, if SE the block uses SE.
            activation_class (callable): A callable taking no arguments that returns another
                callable implementing an activation function.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. RegNet's use "stem" and "s1", "s2", etc for the stages after
                the stem. If None, will return the output of the last layer.
        """
        super().__init__()
        self.stem = stem_class(3, stem_width, norm, activation_class)

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        self.stages_and_names = []
        prev_w = stem_width

        for i, (d, w, s, b, g) in enumerate(
            zip(depths, widths, strides, bottleneck_ratios, group_widths)
        ):
            params = {"bot_mul": b, "group_w": g, "se_r": se_ratio}
            stage = AnyStage(prev_w, w, s, d, block_class, norm, activation_class, params)
            name = "s{}".format(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in stage.children()])
            )
            self._out_feature_channels[name] = list(stage.children())[-1].out_channels
            prev_w = w

        self.apply(init_weights)

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {} does not include {}".format(
                ", ".join(children), out_feature
            )
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Model takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the model. Commonly used in fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this model itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def generate_regnet_parameters(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


class RegNet(AnyNet):
    """RegNet model. See :paper:`dds`."""

    def __init__(
        self,
        *,
        stem_class,
        stem_width,
        block_class,
        depth,
        w_a,
        w_0,
        w_m,
        group_width,
        stride=2,
        bottleneck_ratio=1.0,
        se_ratio=0.0,
        activation_class=None,
        freeze_at=0,
        norm="BN",
        out_features=None,
    ):
        """
        Build a RegNet from the parameterization described in :paper:`dds` Section 3.3.

        Args:
            See :class:`AnyNet` for arguments that are not listed here.
            depth (int): Total number of blocks in the RegNet.
            w_a (float): Factor by which block width would increase prior to quantizing block widths
                by stage. See :paper:`dds` Section 3.3.
            w_0 (int): Initial block width. See :paper:`dds` Section 3.3.
            w_m (float): Parameter controlling block width quantization.
                See :paper:`dds` Section 3.3.
            group_width (int): Number of channels per group in group convolution, if the block uses
                group convolution.
            bottleneck_ratio (float): The ratio of the number of bottleneck channels to the number
                of block input channels (or, equivalently, output channels), if the block uses a
                bottleneck.
            stride (int): The stride that each network stage applies to its input.
        """
        ws, ds = generate_regnet_parameters(w_a, w_0, w_m, depth)[0:2]
        ss = [stride for _ in ws]
        bs = [bottleneck_ratio for _ in ws]
        gs = [group_width for _ in ws]
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)

        def default_activation_class():
            return nn.ReLU(inplace=True)

        super().__init__(
            stem_class=stem_class,
            stem_width=stem_width,
            block_class=block_class,
            depths=ds,
            widths=ws,
            strides=ss,
            group_widths=gs,
            bottleneck_ratios=bs,
            se_ratio=se_ratio,
            activation_class=default_activation_class
            if activation_class is None
            else activation_class,
            freeze_at=freeze_at,
            norm=norm,
            out_features=out_features,
        )
