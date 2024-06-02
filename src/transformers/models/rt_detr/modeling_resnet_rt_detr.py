"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""

import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.timm_backbone.configuration_timm_backbone import TimmBackboneConfig

from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available, requires_backends
from ...utils.backbone_utils import BackboneMixin


if is_timm_available():
    import timm
    from timm.layers import (
        AvgPool2dSame,
        DropBlock2d,
        DropPath,
        LayerType,
        create_attn,
        create_classifier,
        get_act_layer,
        get_norm_layer,
    )
    from timm.models._builder import build_model_with_cfg
    from timm.models._features import feature_take_indices
    from timm.models._manipulate import checkpoint_seq


if is_torch_available():
    from torch import Tensor


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        attn_layer: Optional[Type[nn.Module]] = None,
        aa_layer: Optional[Type[nn.Module]] = None,
        drop_block: Optional[Type[nn.Module]] = None,
        drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
        )
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, "weight", None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
        reduce_first: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        attn_layer: Optional[Type[nn.Module]] = None,
        aa_layer: Optional[Type[nn.Module]] = None,
        drop_block: Optional[Type[nn.Module]] = None,
        drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=1 if use_aa else stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    first_dilation: Optional[int] = None,
    norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        *[
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False
            ),
            norm_layer(out_channels),
        ]
    )


def downsample_avg(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    first_dilation: Optional[int] = None,
    norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(
        *[pool, nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False), norm_layer(out_channels)]
    )


def drop_blocks(drop_prob: float = 0.0):
    return [
        None,
        None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None,
    ]


def make_blocks(
    block_fn: Union[BasicBlock, Bottleneck],
    channels: Tuple[int, ...],
    block_repeats: Tuple[int, ...],
    inplanes: int,
    reduce_first: int = 1,
    output_stride: int = 32,
    down_kernel_size: int = 1,
    avg_down: bool = False,
    drop_block_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f"layer{stage_idx + 1}"  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        # Fixed part due to compatibility with paddle resnet18, renset34
        # if stride != 1 or inplanes != planes * block_fn.expansion:
        down_kwargs = {
            "in_channels": inplanes,
            "out_channels": planes * block_fn.expansion,
            "kernel_size": down_kernel_size,
            "stride": stride,
            "dilation": dilation,
            "first_dilation": prev_dilation,
            "norm_layer": kwargs.get("norm_layer"),
        }
        downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = {"reduce_first": reduce_first, "dilation": dilation, "drop_block": db, **kwargs}
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None,
                    **block_kwargs,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append({"num_chs": inplanes, "reduction": net_stride, "module": stage_name})

    return stages, feature_info


# Inspired by https://github.com/huggingface/pytorch-image-models/blob/5dce71010174ad6599653da4e8ba37fd5f9fa572/timm/models/resnet.py#L363
class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
        self,
        block: Union[BasicBlock, Bottleneck],
        layers: Tuple[int, ...],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        global_pool: str = "avg",
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        replace_stem_pool: bool = False,
        block_reduce_first: int = 1,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        act_layer: Optional[Type[nn.Module]] = nn.ReLU,
        norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        aa_layer: Optional[Type[nn.Module]] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        zero_init_last: bool = True,
        block_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or {}
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if "tiered" in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False),
                ]
            )
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [{"num_chs": inplanes, "reduction": 2, "module": "act1"}]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(
                *filter(
                    None,
                    [
                        nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                        create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                        norm_layer(inplanes),
                        act_layer(inplace=True),
                    ],
                )
            )
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(
                        *[nn.MaxPool2d(kernel_size=3, stride=1, padding=1), aa_layer(channels=inplanes, stride=2)]
                    )
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = (64, 128, 256, 512)
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, "zero_init_last"):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = {"stem": r"^conv1|bn1|maxpool", "blocks": r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)"}
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ("NCHW",), "Output shape must be NCHW."
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if feat_idx in take_indices:
            intermediates.append(x)
        x = self.maxpool(x)

        layer_names = ("layer1", "layer2", "layer3", "layer4")
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int], Tuple[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(5, indices)
        layer_names = ("layer1", "layer2", "layer3", "layer4")
        layer_names = layer_names[max_index:]
        for n in layer_names:
            setattr(self, n, nn.Identity())
        if prune_head:
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# Copied from transformers.models.timm_backbone.modeling_timm_backbone.TimmBackbone
class RTDETRTimmBackbone(PreTrainedModel, BackboneMixin):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """

    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = TimmBackboneConfig

    # Ignore copy
    def __init__(self, config, **kwargs):
        requires_backends(self, "timm")
        super().__init__(config)
        self.config = config

        if config.backbone is None:
            raise ValueError("backbone is not set in the config. Please set it to a timm model name.")

        if config.backbone not in timm.list_models():
            raise ValueError(f"backbone {config.backbone} is not supported by timm.")

        if hasattr(config, "out_features") and config.out_features is not None:
            raise ValueError("out_features is not supported by TimmBackbone. Please use out_indices instead.")

        pretrained = getattr(config, "use_pretrained_backbone", None)
        if pretrained is None:
            raise ValueError("use_pretrained_backbone is not set in the config. Please set it to True or False.")

        # We just take the final layer by default. This matches the default for the transformers models.
        out_indices = config.out_indices if getattr(config, "out_indices", None) is not None else (-1,)

        in_chans = kwargs.pop("in_chans", config.num_channels)

        # This is currently not possible for transformer architectures.
        if config.backbone == "resnet18d":
            model_args = {
                "block": BasicBlock,
                "layers": (2, 2, 2, 2),
                "stem_width": 32,
                "stem_type": "deep",
                "avg_down": True,
            }
        elif config.backbone == "resnet34d":
            model_args = {
                "block": BasicBlock,
                "layers": (3, 4, 6, 3),
                "stem_width": 32,
                "stem_type": "deep",
                "avg_down": True,
            }
        else:
            model_args = {}
        kwargs = {
            "features_only": config.features_only,
            "in_chans": in_chans,
            "out_indices": out_indices,
            **kwargs,
        }
        self._backbone = build_model_with_cfg(ResNet, config.backbone, pretrained, **dict(model_args, **kwargs))

        # Converts all `BatchNorm2d` and `SyncBatchNorm` or `BatchNormAct2d` and `SyncBatchNormAct2d` layers of provided module into `FrozenBatchNorm2d` or `FrozenBatchNormAct2d` respectively
        if getattr(config, "freeze_batch_norm_2d", False):
            self.freeze_batch_norm_2d()

        # These are used to control the output of the model when called. If output_hidden_states is True, then
        # return_layers is modified to include all layers.
        self._return_layers = {
            layer["module"]: str(layer["index"]) for layer in self._backbone.feature_info.get_dicts()
        }
        self._all_layers = {layer["module"]: str(i) for i, layer in enumerate(self._backbone.feature_info.info)}
        super()._init_backbone(config)

    # Ignore copy
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        requires_backends(cls, ["vision", "timm"])
        from ...models.timm_backbone import TimmBackboneConfig

        config = kwargs.pop("config", TimmBackboneConfig())

        use_timm = kwargs.pop("use_timm_backbone", True)
        if not use_timm:
            raise ValueError("use_timm_backbone must be True for timm backbones")

        num_channels = kwargs.pop("num_channels", config.num_channels)
        features_only = kwargs.pop("features_only", config.features_only)
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        out_indices = kwargs.pop("out_indices", config.out_indices)
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )
        return super()._from_config(config, **kwargs)

    def freeze_batch_norm_2d(self):
        timm.utils.model.freeze_batch_norm_2d(self._backbone)

    def unfreeze_batch_norm_2d(self):
        timm.utils.model.unfreeze_batch_norm_2d(self._backbone)

    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        pass

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BackboneOutput, Tuple[Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot output attentions for timm backbones at the moment")

        if output_hidden_states:
            # We modify the return layers to include all the stages of the backbone
            self._backbone.return_layers = self._all_layers
            hidden_states = self._backbone(pixel_values, **kwargs)
            self._backbone.return_layers = self._return_layers
            feature_maps = tuple(hidden_states[i] for i in self.out_indices)
        else:
            feature_maps = self._backbone(pixel_values, **kwargs)
            hidden_states = None

        feature_maps = tuple(feature_maps)
        hidden_states = tuple(hidden_states) if hidden_states is not None else None

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return output

        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)


def load_rt_detr_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    """
    from transformers import AutoConfig

    backbone_config = getattr(config, "backbone_config", None)
    use_timm_backbone = getattr(config, "use_timm_backbone", None)
    use_pretrained_backbone = getattr(config, "use_pretrained_backbone", None)
    backbone_checkpoint = getattr(config, "backbone", None)
    backbone_kwargs = getattr(config, "backbone_kwargs", None)
    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

    # If there is a backbone_config and a backbone checkpoint, and use_pretrained_backbone=False then the desired
    # behaviour is ill-defined: do you want to load from the checkpoint's config or the backbone_config?
    if backbone_config is not None and backbone_checkpoint is not None and use_pretrained_backbone is not None:
        raise ValueError("Cannot specify both config.backbone_config and config.backbone")

    # If any of thhe following are set, then the config passed in is from a model which contains a backbone.
    if (
        backbone_config is None
        and use_timm_backbone is None
        and backbone_checkpoint is None
        and backbone_checkpoint is None
    ):
        return RTDETRTimmBackbone.from_config(config=config, **backbone_kwargs)

    # config from the parent model that has a backbone
    if use_timm_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_timm_backbone is True")
        # Because of how timm backbones were originally added to models, we need to pass in use_pretrained_backbone
        # to determine whether to load the pretrained weights.
        backbone = RTDETRTimmBackbone.from_pretrained(
            backbone_checkpoint,
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            **backbone_kwargs,
        )
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_pretrained_backbone is True")
        backbone = RTDETRTimmBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            raise ValueError("Either config.backbone_config or config.backbone must be set")
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        backbone = RTDETRTimmBackbone.from_config(config=backbone_config)
    return backbone
