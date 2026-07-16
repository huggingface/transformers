# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Modular EfficientViT-SAM configuration and modeling components.
"""

import contextlib
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..sam.configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig
from ..sam.image_processing_pil_sam import SamImageProcessorPil
from ..sam.image_processing_sam import SamImageProcessor
from ..sam.modeling_sam import (
    SamImageSegmentationOutput,
    SamMaskDecoder,
    SamModel,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamVisionEncoderOutput,
)
from ..sam.processing_sam import SamProcessor


logger = logging.get_logger(__name__)


# =============================================================================
# 1. Configuration Classes
# =============================================================================


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamPromptEncoderConfig(SamPromptEncoderConfig):
    pass


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamMaskDecoderConfig(SamMaskDecoderConfig):
    pass


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamVisionConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EfficientViTSamVisionModel`]. It is used to
    instantiate an EfficientViT-SAM vision encoder according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        width_list (`list[int]`, *optional*, defaults to `[32, 64, 128, 256, 512]`):
            Channel dimensions for each stage of the backbone.
        depth_list (`list[int]`, *optional*, defaults to `[1, 1, 1, 6, 6]`):
            Number of blocks for each stage of the backbone.
        block_list (`list[str]`, *optional*):
            Type of block to use in each stage of a large backbone (e.g. `["res", "fmb", "fmb", "fmb", "att@3", "att@3"]`).
        expand_list (`list[float]`, *optional*):
            Expand ratios for blocks in each stage of a large backbone.
        fewer_norm_list (`list[bool]`, *optional*):
            Whether to use fewer normalization layers in each stage of a large backbone.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        qkv_dim (`int`, *optional*, defaults to 32):
            Query/Key/Value dimension in the attention layer.
        norm (`str`, *optional*, defaults to `"bn2d"`):
            Type of normalization layer to use.
        act_func (`str`, *optional*, defaults to `"gelu"`):
            Activation function to use.
        fid_list (`list[str]`, *optional*, defaults to `["stage4", "stage3", "stage2"]`):
            Stages from which to aggregate features in the Neck.
        in_channel_list (`list[int]`, *optional*, defaults to `[512, 256, 128]`):
            Channel widths of features aggregated by the Neck.
        head_width (`int`, *optional*, defaults to 256):
            Projection dimension of Neck inputs.
        head_depth (`int`, *optional*, defaults to 8):
            Depth of intermediate layers in the Neck.
        expand_ratio (`float`, *optional*, defaults to 1.0):
            Expansion ratio in the Neck's FusedMBConv blocks.
        middle_op (`str`, *optional*, defaults to `"fmb"`):
            Middle operator for the Neck blocks.
        out_dim (`int`, *optional*, defaults to 256):
            Output feature dimension of the Neck.
        image_size (`int`, *optional*, defaults to 512):
            Image size expected by the model.
    """

    model_type = "efficientvitsam_vision_model"

    def __init__(
        self,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        block_list=None,
        expand_list=None,
        fewer_norm_list=None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=8,
        expand_ratio=1.0,
        middle_op="fmb",
        out_dim=256,
        image_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width_list = width_list
        self.depth_list = depth_list
        self.block_list = block_list
        self.expand_list = expand_list
        self.fewer_norm_list = fewer_norm_list
        self.in_channels = in_channels
        self.qkv_dim = qkv_dim
        self.norm = norm
        self.act_func = act_func
        self.fid_list = fid_list
        self.in_channel_list = in_channel_list
        self.head_width = head_width
        self.head_depth = head_depth
        self.expand_ratio = expand_ratio
        self.middle_op = middle_op
        self.out_dim = out_dim
        self.image_size = image_size
        self.num_pos_feats = kwargs.get("num_pos_feats", 128)
        self.scale = kwargs.get("scale", 128.0)


@auto_docstring(checkpoint="mit-han-lab/efficientvit-sam-l1")
@strict
class EfficientViTSamConfig(SamConfig):
    r"""
    [`EfficientViTSamConfig`] is the configuration class to store the configuration of a [`EfficientViTSamModel`]. It is
    used to instantiate an EfficientViT-SAM model according to the specified arguments, defining the vision encoder,
    prompt encoder and mask decoder configs.

    Configuration objects inherit from [`SamConfig`] and can be used to control the model outputs. Read the
    documentation from [`SamConfig`] for more information.

    Args:
        vision_config (Union[`dict`, `EfficientViTSamVisionConfig`], *optional*):
            Dictionary of configuration options or an `EfficientViTSamVisionConfig` object used to initialize the
            vision encoder.
        prompt_encoder_config (Union[`dict`, `EfficientViTSamPromptEncoderConfig`], *optional*):
            Dictionary of configuration options or a `EfficientViTSamPromptEncoderConfig` object used to initialize the prompt
            encoder.
        mask_decoder_config (Union[`dict`, `EfficientViTSamMaskDecoderConfig`], *optional*):
            Dictionary of configuration options or a `EfficientViTSamMaskDecoderConfig` object used to initialize the mask
            decoder.
    """

    model_type = "efficientvitsam"
    sub_configs = {
        "prompt_encoder_config": EfficientViTSamPromptEncoderConfig,
        "mask_decoder_config": EfficientViTSamMaskDecoderConfig,
        "vision_config": EfficientViTSamVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = EfficientViTSamVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = EfficientViTSamVisionConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = EfficientViTSamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = EfficientViTSamPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = EfficientViTSamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = EfficientViTSamMaskDecoderConfig()

        super().__post_init__(**kwargs)


# =============================================================================
# 1.5. Output and Auxiliary Classes from SAM
# =============================================================================


class EfficientViTSamVisionEncoderOutput(SamVisionEncoderOutput):
    pass


class EfficientViTSamImageSegmentationOutput(SamImageSegmentationOutput):
    pass


class EfficientViTSamPositionalEmbedding(SamPositionalEmbedding):
    pass


class EfficientViTSamPromptEncoder(SamPromptEncoder):
    pass


class EfficientViTSamMaskDecoder(SamMaskDecoder):
    pass


# =============================================================================
# 2. Helpers and Basic Layers
# =============================================================================


def val2list(x: list | tuple | Any, repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name: str, num_features: int, **kwargs) -> nn.Module:
    if name == "bn2d":
        return nn.BatchNorm2d(num_features, **kwargs)
    elif name == "ln":
        return nn.LayerNorm(num_features, **kwargs)
    elif name == "ln2d":
        return LayerNorm2d(num_features, **kwargs)
    else:
        return nn.Identity()


def build_act(name: str | None) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    elif name == "relu6":
        return nn.ReLU6()
    elif name == "hswish":
        return nn.Hardswish()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU(approximate="tanh")
    else:
        return nn.Identity()


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0.0,
        norm: str | None = "bn2d",
        act_func: str | None = "relu",
    ):
        super().__init__()
        padding = get_same_padding(kernel_size)
        if isinstance(padding, int):
            padding *= dilation
        else:
            padding = tuple(p * dilation for p in padding)

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels) if norm else None
        self.act = build_act(act_func) if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        size: int | tuple[int, int] | None = None,
        factor: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == tuple(self.size)) or self.factor == 1:
            return x
        # Cast to float32 if half precision to avoid interpolation overflow issues
        original_dtype = x.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            x = x.float()

        out = F.interpolate(
            x,
            size=self.size,
            scale_factor=self.factor,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ["bilinear", "bicubic"] else None,
        )
        return out.to(original_dtype)


# =============================================================================
# 3. Model Architecture Blocks
# =============================================================================


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module | None,
        shortcut: nn.Module | None,
        post_act: str | None = None,
        pre_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act) if post_act else None

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool | tuple[bool, ...] = False,
        norm: tuple[str | None, str | None] = ("bn2d", "bn2d"),
        act_func: tuple[str | None, str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 6.0,
        use_bias: bool | tuple[bool, ...] = False,
        norm: tuple[str | None, ...] = ("bn2d", "bn2d", "bn2d"),
        act_func: tuple[str | None, ...] = ("relu6", "relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 6.0,
        groups: int = 1,
        use_bias: bool | tuple[bool, ...] = False,
        norm: tuple[str | None, str | None] = ("bn2d", "bn2d"),
        act_func: tuple[str | None, str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: int | None = None,
        expand_ratio: float = 1.0,
        use_bias: bool | tuple[bool, ...] = False,
        norm: tuple[str | None, str | None] = ("bn2d", "bn2d"),
        act_func: tuple[str | None, str | None] = ("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int | None = None,
        heads_ratio: float = 1.0,
        dim: int = 8,
        use_bias: bool | tuple[bool, ...] = False,
        norm: tuple[str | None, str | None] = (None, "bn2d"),
        act_func: tuple[str | None, str | None] = (None, None),
        kernel_func: str = "relu",
        scales: tuple[int, ...] = (5,),
        eps: float = 1.0e-15,
    ):
        super().__init__()
        self.eps = eps
        heads = int(in_channels // dim * heads_ratio) if heads is None else heads
        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())
        device_type = qkv.device.type
        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )

        with autocast_context:
            if qkv.dtype in [torch.float16, torch.bfloat16]:
                qkv = qkv.float()

            qkv = torch.reshape(
                qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
            q, k, v = (
                qkv[:, :, 0 : self.dim],
                qkv[:, :, self.dim : 2 * self.dim],
                qkv[:, :, 2 * self.dim :],
            )

            # lightweight linear attention
            q = self.kernel_func(q)
            k = self.kernel_func(k)

            trans_k = k.transpose(-1, -2)

            v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)
            vk = torch.matmul(v, trans_k)
            out = torch.matmul(vk, q)
            out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

            out = torch.reshape(out, (B, -1, H, W))
            return out

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())
        device_type = qkv.device.type
        autocast_context = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in {"cuda", "cpu"}
            else contextlib.nullcontext()
        )

        with autocast_context:
            if qkv.dtype in [torch.float16, torch.bfloat16]:
                qkv = qkv.float()

            qkv = torch.reshape(
                qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
            q, k, v = (
                qkv[:, :, 0 : self.dim],
                qkv[:, :, self.dim : 2 * self.dim],
                qkv[:, :, 2 * self.dim :],
            )

            q = self.kernel_func(q)
            k = self.kernel_func(k)

            att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
            att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
            out = torch.matmul(v, att_map)  # b h d n

            out = torch.reshape(out, (B, -1, H, W))
            return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if self.dim < H * W:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv).to(qkv.dtype)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        expand_ratio: float = 4.0,
        scales: tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                scales=scales,
            ),
            nn.Identity(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm=(None, None, norm),
                act_func=(act_func, act_func, None),
            ),
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


# =============================================================================
# 4. Backbone and Neck Assembly
# =============================================================================


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: list[str] | None = None,
        expand_list: list[float] | None = None,
        fewer_norm_list: list[bool] | None = None,
        in_channels: int = 3,
        qkv_dim: int = 32,
        norm: str = "bn2d",
        act_func: str = "gelu",
    ):
        super().__init__()
        block_list = ["res", "fmb", "fmb", "mb", "att"] if block_list is None else block_list
        expand_list = [1.0, 4.0, 4.0, 4.0, 6.0] if expand_list is None else expand_list
        fewer_norm_list = [False, False, False, True, True] if fewer_norm_list is None else fewer_norm_list

        self.width_list = []
        self.stages = nn.ModuleList()

        # stage 0 (Stem)
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, nn.Identity()))
        in_channels = width_list[0]
        self.stages.append(nn.Sequential(*stage0))
        self.width_list.append(in_channels)

        # stages 1-N
        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4.0,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    stage.append(ResidualBlock(block, nn.Identity()))
            self.stages.append(nn.Sequential(*stage))
            self.width_list.append(in_channels)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            return ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            return FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            return MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(f"Unknown block type: {block}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict[f"stage{stage_id}"] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


class SamNeck(nn.Module):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        out_dim: int = 256,
        norm: str = "bn2d",
        act_func: str = "gelu",
    ):
        super().__init__()
        self.fid_list = fid_list

        self.proj_layers = nn.ModuleDict()
        for fid, in_channel in zip(fid_list, in_channel_list):
            self.proj_layers[fid] = nn.Sequential(
                ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                UpSampleLayer(size=(64, 64)),
            )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mb":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmb":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            elif middle_op == "res":
                block = ResBlock(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError(f"Neck operator {middle_op} is not supported.")
            middle.append(ResidualBlock(block, nn.Identity()))
        self.middle = nn.Sequential(*middle)

        self.proj_out = ConvLayer(
            head_width,
            out_dim,
            1,
            use_bias=True,
            norm=None,
            act_func=None,
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = [self.proj_layers[fid](features[fid]) for fid in self.fid_list]
        fused = sum(projected)
        fused = self.middle(fused)
        out = self.proj_out(fused)
        features["sam_encoder"] = out
        return features


class EfficientViTSamPreTrainedModel(PreTrainedModel):
    config_class = EfficientViTSamConfig
    base_model_prefix = "efficientvitsam"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, LayerNorm2d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, "positional_embedding") and hasattr(module, "scale"):
            nn.init.normal_(module.positional_embedding, std=module.scale)


class EfficientViTSamImageEncoder(EfficientViTSamPreTrainedModel):
    def __init__(self, config: EfficientViTSamVisionConfig):
        super().__init__(config)
        self.backbone = EfficientViTLargeBackbone(
            width_list=config.width_list,
            depth_list=config.depth_list,
            block_list=config.block_list,
            expand_list=config.expand_list,
            fewer_norm_list=config.fewer_norm_list,
            in_channels=config.in_channels,
            qkv_dim=config.qkv_dim,
            norm=config.norm,
            act_func=config.act_func,
        )
        self.neck = SamNeck(
            fid_list=config.fid_list,
            in_channel_list=config.in_channel_list,
            head_width=config.head_width,
            head_depth=config.head_depth,
            expand_ratio=config.expand_ratio,
            middle_op=config.middle_op,
            out_dim=config.out_dim,
            norm=config.norm,
            act_func=config.act_func,
        )
        self.norm = build_norm("ln2d", config.out_dim)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | EfficientViTSamVisionEncoderOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        features = self.backbone(pixel_values)
        features = self.neck(features)
        output = self.norm(features["sam_encoder"])

        if not return_dict:
            return (output,)

        return EfficientViTSamVisionEncoderOutput(last_hidden_state=output)


class EfficientViTSamVisionModel(EfficientViTSamPreTrainedModel):
    config_class = EfficientViTSamVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: EfficientViTSamVisionConfig):
        super().__init__(config)
        self.vision_encoder = EfficientViTSamImageEncoder(config)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> EfficientViTSamVisionEncoderOutput:
        return self.vision_encoder(pixel_values)


class EfficientViTSamModel(SamModel):
    config_class = EfficientViTSamConfig

    def __init__(self, config: EfficientViTSamConfig):
        super(SamModel, self).__init__(config)
        self.shared_image_embedding = EfficientViTSamPositionalEmbedding(config.vision_config)
        self.vision_encoder = EfficientViTSamImageEncoder(config.vision_config)
        self.prompt_encoder = EfficientViTSamPromptEncoder(config)

        config.mask_decoder_config._attn_implementation = config._attn_implementation
        self.mask_decoder = EfficientViTSamMaskDecoder(config.mask_decoder_config)
        self.post_init()


class EfficientViTSamImageProcessor(SamImageProcessor):
    size = {"longest_edge": 512}
    pad_size = {"height": 512, "width": 512}


class EfficientViTSamImageProcessorPil(SamImageProcessorPil):
    size = {"longest_edge": 512}
    pad_size = {"height": 512, "width": 512}


NestedList = list[Union[float, int, None, "NestedList"]]


class EfficientViTSamProcessor(SamProcessor):
    # Dummy reference to NestedList to ensure modular compiler includes its definition
    _nested_list_type = NestedList


__all__ = [
    "EfficientViTSamPromptEncoderConfig",
    "EfficientViTSamMaskDecoderConfig",
    "EfficientViTSamVisionConfig",
    "EfficientViTSamConfig",
    "EfficientViTSamVisionEncoderOutput",
    "EfficientViTSamImageSegmentationOutput",
    "EfficientViTSamPositionalEmbedding",
    "EfficientViTSamPromptEncoder",
    "EfficientViTSamMaskDecoder",
    "EfficientViTSamImageEncoder",
    "EfficientViTSamVisionModel",
    "EfficientViTSamModel",
    "EfficientViTSamImageProcessor",
    "EfficientViTSamImageProcessorPil",
    "EfficientViTSamProcessor",
    "NestedList",
]
