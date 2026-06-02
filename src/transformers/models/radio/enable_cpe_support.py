# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from contextlib import contextmanager
from types import MethodType
from typing import Optional

import torch
from timm.models import VisionTransformer, checkpoint_seq
from torch import nn

from .dual_hybrid_vit import HybridModel
from .extra_models import DinoWrapper
from .forward_intermediates import forward_intermediates
from .vit_patch_generator import ViTPatchGenerator


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


@contextmanager
def _video_mode(self: VisionTransformer, t: int):
    """
    Context manager to temporarily set the model in video mode.
    This is used to handle models that support both image and video inputs.
    """
    original_num_frames = self.patch_generator.num_video_frames
    self.patch_generator.num_video_frames = t
    try:
        yield
    finally:
        self.patch_generator.num_video_frames = original_num_frames


def _take_indices(
    num_blocks: int,
    n: int | list[int] | tuple[int] | None,
) -> tuple[set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def _forward_intermediates_cpe(
    self,
    x: torch.Tensor,
    norm: bool = False,
    **kwargs,
) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
    return forward_intermediates(
        self,
        patch_extractor=self.patch_generator,
        num_summary_tokens=self.patch_generator.num_skip,
        num_cls_tokens=self.patch_generator.num_cls_tokens,
        norm=self.norm if norm else lambda y: y,
        x=x,
        **kwargs,
    )


def _forward_cpe_dinov2(self: DinoWrapper, x: torch.Tensor) -> torch.Tensor:
    y = _forward_cpe(self.inner, x)

    return y[:, 0], y[:, self.num_summary_tokens :]


def _forward_intermediates_cpe_dinov2(self: DinoWrapper, *args, **kwargs):
    return _forward_intermediates_cpe(self.inner, *args, **kwargs)


def _enable_cpe_for_timm_vit(
    model: VisionTransformer,
    max_img_size: int | tuple[int, int] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int = Optional[None],
    num_registers: int = Optional[None],
):
    if not isinstance(model, VisionTransformer):
        raise ValueError("CPE only support for VisionTransformer models!")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
    )

    model.patch_generator = patch_generator
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = patch_generator.num_registers

    model.forward_features = MethodType(_forward_cpe, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe, model)


def _enable_cpe_for_dv2_reg_vit(
    model: DinoWrapper,
    max_img_size: int | tuple[int, int] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: int = Optional[None],
    num_registers: int = Optional[None],
):
    patch_size = model.patch_size
    embed_dim = model.embed_dim
    input_dims = model.inner.patch_embed.patches_resolution
    normalize_patches = not isinstance(model.inner.patch_embed.norm, nn.Identity)
    cls_token = True

    max_img_size = int(round(max_img_size / patch_size) * patch_size)

    patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
        patch_bias=True,
    )

    inner = model.inner
    inner.patch_generator = patch_generator
    inner.patch_embed = None
    inner.cls_token = None
    inner.pos_embed = None
    inner.register_tokens = None
    inner.patch_size = patch_size

    model.forward_features = MethodType(_forward_cpe_dinov2, model)
    model.forward_intermediates = MethodType(_forward_intermediates_cpe_dinov2, model)


def enable_cpe(
    model: nn.Module,
    *args,
    **kwargs,
):
    if isinstance(model, VisionTransformer):
        _enable_cpe_for_timm_vit(model, *args, **kwargs)
    elif isinstance(model, DinoWrapper):
        _enable_cpe_for_dv2_reg_vit(model, *args, **kwargs)
    elif isinstance(model, HybridModel):
        _enable_cpe_for_timm_vit(model.vit, *args, **kwargs)
    else:
        raise ValueError(f"CPE not supported for this model type: {type(model)}")

    model.cpe_video_mode = MethodType(_video_mode, model)
