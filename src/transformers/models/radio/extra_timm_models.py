# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import warnings

import torch
from timm.models import PretrainedCfg, register_model
from timm.models.vision_transformer import (
    Block,
    Mlp,
    VisionTransformer,
)
from timm.models.vision_transformer import (
    LayerScale as TIMMLayerScale,
)
from timm.models.vision_transformer import (
    _create_vision_transformer as _timm_create_vision_transformer,
)
from torch import nn
from torch.nn import functional as F

# Import these to also register them
from . import dinov2_arch


@register_model
def vit_tiny_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Tiny (Vit-Ti/16)"""
    model_args = dict(patch_size=14, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer("vit_tiny_patch14_224", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer("vit_small_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer("vit_base_patch14_224", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_v2_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        reg_tokens=4,
        no_embed_class=True,
        img_size=518 * 16 // 14,
    )
    model = _create_vision_transformer("vit_base_patch14_reg4_dinov2", pretrained=False, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_v2_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    name = "vit_large_patch14_reg4_dinov2"
    model_args = dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        reg_tokens=4,
        no_embed_class=True,
        img_size=518 * 16 // 14,
    )
    model = _create_vision_transformer(name, pretrained=False, **dict(model_args, **kwargs))

    return model


@register_model
def vit_so400m_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT model matching the architecture of the So400M model from
    "Scaling Vision Transformers to 400 Million Parameters" (https://arxiv.org/abs/2302.05442).
    """
    if pretrained:
        raise ValueError("There is no pretrained weights for vit_so400m_patch16_224")
    mlp_ratio = 4304 / 1152

    model_args = dict(patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=mlp_ratio)
    model = _create_vision_transformer("vit_so400m_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_v2_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT model matching the architecture of the So400M model from
    "Scaling Vision Transformers to 400 Million Parameters" (https://arxiv.org/abs/2302.05442).
    """
    if pretrained:
        raise ValueError("There is no pretrained weights for vit_so400m_patch16_224")

    normal_target = 4304
    # TP4 requires channels to be a multiple of 4, and then within that, FP8 requires a multiple of 8,
    # thus, a multiple of 32 is required.
    tp4_fp8_safe_target = ((normal_target + 31) // 32) * 32

    mlp_ratio = tp4_fp8_safe_target / 1152

    model_args = dict(patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=mlp_ratio)
    model = _create_vision_transformer(
        "vit_so400m_v2_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


@register_model
def vit_huge_patch16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_args = dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16)
    if pretrained:
        # There is no pretrained version of ViT-H/16, but we can adapt a ViT-H/14 for this purpose
        model = _create_vision_transformer("vit_huge_patch14_224", pretrained=True, **dict(model_args, **kwargs))
    else:
        model = _create_vision_transformer("vit_huge_patch16_224", pretrained=False, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch16_224_mlpnorm(pretrained=False, **kwargs) -> VisionTransformer:
    """ViT-Huge model (ViT-H/16) from original paper (https://arxiv.org/abs/2010.11929)."""
    model = vit_huge_patch16_224(pretrained=pretrained, **kwargs)

    for m in model.modules():
        if isinstance(m, Mlp) and not isinstance(m.norm, nn.LayerNorm):
            m.norm = nn.LayerNorm(m.fc1.out_features)

    return model


@register_model
def vit_giant_patch16_224(pretrained=False, scaled_ln: bool = False, **kwargs) -> VisionTransformer:
    """ViT-giant model (ViT-g/16) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_args = dict(patch_size=16, embed_dim=1536, depth=40, num_heads=24)
    model = _create_vision_transformer("vit_giant_patch16_224", pretrained=False, **dict(model_args, **kwargs))
    if scaled_ln:
        _apply_scaled_ln(model)
    return model


@register_model
def vit_bigG_patch14_224(pretrained=False, **kwargs) -> VisionTransformer:
    model_args = dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, init_values=1e-6)
    model = _create_vision_transformer("vit_bigG_patch14", pretrained=False, **dict(model_args, **kwargs))
    return model


def _create_vision_transformer(*args, **kwargs):
    if kwargs.get("pretrained_cfg") is None:
        # This prevents the warning from being emitted
        kwargs["pretrained_cfg"] = PretrainedCfg()

    model = _timm_create_vision_transformer(*args, **kwargs)
    _patch_layer_scale(model)
    return model


def _patch_layer_scale(model: VisionTransformer):
    def replace_ls(old_ls: TIMMLayerScale):
        new_ls = dinov2_arch.LayerScale(old_ls.gamma.shape[0], inplace=old_ls.inplace)
        new_ls.load_state_dict(old_ls.state_dict())
        return new_ls

    # Monkey patch: Replace TIMM's LayerScale with our modified DINOv2 one, that uses a param name
    # other than gamma, so that HFHub doesn't mess with it!
    for mod in model.modules():
        if isinstance(mod, Block):
            if isinstance(mod.ls1, TIMMLayerScale):
                mod.ls1 = replace_ls(mod.ls1)
            if isinstance(mod.ls2, TIMMLayerScale):
                mod.ls2 = replace_ls(mod.ls2)
    pass


class ScaledLayerNorm(nn.LayerNorm):
    """
    https://arxiv.org/pdf/2502.05795v1
    """

    def __init__(self, ln_base: nn.LayerNorm, depth: int = 0):
        super().__init__(ln_base.normalized_shape, eps=ln_base.eps, elementwise_affine=ln_base.elementwise_affine)
        self.load_state_dict(ln_base.state_dict())
        self.register_buffer("ln_scale", torch.tensor(1.0 / math.sqrt(depth)), persistent=False)

    def forward(self, x):
        y = super().forward(x)
        y = y * self.ln_scale
        return y


class DyT(nn.Module):
    def __init__(self, C: int, init_alpha: float):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), init_alpha))
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))

    def forward(self, x: torch.Tensor):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta


@register_model
def vit_large_dyt_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer(
        "vit_large_dyt_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )

    def _replace_ln_with_dyt(ln: nn.LayerNorm, depth: int):
        return DyT(ln.normalized_shape[0], init_alpha=0.9)

    _replace_ln(model, _replace_ln_with_dyt)

    return model


def _apply_scaled_ln(model: VisionTransformer):
    warnings.warn("Post-LayerNorm scaling activated!")

    _replace_ln(model, lambda ln, depth: ScaledLayerNorm(ln, depth=depth))


def _replace_ln(model: VisionTransformer, fn):
    def _inner_replace_ln(block: Block, depth: int, key: str):
        prev = getattr(block, key)
        if isinstance(prev, nn.LayerNorm):
            setattr(block, key, fn(prev, depth=depth))

    for i, block in enumerate(model.blocks):
        _inner_replace_ln(block, i + 1, "norm1")
        _inner_replace_ln(block, i + 1, "norm2")
