import warnings
from distutils.version import LooseVersion
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


try:
    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .forward_intermediates import forward_intermediates
from .input_conditioner import InputConditioner


_has_torch_sdpa = hasattr(F, "scaled_dot_product_attention")


class PaliGemmaWrapper(nn.Module):
    def __init__(self, vis_model: nn.Module, embed_dim: int):
        super().__init__()

        self.vis_model = vis_model
        self.embed_dim = embed_dim

    @property
    def patch_size(self):
        return self.vis_model.embeddings.patch_size

    @property
    def blocks(self):
        return self.vis_model.encoder.layers

    @property
    def embed_dim(self):
        return self.vis_model.embeddings.embed_dim

    def forward(self, x: torch.Tensor):
        outputs = self.vis_model(
            x,
            return_dict=False,
            interpolate_pos_encoding=True,
        )

        features = outputs[0].to(torch.float32)

        summary = features.mean(dim=1)

        return summary, features

    def forward_features(self, x: torch.Tensor):
        return self(x)


def _get_paligemma_model(repo: str, embed_dim: int = None, dtype: torch.dtype = torch.bfloat16):
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import __version__ as tx_version

    if LooseVersion(tx_version) > LooseVersion("4.44.2"):
        warnings.warn(
            f'Your transformers version "{tx_version}" is higher than 4.44.2, and for whatever reason, PaliGemma might be broken.'
        )

    extra_args = dict()

    if dtype is not None:
        extra_args["torch_dtype"] = dtype
        rev = str(dtype).split(".")[-1]
        extra_args["revision"] = rev

    model = PaliGemmaForConditionalGeneration.from_pretrained(repo, **extra_args)

    vis_model = model.vision_tower.vision_model

    vis_model = PaliGemmaWrapper(vis_model, embed_dim)

    return vis_model


@register_model
def paligemma_896_student(**kwargs):
    model = _get_paligemma_model("google/paligemma-3b-pt-896", embed_dim=1152, dtype=None)

    return model


def dv2_sdpa(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        dropout_p=self.attn_drop.p if self.training else 0.0,
        scale=self.scale,
    )
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _load_dino_v2(dino_v2_model, cache_dir: str | None = None, pretrained=True, **kwargs):
    if cache_dir:
        torch.hub.set_dir(cache_dir)
    model: nn.Module = torch.hub.load(
        "facebookresearch/dinov2",
        dino_v2_model,
        pretrained=pretrained,
        # **kwargs,
    )

    if _has_torch_sdpa:
        for n, m in model.named_modules():
            if n.endswith(".attn"):
                m.forward = MethodType(dv2_sdpa, m)

    return model


class DinoWrapper(nn.Module):
    def __init__(self, dino_model: nn.Module):
        super().__init__()

        self.inner = dino_model
        dino_model.blocks = nn.Sequential(*dino_model.blocks)

    @property
    def embed_dim(self):
        return self.inner.embed_dim

    @property
    def patch_size(self):
        return self.inner.patch_size

    @property
    def num_cls_tokens(self):
        return getattr(self.inner, "num_tokens", 1)

    @property
    def num_registers(self):
        return getattr(self.inner, "num_register_tokens", 0)

    @property
    def num_summary_tokens(self):
        return self.num_cls_tokens + self.num_registers

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.inner.forward_features(*args, **kwargs)

        cls_token = parts["x_norm_clstoken"]
        features = parts["x_norm_patchtokens"]

        return cls_token, features

    def forward_features(self, x: torch.Tensor):
        x = self.inner.prepare_tokens_with_masks(x)
        x = self.inner.blocks(x)
        x_norm = self.inner.norm(x)

        return x_norm[:, 0], x_norm[:, self.num_summary_tokens :]

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner.prepare_tokens_with_masks(x)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        norm: bool = False,
        **kwargs,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        return forward_intermediates(
            self,
            patch_extractor=self.inner.prepare_tokens_with_masks,
            num_summary_tokens=self.num_summary_tokens,
            num_cls_tokens=self.num_cls_tokens,
            norm=self.inner.norm if norm else lambda y: y,
            x=x,
            **kwargs,
        )


def _dino_student(arch: str, **kwargs):
    from . import dinov2_arch

    factory = getattr(dinov2_arch, arch)
    model = factory()

    model = DinoWrapper(model)

    conditioner = InputConditioner(
        input_scale=1.0,
        norm_mean=IMAGENET_DEFAULT_MEAN,
        norm_std=IMAGENET_DEFAULT_STD,
    )

    model.input_conditioner = conditioner

    return model


@register_model
def dino_v2_l_student(**kwargs):
    return _dino_student("dinov2_vitl14_reg", **kwargs)


@register_model
def dino_v2_g_student(**kwargs):
    return _dino_student("dinov2_vitg14_reg", **kwargs)
