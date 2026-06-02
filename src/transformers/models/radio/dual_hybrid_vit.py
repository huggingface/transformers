import torch
from einops import rearrange
from timm.models import convnext as tconv
from timm.models import register_model
from timm.models import vision_transformer as tvit
from torch import nn
from torch.nn import functional as F

from . import extra_timm_models as et


class Fuser(nn.Module):
    def __init__(self, src_dim: int, tgt_dim: int, gated: bool = True):
        super().__init__()
        self.gated = gated

        mid_dim = max(src_dim, tgt_dim) * 2

        self.fwd = nn.Sequential(
            nn.Conv2d(src_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, tgt_dim * (2 if gated else 1), kernel_size=3, stride=1, padding=1),
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if src.ndim == 3:
            shape = tgt.shape[-2:]
        else:
            shape = src.shape[-2:]

        nd = shape[0] * shape[1]

        if src.ndim == 3:
            src = src[:, -nd:].reshape(src.shape[0], src.shape[2], *shape)

        if tgt.ndim == 3:
            tgt_pre = tgt[:, :-nd]
            tgt = tgt[:, -nd:].reshape(tgt.shape[0], tgt.shape[2], *shape)
        else:
            tgt_pre = None

        pred = self.fwd(src)

        if self.gated:
            g, pred = torch.chunk(pred, 2, dim=1)

            g = F.sigmoid(g)

            pred = g * pred

        tgt = tgt + pred

        if tgt_pre is not None:
            tgt = rearrange(tgt, "b c h w -> b (h w) c")
            tgt = torch.cat([tgt_pre, tgt], dim=1)

        return tgt


class AttnDownsample(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int = 16):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, num_heads, 1, dim // num_heads) * 0.01)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, twod_shape: tuple[int, int]) -> torch.Tensor:
        ntok = twod_shape[0] * twod_shape[1]
        x_pre = x[:, :-ntok]

        B = x.shape[0]
        ds_hw = tuple(s // self.window_size for s in twod_shape)

        x_spat = rearrange(
            x[:, -ntok:],
            "b (h d1 w d2) c -> (b h w) (d1 d2) c",
            h=ds_hw[0],
            w=ds_hw[1],
            d1=self.window_size,
            d2=self.window_size,
        )

        B, N, C = x_spat.shape

        k, v = self.kv(x_spat).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = (self.q * self.scale).expand(B, -1, -1, -1)
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, C)
        x = self.proj(x)

        x = rearrange(x, "(b h w) c -> b (h w) c", b=x_pre.shape[0], h=ds_hw[0], w=ds_hw[1])

        x = torch.cat([x_pre, x], dim=1)
        return x


class HybridModel(nn.Module):
    def __init__(
        self,
        vit: tvit.VisionTransformer,
        conv: tconv.ConvNeXt,
        pretrained: bool = False,
        concatenate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.conv = conv
        self.vit = vit
        self.concatenate = concatenate

        conv.stages = nn.ModuleList(conv.stages)
        vit.blocks = nn.ModuleList(vit.blocks)

        self._half_vit_idx = len(vit.blocks) // 2 + 1

        self._half_conv_idx = None
        x = torch.empty(1, 3, 256, 256)
        x = self.conv.stem(x)
        for i in range(len(conv.stages)):
            x = conv.stages[i](x)
            if self._half_conv_idx is None and x.shape[-2:] == (16, 16):
                self._half_conv_idx = i + 1
                half_conv_dim = x.shape[1]
            final_conv_dim = x.shape[1]

        self.vit_to_conv_fusion = Fuser(vit.embed_dim, half_conv_dim)
        self.conv_to_vit_fusion = Fuser(half_conv_dim, vit.embed_dim)
        self.vit_ds = AttnDownsample(vit.embed_dim, window_size=2)

        embed_dim = vit.embed_dim + (final_conv_dim if concatenate else 0)
        if not concatenate:
            self.final_fuse = Fuser(final_conv_dim, vit.embed_dim, gated=False)
        self.final_block = tvit.Block(embed_dim, num_heads=16)

        self.embed_dim = embed_dim

    @property
    def patch_size(self):
        return 32

    @property
    def no_fsdp_wrap_types(self):
        return {tvit.VisionTransformer, tconv.ConvNeXt}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y_vit = self.vit.patch_generator(x)

        for i in range(self._half_vit_idx):
            y_vit = self.vit.blocks[i](y_vit)

        y_conv = self.conv.stem(x)
        for i in range(self._half_conv_idx):
            y_conv = self.conv.stages[i](y_conv)

        y_vit, y_conv = self.conv_to_vit_fusion(y_conv, y_vit), self.vit_to_conv_fusion(y_vit, y_conv)

        y_vit = self.vit_ds(y_vit, y_conv.shape[-2:])

        for i in range(self._half_vit_idx, len(self.vit.blocks)):
            y_vit = self.vit.blocks[i](y_vit)

        for i in range(self._half_conv_idx, len(self.conv.stages)):
            y_conv = self.conv.stages[i](y_conv)

        if self.concatenate:
            y_conv = rearrange(y_conv, "b c h w -> b (h w) c")
            # Average pool across the board, and replicate for each cls/register token
            conv_summary = y_conv.mean(dim=1, keepdim=True).expand(-1, self.vit.patch_generator.num_cls_patches, -1)
            y_conv = torch.cat([conv_summary, y_conv], dim=1)
            y = torch.cat([y_vit, y_conv], dim=2)
        else:
            y = self.final_fuse(y_conv, y_vit)
        y = self.final_block(y)

        summary = y[:, : self.vit.patch_generator.num_cls_tokens]
        features = y[:, self.vit.patch_generator.num_cls_patches :]

        return summary, features


@register_model
def hybrid_base(pretrained=False, concatenate: bool = False, weight_init: str = "skip", **kwargs):
    cfg = dict(num_classes=0, **kwargs)
    conv = tconv.convnextv2_base(pretrained=pretrained, **cfg)
    vit = tvit.vit_base_patch16_224(pretrained=pretrained, weight_init=weight_init, **cfg)

    return HybridModel(vit, conv, pretrained, concatenate=concatenate)


@register_model
def hybrid_large(pretrained=False, concatenate: bool = False, weight_init: str = "skip", **kwargs):
    cfg = dict(num_classes=0, **kwargs)
    conv = tconv.convnextv2_large(pretrained=pretrained, **cfg)
    vit = tvit.vit_large_patch16_224(pretrained=pretrained, weight_init=weight_init, **cfg)

    return HybridModel(vit, conv, pretrained, concatenate=concatenate)


@register_model
def hybrid_huge(pretrained=False, concatenate: bool = False, weight_init: str = "skip", **kwargs):
    cfg = dict(num_classes=0, **kwargs)
    conv = tconv.convnextv2_huge(pretrained=pretrained, **cfg)
    vit = et.vit_huge_patch16_224(pretrained=pretrained, weight_init=weight_init, **cfg)

    return HybridModel(vit, conv, pretrained, concatenate=concatenate)
