# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
import torch
import torch.nn as nn
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    "beit_base_patch16_224",
    # 'beit_large_patch16_224_8k_vocab',
]


class VisionTransformerForCyclicalTraining(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_values=None,
        attn_head_dim=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        init_std=0.02,
    ):
        super().__init__()
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape
                    if use_rel_pos_bias
                    else None,
                    attn_head_dim=attn_head_dim,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        # self.lm_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 2, embed_dim),
        # )
        # self.lm_head = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        # )
        self.lm_head = nn.Linear(embed_dim, embed_dim)
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos, layer_results):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.view(bool_masked_pos.size(0), -1, 1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w # B x T x C
            
            # print(bool_masked_pos.shape)
            # print(bool_masked_pos.sum((1,2)))
            # print('x', x.shape)
            # bool_masked = bool_masked_pos.reshape(bool_masked_pos.size(0), -1).bool()
            # print('bool_masked', bool_masked.shape)
            # print('asd1', x[bool_masked].shape)
            # exit(0)

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        z = []
        for i, blk in enumerate(self.blocks):
            x, fc_feature = blk(x, rel_pos_bias=rel_pos_bias)
            if layer_results == 'end':
                z.append(x)
            elif layer_results == 'fc':
                z.append(fc_feature)

        return z if layer_results else self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False, layer_results=None):
        x = self.forward_features(
            x, bool_masked_pos=bool_masked_pos, layer_results=layer_results
        )
        # TODO(PVP)
        return x

        if layer_results:
            return [z[:, 1:] for z in x]
        elif return_all_tokens:
            x = x[:, 1:]
            return self.lm_head(x)
        else:
            # return the masked tokens
            x = x[:, 1:]
            bsz = x.size(0)
            fsz = x.size(-1)
            bool_masked_pos = bool_masked_pos.flatten().bool()
            x = x.reshape(-1, fsz)[bool_masked_pos]
            return self.lm_head(x)


@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    # _ = kwargs.pop("num_classes")
    model = VisionTransformerForCyclicalTraining(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    # _ = kwargs.pop("num_classes")
    model = VisionTransformerForCyclicalTraining(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_huge_patch16_224(pretrained=False, **kwargs):
    # _ = kwargs.pop("num_classes")
    model = VisionTransformerForCyclicalTraining(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# @register_model
# def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
#     _ = kwargs.pop("num_classes")
#     model = VisionTransformerForMaskedImageModeling(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.load(
#             kwargs["init_ckpt"], map_location="cpu"
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
