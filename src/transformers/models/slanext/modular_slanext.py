# Copyright 2025 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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


import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BaseImageProcessor
from ...image_transforms import normalize, pad
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ..vitdet.modeling_vitdet import VitDetLayerNorm


class SLANeXtMLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embedding_dim (int): Embedding dimension.
            mlp_dim (int): Hidden dimension of MLP.
            act (type[nn.Module]): Activation layer.
        """
        super().__init__()

        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class SLANeXtImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()

        self.img_size = img_size
        self.patch_embed = SLANeXtPatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()

        for i in range(depth):
            block = SLANeXtVary_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            VitDetLayerNorm(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            VitDetLayerNorm(out_chans),
        )

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)

        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.net_2(x)
        return x


def window_partition(x, window_size: int):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, 0))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class SLANeXtVary_Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SLANeXtAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = SLANeXtMLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


def get_rel_pos(q_size: int, k_size: int, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
            align_corners=False,
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, dtype=torch.float32, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=torch.float32, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn,
    q,
    rel_pos_h,
    rel_pos_w,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.reshape(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).reshape(
        B, q_h * q_w, k_h * k_w
    )
    return attn


class SLANeXtAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = torch.matmul(q * self.scale, k.transpose(1, 2))
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v).reshape(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class SLANeXtPatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def _build_vary(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_encoder = SLANeXtImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    return image_encoder


class SLANeXtVary_VIT_B(nn.Module):
    def __init__(
        self,
        in_channels=3,
        image_size=768,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    ):
        super().__init__()

        self.vision_tower_high = _build_vary(
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_global_attn_indexes=[2, 5, 8, 11],
            image_size=image_size,
        )
        self.out_channels = 1024

    def forward(self, input_data):
        pixel_values = input_data
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = torch.repeat_interleave(pixel_values, repeats=3, dim=1)
        cnn_feature = self.vision_tower_high(pixel_values)
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)
        return cnn_feature


class SLANeXtAttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super().__init__()

        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden).unsqueeze(1)

        res = batch_H_proj + prev_hidden_proj
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = alpha.transpose(1, 2)
        context = torch.bmm(alpha, batch_H).squeeze(1)
        concat_context = torch.cat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return (cur_hidden, cur_hidden), alpha


class SLANeXtMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SLANeXtHWAttention(nn.Module):
    def __init__(
        self,
        head_dim=32,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.scale = qk_scale or self.head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        C_ = C // 3
        qkv = x.reshape(B, N, 3, C_ // self.head_dim, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, -1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C_)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, H, W, C = img.shape
    img_reshape = img.reshape(B, H // H_sp, H_sp, W // W_sp, W_sp, C)
    img_perm = img_reshape.permute(0, 1, 3, 2, 4, 5).reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.reshape(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).flatten(1, 4)
    return img


class SLANeXtHead_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        split_h=4,
        split_w=4,
        h_num_heads=None,
        w_num_heads=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        eps=1e-6,
    ):
        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.split_h = split_h
        self.split_w = split_w
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim, eps=eps)
        self.h_num_heads = h_num_heads if h_num_heads is not None else num_heads // 2
        self.w_num_heads = w_num_heads if w_num_heads is not None else num_heads // 2
        self.head_dim = dim // num_heads
        self.mixer = SLANeXtHWAttention(head_dim=dim // num_heads, qk_scale=qk_scale, attn_drop=attn_drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = SLANeXtMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        qkv = self.qkv(x).reshape(B, H, W, 3 * C)

        x1 = qkv[:, :, :, : 3 * self.h_num_heads * self.head_dim]
        x2 = qkv[:, :, :, 3 * self.h_num_heads * self.head_dim :]

        x1 = self.mixer(img2windows(x1, self.split_h, W))
        x2 = self.mixer(img2windows(x2, H, self.split_w))
        x1 = windows2img(x1, self.split_h, W, H, W)
        x2 = windows2img(x2, H, self.split_w, H, W)

        attened_x = torch.cat([x1, x2], 2)
        attened_x = self.proj(attened_x)

        x = self.norm1(x + self.drop_path(attened_x))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.permute(0, 2, 1).reshape(-1, C, H, W)
        return x


class SLANeXtSLAHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_size,
        out_channels=30,
        max_text_length=500,
        loc_reg_num=4,
        fc_decay=0.0,
        use_attn=False,
        **kwargs,
    ):
        """
        @param in_channels: input shape
        @param hidden_size: hidden_size for RNN and Embedding
        @param out_channels: num_classes to rec
        @param max_text_length: max text pred
        """
        super().__init__()

        if isinstance(in_channels, int):
            self.is_next = True
            in_channels = 512
        else:
            self.is_next = False
            in_channels = in_channels[-1]
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.emb = self._char_to_onehot
        self.num_embeddings = out_channels
        self.loc_reg_num = loc_reg_num
        self.eos = self.num_embeddings - 1

        self.structure_attention_cell = SLANeXtAttentionGRUCell(in_channels, hidden_size, self.num_embeddings)
        self.structure_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(hidden_size, out_channels),
        )

        dpr = np.linspace(0, 0.1, 2)

        self.use_attn = use_attn
        if use_attn:
            layer_list = [
                SLANeXtHead_Block(
                    in_channels,
                    num_heads=2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path=dpr[i],
                )
                for i in range(2)
            ]
            self.cross_atten = nn.Sequential(*layer_list)

        self.loc_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, loc_reg_num),
            nn.Sigmoid(),
        )

    def forward(self, inputs, targets=None):
        if self.is_next:
            fea = inputs
            batch_size = fea.shape[0]
        else:
            fea = inputs[-1]
            batch_size = fea.shape[0]
            if self.use_attn:
                fea = fea + self.cross_atten(fea)
            fea = fea.reshape(fea.shape[0], fea.shape[1], -1)
            fea = fea.permute(0, 2, 1)

        hidden = torch.zeros((batch_size, self.hidden_size), device=fea.device)
        structure_preds = torch.zeros((batch_size, self.max_text_length + 1, self.num_embeddings), device=fea.device)
        loc_preds = torch.zeros((batch_size, self.max_text_length + 1, self.loc_reg_num), device=fea.device)
        structure_preds.requires_grad = False
        loc_preds.requires_grad = False

        if self.training and targets is not None:
            structure = targets[0]
            max_len = targets[-2].max().int()
            for i in range(max_len + 1):
                hidden, structure_step, loc_step = self._decode(structure[:, i], fea, hidden)
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step
            structure_preds = structure_preds[:, : max_len + 1]
            loc_preds = loc_preds[:, : max_len + 1]
        else:
            structure_ids = torch.zeros((batch_size, self.max_text_length + 1), dtype=torch.long, device=fea.device)
            pre_chars = torch.zeros(size=[batch_size], dtype=torch.long, device=fea.device)
            max_text_length = self.max_text_length
            for i in range(max_text_length + 1):
                hidden, structure_step, loc_step = self._decode(pre_chars, fea, hidden)
                pre_chars = structure_step.argmax(dim=1)
                structure_preds[:, i, :] = structure_step
                loc_preds[:, i, :] = loc_step

                structure_ids[:, i] = pre_chars
                if (structure_ids == self.eos).any(-1).all():
                    break
        if not self.training:
            structure_preds = F.softmax(structure_preds[:, : i + 1], dim=-1)
            loc_preds = loc_preds[:, : i + 1]
        return {"structure_probs": structure_preds, "loc_preds": loc_preds}

    def _decode(self, pre_chars, features, hidden):
        """
        Predict table label and coordinates for each step
        @param pre_chars: Table label in previous step
        @param features:
        @param hidden: hidden status in previous step
        @return:
        """
        emb_feature = self.emb(pre_chars)
        (output, hidden), alpha = self.structure_attention_cell(hidden, features, emb_feature)

        structure_step = self.structure_generator(output)
        loc_step = self.loc_generator(output)
        return hidden, structure_step, loc_step

    def _char_to_onehot(self, input_char):
        return F.one_hot(input_char, self.num_embeddings).float()


@auto_docstring(custom_intro="Configuration for the SLANeXt model.")
class SLANeXtConfig(PreTrainedConfig):
    model_type = "slanext"
    """
    This is the configuration class to store the configuration of a [`SLANeXt`]. It is used to instantiate a
    SLANeXt table recognition model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the SLANeXt
    PaddlePaddle/SLANeXt_wired and PaddlePaddle/SLANeXt_wireless architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_global_attn_indexes: list[int] = [2, 5, 8, 11],
        out_channels: int = 50,
        hidden_size: int = 512,
        max_text_length: int = 500,
        loc_reg_num: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.loc_reg_num = loc_reg_num


class SLANeXtPreTrainedModel(PreTrainedModel):
    """
    Base class for all SLANeXt pre-trained models. Handles model initialization,
    configuration, and loading of pre-trained weights, following the Transformers library conventions.
    """

    config: SLANeXtConfig
    base_model_prefix = "slanext"
    main_input_name = "pixel_values"
    input_modalities = ("image",)

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)

        # Initialize positional embeddings to zero
        if isinstance(module, SLANeXtImageEncoderViT):
            if module.pos_embed is not None:
                init.constant_(module.pos_embed, 0.0)

        # Initialize relative positional embeddings to zero
        if isinstance(module, SLANeXtAttention):
            if module.use_rel_pos:
                init.constant_(module.rel_pos_h, 0.0)
                init.constant_(module.rel_pos_w, 0.0)

        # Initialize SLAHead layers
        if isinstance(module, SLANeXtSLAHead):
            # Initialize structure_generator layers
            for i, layer in enumerate(module.structure_generator):
                if isinstance(layer, nn.Linear):
                    stdv = 1.0 / math.sqrt(module.hidden_size * 1.0)
                    init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        init.uniform_(layer.bias, -stdv, stdv)

            # Initialize loc_generator layers
            for i, layer in enumerate(module.loc_generator):
                if isinstance(layer, nn.Linear):
                    stdv = 1.0 / math.sqrt(module.hidden_size * 1.0)
                    init.uniform_(layer.weight, -stdv, stdv)
                    if layer.bias is not None:
                        init.uniform_(layer.bias, -stdv, stdv)

        # Initialize VitDetLayerNorm (imported from vitdet)
        if isinstance(module, VitDetLayerNorm):
            init.constant_(module.weight, 1.0)
            init.constant_(module.bias, 0.0)


@auto_docstring(custom_intro="The SLANeXt model.")
class SLANeXtModel(SLANeXtPreTrainedModel):
    """
    Core SLANeXt model, consisting of Backbone and Head networks.
    Generates structure probs for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.backbone = SLANeXtVary_VIT_B(
            image_size=512,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            encoder_global_attn_indexes=config.encoder_global_attn_indexes,
        )
        self.head = SLANeXtSLAHead(
            in_channels=self.backbone.out_channels,
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            max_text_length=config.max_text_length,
            loc_reg_num=config.loc_reg_num,
        )
        self.post_init()

    def forward(self, pixel_values):
        x = self.backbone(pixel_values)
        x = self.head(x)

        return x


@auto_docstring(custom_intro="ImageProcessor for the SLANeXt model.")
class SLANeXtImageProcessor(BaseImageProcessor):
    def __init__(self):
        self.target_long_edge = 512
        self.target_pad_size = 512
        self.init_decoder()

    def _tablerec_resize(self, img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """
        Resize image to match cv2.resize with INTER_LINEAR as closely as possible.

        This implementation uses OpenCV's approach with vectorized operations:
        1. Float32 precision for all floating-point calculations
        2. Fixed-point arithmetic with 11-bit precision (scale = 2048)
        3. Vectorized bilinear interpolation for efficiency
        4. Proper boundary handling

        Args:
            img (`np.ndarray`):
                Input image in HWC format (height, width, channels).
            target_size (`tuple[int, int]`):
                Target size as (width, height) to match cv2.resize convention.

        Returns:
            `np.ndarray`: Resized image in HWC format.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle grayscale images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            squeeze_output = True
        else:
            squeeze_output = False

        # OpenCV's fixed-point arithmetic constants
        INTER_RESIZE_COEF_BITS = 11
        INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS  # 2048

        # Calculate scale factors using float32 (matching OpenCV)
        scale_x = np.float32(w) / np.float32(target_w)
        scale_y = np.float32(h) / np.float32(target_h)

        # Pre-compute X interpolation tables (vectorized)
        dx_arr = np.arange(target_w, dtype=np.float32)
        fx_arr = (dx_arr + np.float32(0.5)) * scale_x - np.float32(0.5)
        sx_arr = np.floor(fx_arr).astype(np.int32)
        fx_frac_arr = fx_arr - sx_arr.astype(np.float32)

        # Handle X boundaries
        mask_left = sx_arr < 0
        mask_right = sx_arr >= w - 1
        sx_arr[mask_left] = 0
        fx_frac_arr[mask_left] = 0.0
        sx_arr[mask_right] = w - 2
        fx_frac_arr[mask_right] = 1.0

        xalpha = np.round(fx_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        xofs = sx_arr

        # Pre-compute Y interpolation tables (vectorized)
        dy_arr = np.arange(target_h, dtype=np.float32)
        fy_arr = (dy_arr + np.float32(0.5)) * scale_y - np.float32(0.5)
        sy_arr = np.floor(fy_arr).astype(np.int32)
        fy_frac_arr = fy_arr - sy_arr.astype(np.float32)

        # Handle Y boundaries
        mask_top = sy_arr < 0
        mask_bottom = sy_arr >= h - 1
        sy_arr[mask_top] = 0
        fy_frac_arr[mask_top] = 0.0
        sy_arr[mask_bottom] = h - 2
        fy_frac_arr[mask_bottom] = 1.0

        yalpha = np.round(fy_frac_arr * np.float32(INTER_RESIZE_COEF_SCALE)).astype(np.int32)
        yofs = sy_arr

        # Create meshgrid for vectorized operations
        sy_grid = yofs[:, np.newaxis]  # (target_h, 1)
        sx_grid = xofs[np.newaxis, :]  # (1, target_w)
        ay_grid = yalpha[:, np.newaxis]  # (target_h, 1)
        ax_grid = xalpha[np.newaxis, :]  # (1, target_w)

        ay_inv = INTER_RESIZE_COEF_SCALE - ay_grid
        ax_inv = INTER_RESIZE_COEF_SCALE - ax_grid

        # Perform vectorized bilinear interpolation for each channel
        output = np.zeros((target_h, target_w, img.shape[2]), dtype=np.uint8)

        for c in range(img.shape[2]):
            # Get 4 corner pixels using advanced indexing
            p00 = img[sy_grid, sx_grid, c].astype(np.int32)  # (target_h, target_w)
            p10 = img[sy_grid, sx_grid + 1, c].astype(np.int32)
            p01 = img[sy_grid + 1, sx_grid, c].astype(np.int32)
            p11 = img[sy_grid + 1, sx_grid + 1, c].astype(np.int32)

            # Vectorized bilinear interpolation
            val = ay_inv * (ax_inv * p00 + ax_grid * p10) + ay_grid * (ax_inv * p01 + ax_grid * p11)

            # Divide with rounding
            shift_bits = INTER_RESIZE_COEF_BITS * 2
            val = (val + (1 << (shift_bits - 1))) >> shift_bits

            output[:, :, c] = np.clip(val, 0, 255).astype(np.uint8)

        if squeeze_output:
            output = output[:, :, 0]

        return output

    def calc_padding(self, img):
        h, w = img.shape[:2]
        pad_right = max(0, self.target_pad_size - w)
        pad_bottom = max(0, self.target_pad_size - h)
        return ((0, pad_bottom), (0, pad_right))

    def calc_resize(self, img):
        h, w = img.shape[:2]
        scale = self.target_long_edge / max(h, w)
        h_resize = round(h * scale)
        w_resize = round(w * scale)
        return [w_resize, h_resize]

    def preprocess(self, img):
        img = np.array(img)
        img = self._tablerec_resize(img, self.calc_resize(img))
        img = img / 255.0
        img = normalize(image=img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = pad(image=img, padding=self.calc_padding(img))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img).float()

        return img

    def post_process_table_recognition(self, x):
        return self.decode(x[0].detach().cpu())[0]

    def init_decoder(self, merge_no_span_structure=True):
        dict_character = [
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tr>",
            "</tr>",
            "<td>",
            "<td",
            ">",
            "</td>",
        ]
        for i in range(19):
            dict_character.append(f' colspan="{i + 2}"')
        for i in range(19):
            dict_character.append(f' rowspan="{i + 2}"')

        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]

    def add_special_char(self, dict_character):
        """add_special_char"""
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        """get_ignored_tokens"""
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        """get_beg_end_flag_idx"""
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def decode(self, pred):
        self.pred = pred
        structure_probs = np.array([list(self.pred[0])])
        """convert text-label into text-index."""
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_str_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_str_list.append(structure_list)
            structure_score = np.mean(score_list)

        structure_str_list = [
            (["<html>", "<body>", "<table>"] + structure + ["</table>", "</body>", "</html>"])
            for structure in structure_str_list
        ]

        return [{"structure": structure, "structure_score": structure_score} for structure in structure_str_list]


@auto_docstring(custom_intro="TableRecognition for the SLANeXt model.")
class SLANeXtForTableRecognition(SLANeXtPreTrainedModel):
    """
    SLANeXt model for table recognition tasks.
    """

    def __init__(self, config: SLANeXtConfig):
        super().__init__(config)
        self.model = SLANeXtModel(config)
        self.post_init()

    def forward(self, pixel_values, return_dict: bool | None = None, **kwargs):
        x = self.model(pixel_values)
        if not return_dict:
            return ((x["structure_probs"]),)
        else:
            return SLANeXtOutput(structure_probs=x["structure_probs"])


@dataclass
class SLANeXtOutput(ModelOutput):
    """
    Output class for SLANeXtForTableRecognition. Extends ModelOutput
    to include table recognition probs.
    """

    structure_probs: torch.FloatTensor | None = None


__all__ = [
    "SLANeXtForTableRecognition",
    "SLANeXtImageProcessor",
    "SLANeXtConfig",
    "SLANeXtModel",
    "SLANeXtPreTrainedModel",
]
