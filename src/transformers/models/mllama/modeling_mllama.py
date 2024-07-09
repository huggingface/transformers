# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Mllama model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn, Tensor

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_mllama import MllamaConfig, MllamaCrossAttentionTextConfig, MllamaCrossAttentionVisionConfig
import collections
from functools import partial
import torch.nn.functional as F
from ...configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MllamaConfig"

import collections
import torch
import math

def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def _stack_images(
    images, # TODO type as List[List[PIL.Image]]
    max_num_chunks: int,
    image_res: int,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Takes a list of list of images and stacks them into a tensor.
    This function is needed since images can be of completely
    different resolutions and aspect ratios.
    """
    max_num_images = max(max(len(xx) for xx in images), 1)

    out_images, out_num_chunks = [], []
    for imgs_sample in images:
        out_images_i = torch.zeros(
            max_num_images,
            max_num_chunks,
            3,
            image_res,
            image_res,
        )
        _num_chunks = []
        for j, chunks_image in enumerate(imgs_sample):
            out_images_i[j, : chunks_image.shape[0]] = chunks_image
            _num_chunks.append(chunks_image.shape[0])
        out_images.append(out_images_i)
        out_num_chunks.append(_num_chunks)
    return torch.stack(out_images), out_num_chunks


def _pad_masks(
    all_masks: List[List[List[int]]],
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
) -> torch.Tensor:
    dtype = torch.bfloat16
    inf_value = get_negative_inf_value(dtype)

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[
                    idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks
                ].fill_(0.0)

    return out_masks



# vision encoder utils 
# -------------------
def resize_local_position_embedding(orig_pos_embed, grid_size):
    """
    Resize position embedding for vision encoder.
    Original position embedding is [n_tiles * n_tiles + 1, dim]
    New position embedding will be [grid_size[0] * grid_size[1] + 1, dim]
    """
    new_grid_size = to_2tuple(grid_size)
    orig_grid_size = to_2tuple(int(math.sqrt(len(orig_pos_embed) - 1)))
    new_seq_len = new_grid_size[0] * new_grid_size[1] + 1

    new_pos_emb_tok, new_pos_emb_img = (
        orig_pos_embed[:1],
        orig_pos_embed[1:],
    )
    logger.info(
        f"resizing position embedding grid-size from {orig_grid_size} to {new_grid_size}"
    )

    new_pos_emb_img = new_pos_emb_img.reshape(
        1, orig_grid_size[0], orig_grid_size[1], -1
    ).permute(0, 3, 1, 2)

    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, new_grid_size[0] * new_grid_size[1], -1
    )[0]
    new_pos_embed = torch.cat([new_pos_emb_tok, new_pos_emb_img], dim=0)
    return new_pos_embed

def initialize_global_position_embedding_from_local(pos_and_cls_embed, grid_size, x_scale, y_scale):
    """
    Takes a local position embedding for vision encoder and uses it
    to initialize the global position embedding.
    Input: local position embedding of shape [grid_size[0] * grid_size[1] + 1, dim]
    Returns: global position embedding of shape [x_scale, y_scale, grid_size[0] * grid_size[1] + 1, dim]
    Here x_scale and y_scale are the number of tiles along x-axis and y-axis respectively.
    """
    pos_embed = pos_and_cls_embed[1:]
    cls_embed = pos_and_cls_embed[0].view(1, 1, 1, -1)
    grid_size = to_2tuple(grid_size)
    new_pos_emb_img = pos_embed.reshape(
        1, grid_size[0], grid_size[1], -1
    ).permute(0, 3, 1, 2)
    new_grid_size = (x_scale * grid_size[0], y_scale * grid_size[1])
    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1)
    new_pos_emb_img = new_pos_emb_img.view(
        x_scale, grid_size[0], y_scale, grid_size[1], -1
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 1, 3, 4).contiguous()
    new_pos_emb_img = new_pos_emb_img.reshape(
        x_scale, y_scale, grid_size[0] * grid_size[1], -1
    )
    cls_embed = cls_embed.expand(x_scale, y_scale, -1, -1)
    pos_and_cls_embed = torch.cat([cls_embed, new_pos_emb_img], dim=2)
    return pos_and_cls_embed


def resize_global_position_embedding(pos_and_cls_embed, grid_size, x_scale, y_scale):
    """
    Takes a global position embedding for vision encoder and resizes it to new size.
    Input: global position embedding of shape [x_old, y_old, old_grid_size[0] * old_grid_size[1] + 1, dim]
    Returns: global position embedding of shape [x_scale, y_scale, grid_size[0] * grid_size[1] + 1, dim]
    Here x_scale and y_scale are the number of tiles along x-axis and y-axis respectively.
    """
    # first remove cls token
    pos_embed = pos_and_cls_embed[:, :, 1:]
    cls_embed = pos_and_cls_embed[:, :, 0].unsqueeze(2)

    xs_old, ys_old, ntok, dim = pos_embed.shape
    old_grid_size = int(math.sqrt(ntok))

    # move to correct form for interpolation
    pos_embed = pos_embed.view(xs_old, ys_old, old_grid_size, old_grid_size, dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(xs_old * old_grid_size, ys_old * old_grid_size, dim)
    pos_embed = pos_embed.unsqueeze(0)

    # interpolate
    new_size = (grid_size[0] * x_scale, grid_size[1] * y_scale)
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed_resized = F.interpolate(
        pos_embed,
        size=new_size,
        mode="bilinear",
        align_corners=True,
    )
    pos_embed = pos_embed_resized.permute(0, 2, 3, 1)[0]

    # move it back in place
    pos_embed = pos_embed.view(x_scale, grid_size[0], y_scale, grid_size[1], dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(x_scale, y_scale, grid_size[0] * grid_size[1], dim)

    # interpolate cls token
    cls_embed = cls_embed.permute(2, 3, 0, 1)
    cls_embed_resized = F.interpolate(
        cls_embed,
        size=(x_scale, y_scale),
        mode="bilinear",
        align_corners=True,
    )
    cls_embed = cls_embed_resized.permute(2, 3, 0, 1)
    # add cls token back in
    pos_and_cls_embed = torch.cat([cls_embed, pos_embed], dim=2)

    return pos_and_cls_embed

def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    print(ar)
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks

def expand_num_tokens_to_mult8(x):
    num_pad_tokens = (8 - (x.shape[-2] % 8))
    if num_pad_tokens == 0:
        return x, 0
    else:
        return torch.cat(
            [
                x,
                torch.zeros(
                    (x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]),
                    dtype=x.dtype,
                    device=x.device
                ),
            ],
            dim=-2,
        ), num_pad_tokens

def contract_num_tokens_from_mult8(x, num_pad_tokens):
    if num_pad_tokens == 0:
        return x
    return x[:, :, :-num_pad_tokens]

# --------------------
# end of vision encoder utils


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]


# Image encoder for inference
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x



class ImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.fc1 = nn.Linear(
            dim,
            hidden_dim,
            bias=True,
        )
        self.fc2 = nn.Linear(
            hidden_dim,
            dim,
            bias=True,
        )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = F.linear(x, self.fc1.weight, self.fc1.bias)
        hidden = self.non_linearity(hidden)
        hidden = F.linear(hidden, self.fc2.weight)
        hidden += self.fc2.bias
        return hidden


class ImageAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        n_heads,
    ):
        super().__init__()

        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads 
        self.n_local_kv_heads = (
            self.n_kv_heads
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            n_heads * self.head_dim,
            dim,
            bias=True,

        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        xq, xk, xv = [
            F.linear(x, w, b)
            for (w, b) in [
                (self.q_proj.weight, self.q_proj.bias),
                (self.k_proj.weight, self.k_proj.bias),
                (self.v_proj.weight, self.v_proj.bias),
            ]
        ]

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = F.linear(attn_output, self.o_proj.weight)
        out = out / self.qkvo_replication
        out += self.o_proj.bias
        return out


class ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads
        self.self_attn = ImageAttention(
            dim=d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.mlp = ImageFeedForward(
            dim=d_model, # 1280
            hidden_dim=int(mlp_ratio * d_model), # 4 x 1280 = 5120
            dropout=0.0,
            act_layer=act_layer,
        )
        self.layer_norm2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.self_attn(self.ln_1(x), mask=mask)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        num_layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.width = width
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                ImageTransformerBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    gated=gated,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None):
        out = []
        for idx, r in enumerate(self.layers):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, torch.stack(out, dim=-1)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        max_num_tiles: int,
        ckpt_path: str = None,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1280,
        num_layers: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        in_channels: int = 3,
        load_ckpt: bool = False,
        n_global_layers: int = 2,
        global_model: bool = False,
        return_intermediate=None,
    ):
        super().__init__()
        self.global_model = global_model
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = ImageTransformer(
            width, num_layers, heads, mlp_ratio, act_layer=act_layer
        )
        # pre and post tile position embedding
        self.global_transformer = ImageTransformer(
            width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True
        )
        # pre and post tile position embedding
        self.pre_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.post_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.grid_size[0] * self.grid_size[1] + 1,
                width,
            )
        )
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

    def apply_positional_embedding(self, x, ar):
        out = []
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (
            1 - self.gated_positional_embedding_gate.tanh()
        )
        x = x.view(bsz, num_chunks, num_tokens, dim)
        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += (
                _pos_embed * self.gated_positional_embedding_gate.tanh()
            )
        return x

    def apply_class_embedding(self, x):
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        return x

    def forward(self, images: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.patch_embedding(x)  # shape = [*, width, grid ** 2]
        _, ntok, dim = x.shape
        # equivalent to column parallel conv2d op
        x = x.permute(0, 2, 3, 1).reshape(bsz * num_concurrent_media, num_chunks, -1, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar)

        x = self.ln_pre(x)
        npad, attn_mask = 0, None
        x, npad = expand_num_tokens_to_mult8(x)
        attn_mask = build_encoder_attention_mask(x, ar, ntok, num_chunks, 1)
        print(attn_mask)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        x, int_x = self.transformer(
            x, return_intermediate=self.return_intermediate, mask=attn_mask
        )

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = torch.cat([x, int_x], dim=-1)
        return x


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config: MllamaCrossAttentionTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads

        self.n_local_heads = config.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len

        self.q_proj = nn.Linear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.n_heads * self.head_dim,
            config.dim,
            bias=False,

        )
        self.n_heads = config.n_heads
        
    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        cache_shape = (
            max_batch_size,
            self.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )
        device = next(self.parameters()).device
        self.register_buffer(
            "key_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                cache_shape,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_ids: torch.LongTensor,
    ):

        xq, xk, xv = [
            F.linear(x, w) for w in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
        ]

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.key_cache[:bs, position_ids, ...] = xk
        self.value_cache[:bs, position_ids, ...] = xv

        # TODO: we can avoid slicing on first dimension by always padding to max_batch_size()
        xk = self.key_cache[:bs, ...]
        xv = self.value_cache[:bs, ...]

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = F.linear(attn_output, self.o_proj.weight)
        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(
            dim, hidden_dim
        )
        self.down_proj = nn.Linear(
            hidden_dim, dim
        )
        self.up_proj = nn.Linear(
            dim, hidden_dim
        )

    def forward(self, x):
        x1, x3 = [F.linear(x, w) for w in [self.w1.weight, self.w3.weight]]
        x1 = F.silu(x1)
        x_in = x1 * x3
        out = F.linear(x_in, self.w2.weight)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_id: int):
        """
        Initialize a TransformerBlock.
        Args:
            config (PretrainedConfig): configuration object for the layer.
            layer_id (int): Identifier for the layer.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.attention.setup_cache(max_batch_size, dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.
        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        h = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            position_ids=position_ids,
        )
        h = h + x
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width)
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))


    @staticmethod
    def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = F.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(
            x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype
        )
        for idx, arx in enumerate(ar):
            w, h = arx
            out_pos_embed[idx, : w * h] = embed[:w, :h].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed
        return x


def _noinit(x):
    return x


class CrossAttention(torch.nn.Module):
    """Cross attention layer."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.q_proj = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )

        # local heads
        self.n_local_heads = self.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

    def _compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        bsz = xattn_tokens.shape[0]
        xk = self.k_proj(xattn_tokens)
        xv = self.v_proj(xattn_tokens)

        _, seqlen_y, _ = xk.shape

        xk = xk.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)

        xk, xv = [tensor.transpose(1, 2) for tensor in (xk, xv)]

        # repeat k/v heads if n_kv_heads < n_heads
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        xk = self.k_norm(xk)

        return torch.stack([xk, xv])

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        xq = F.linear(x, self.q_proj.weight)
        bsz, seqlen, _ = x.shape

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2)

        xk, xv = xattn_cache

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=xattn_mask, dropout_p=0.0
        )
        output = output * full_text_row_masked_out_mask
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)

        out = F.linear(output, self.o_proj.weight)
        return out


class CrossAttentionTransformerBlock(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(
        self,
        config: MllamaCrossAttentionTextConfig,
        layer_id: int,
        no_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.norm_eps = config.norm_eps
        self.ffn_dim_multiplier = config.ffn_dim_multiplier
        self.multiple_of = config.multiple_of

        self.attention = CrossAttention(
            dim=self.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=self.norm_eps,
        )

        self.input_layernorm = RMSNorm(
            self.dim,
            eps=self.norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.mlp = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            ffn_dim_multiplier=self.ffn_dim_multiplier,
            multiple_of=self.multiple_of,
        )
        self.post_attention_layernorm = RMSNorm(
            self.dim,
            eps=self.norm_eps,
        )
        self.ffn_gate = torch.nn.Parameter(torch.zeros(1))

        self.no_ffn = no_ffn


    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self.attention.compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: Tuple[torch.Tensor, torch.Tensor],
        xattn_cache: torch.Tensor,
    ) -> torch.Tensor:
        _attn_out = self.attention(
            x=self.attention_norm(x),
            xattn_mask=xattn_mask,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )
        h = x + self.gate_attn.tanh() * _attn_out
        _ffn = self.feed_forward(self.ffn_norm(h))
        _ffn = full_text_row_masked_out_mask[:, 0] * _ffn  # type: ignore
        h = h + self.ffn_gate.tanh() * _ffn * float(not self.no_ffn)
        return h


class DummyCrossAttentionTransformerBlock:
    """Dummy cross-attention transformer block with tanh-gated attention and feedforward."""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class DummySelfAttentionTransformerBlock:
    """Dummy self-attention transformer block"""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class MllamaCrossAttentionVisionModel(torch.nn.Module):
    def __init__(self, config: MllamaCrossAttentionVisionConfig):
        super().__init__()
        return_intermediate = config.return_intermediate
        self.vision_input_dim = config.vision_input_dim
        self.image_res = config.vision_chunk_size
        self.max_num_chunks = config.vision_max_num_chunks
        if return_intermediate is not None:
            return_intermediate = [int(l) for l in return_intermediate.split(",")]
            self.vision_input_dim = (
                len(return_intermediate) + 1
            ) * self.vision_input_dim
        self.patch_size = config.patch_size
        self.vision_encoder = VisionEncoder(
            max_num_tiles=config.max_num_tiles,
            image_size=self.image_res,
            patch_size=self.patch_size,
            n_global_layers=config.global_vision_layers,
            global_model=True,
            return_intermediate=return_intermediate,
        )
        # vision token projection
        self.vision_projection = nn.Linear(
            self.vision_input_dim,
            config.projection_dim,
            bias=True,
        )

    def forward(
        self, images: torch.Tensor, aspect_ratios: torch.Tensor
    ) -> torch.Tensor:
        # vision_tokens: (B, T, D)
        # aspect_ratios: (B, T)
        # h: (B, T, D)
        vision_tokens = self.vision_encoder(
            images.to(dtype=torch.bfloat16), aspect_ratios
        )

        vision_tokens = F.linear(vision_tokens, self.vision_projection.weight)
        return vision_tokens


class MllamaCrossAttentionTextModel(PreTrainedModel):
    INFERENCE_IMAGE_TOKEN_ID = 128010

    def __init__(self, config: MllamaCrossAttentionTextConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.rope_theta = config.rope_theta
        self.use_scaled_rope=config.use_scaled_rope
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.dim
        )
        self.max_seq_len = config.max_seq_len
        self.vision_num_cross_attention_layers = config.vision_num_cross_attention_layers
        self.pos_embeddings = None
        # final norm layer (not necessary for post-norm)
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # output layer
        self.output = nn.Embedding(
            self.dim, self.vocab_size
        )

        self.n_llama_layers = self.n_layers
        self.model_dim = self.dim

        # BLOCKS

        self.fusion_schedule = self._init_fusion_schedule(
            self.vision_num_cross_attention_layers
        )
        self.learnable_embedding = nn.Embedding(
            8,
            self.dim,
        )
        self.num_frozen_embeddings = self.tok_embeddings.num_embeddings
        self._thresh = self.num_frozen_embeddings - 1

        # transformer blocks
        self.layers = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            layer_id = i
            block = TransformerBlock(config=config, layer_id=layer_id)
            self.layers.append(block)
            if layer_id in self.fusion_schedule:
                xa_layer_id = self.fusion_schedule.index(layer_id) + self.n_layers
                block = CrossAttentionTransformerBlock(
                    config,
                    layer_id=xa_layer_id,
                )
                self.cross_attention_layers.append(block)

        # add xattn and dummy layers to avoid conditionals in forward()
        self.text_and_xattn_layers = []

        for idx, layer in enumerate(self.layers):
            if idx in self.fusion_schedule:
                xattn_layer_idx = self.fusion_schedule.index(idx)
                xattn_layer = self.cross_attention_layers[xattn_layer_idx]
            else:
                xattn_layer_idx = 0
                xattn_layer = DummyCrossAttentionTransformerBlock()

            self.text_and_xattn_layers.append(
                (
                    layer,
                    xattn_layer,
                    xattn_layer_idx,
                )
            )
        self.freqs_cis = precompute_freqs_cis(
            self.dim // self.n_heads,
            self.max_seq_len * 2,
            self.rope_theta,
            self.use_scaled_rope,
        )

        self.cache_is_setup = False
        self.max_seq_len = self.max_seq_len

    def _init_fusion_schedule(
        self,
        num_layers: int,
    ) -> List[int]:
        llama_layers = list(range(self.n_llama_layers))

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)
        return llama_layers[::-1][::k][:num_layers][::-1]

    def get_partially_trainable_embedding(self, x):
        xz = torch.zeros_like(x, device=x.device)
        oz = torch.ones_like(x, device=x.device)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))
        x_new = (
            torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device))
            - self.num_frozen_embeddings
        )

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)

        x_orig = self.tok_embeddings(x_orig)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)


    def forward(
        self,
        position_ids: torch.LongTensor,
        h: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
    ):
        assert self.cache_is_setup, "Please set up cache before calling forward"
        mask = self.mask_cache.index_select(2, position_ids)
        freqs_cis = self.freqs_cis.index_select(0, position_ids)

        for idx, (
            layer,
            xattn_layer,
            xattn_layer_idx,
        ) in enumerate(self.text_and_xattn_layers):
            h = xattn_layer(
                x=h,
                xattn_mask=xattn_mask,
                xattn_cache=xattn_caches[xattn_layer_idx],
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            )
            h = layer(
                x=h,
                mask=mask,
                freqs_cis=freqs_cis,
                position_ids=position_ids,
            )

        h = self.norm(h)

        output = F.linear(h, self.output.weight)
        return output.float()

    def setup_cache(self, max_batch_size: int, dtype=torch.bfloat16):
        # Set up the text kv caches
        device = next(self.parameters()).device
        ones = torch.ones(
            (self.max_seq_len, self.max_seq_len),
            dtype=torch.bool,
            device=device,
        )
        self.register_buffer(
            "mask_cache",
            torch.tril(
                ones,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False,
        )
        for layer in self.layers:
            layer.setup_cache(max_batch_size, dtype=dtype)
        self.cache_is_setup = True

    def _get_xattn_mask(
        self,
        num_tokens,
        text_device,
        text_dtype,
        vision_tokens,
        cross_attention_masks,
    ) -> Tuple[Tensor, Tensor]:
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        assert (
            num_tokens == cross_attention_masks.shape[1]
        ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=2)
            .view(bsz, ntext, -1)
            .unsqueeze(1)
        )
        full_text_row_masked_out_mask = _get_full_row_masked_out_mask(
            cross_attention_masks,
            get_negative_inf_value(cross_attention_masks.dtype),
        )
        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(device=text_device, dtype=text_dtype),
            full_text_row_masked_out_mask,
        )


class MllamaPreTrainedModel(PreTrainedModel):
    config_class = MllamaConfig
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = MllamaCrossAttentionVisionModel(config.vision_config)
        self.language_model = MllamaCrossAttentionTextModel(config.text_config)

MLLAMA_START_DOCSTRING = "" # TODO add docstring to MLLAMA start and other classes

@add_start_docstrings(
    """The MLLAMA model which consists of a vision backbone and a language model.""",
    MLLAMA_START_DOCSTRING,
)
class MllamaForConditionalGeneration(MllamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = MllamaPreTrainedModel(config)

        self.lm_head = nn.Linear(config.text_config.dim, config.text_config.vocab_size)
        
        self.vision_max_num_chunks = config.vision_config.vision_max_num_chunks
        self.vision_chunk_size = config.vision_config.vision_chunk_size 
        # TODO - decide if we want to derive from config params.max_seq_len in the processor...
        """
        self.image_transform = partial(
            VariableSizeImageTransform(size=args.vision_chunk_size),
            max_num_chunks=args.vision_max_num_chunks,
        )
        """

    def setup_cache(self, max_batch_size: int, dtype: torch.dtype):
        self.language_model.setup_cache(max_batch_size, dtype)

    def compute_vision_tokens_masks(
        self,
        model_inputs, # TODO type this
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: make sure vision transformer with pure text also works
        skip_vision_encoder = False 
        images = [m.vision.images for m in model_inputs if m.vision is not None]
        if (
            len(images) == 0
            or max(len(i) for i in images) == 0
        ):
            max_num_images = 0
            num_chunks = [[self.max_num_chunks] for _ in model_inputs]
            skip_vision_encoder = True
        else:
            images_and_aspect_ratios = [
                [self.image_transform(im) for im in m.vision.images] for m in model_inputs
            ]
            transformed_images = [[x[0] for x in row] for row in images_and_aspect_ratios]
            aspect_ratios = [
                [torch.tensor(x[1]) for x in row] for row in images_and_aspect_ratios
            ]
            images, num_chunks = _stack_images(
                transformed_images,
                max_num_chunks=self.vision_max_num_chunks,
                image_res=self.vision_chunk_size,
            )
            aspect_ratios = torch.stack([torch.stack(x) for x in aspect_ratios])

        if skip_vision_encoder:
            vision_tokens = torch.zeros(
                (
                    len(model_inputs),  # batch size 
                    max_num_images,   # most likely 1 but take it from model_inputs
                    int(self.max_num_chunks), 
                    int(
                        (self.vision_model.image_res / self.vision_model.patch_size)
                        ** 2
                        + 1
                    ),
                    int(self.model_dim),
                ),
            )
        else: 
            vision_tokens = self.vision_model(images, aspect_ratios)

        vision_tokens = vision_tokens.to("cuda")
            
        bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
        xattn_caches = torch.stack(
            [
                layer.compute_xattn_kv_cache(
                    vision_tokens.view(bsz, -1, image_token_dim)
                )
                for layer in self.language_model.cross_attention_layers
            ]
        )
        padded_masks = _pad_masks(
            [x.vision.mask for x in model_inputs],
            num_chunks,
            total_len,
            self.max_num_chunks,
        )

        cross_attention_masks, full_text_row_masked_out_mask = (
            self.language_model._get_xattn_mask(
                num_tokens=total_len,
                text_device="cuda",
                text_dtype=next(self.language_model.parameters()).dtype,
                vision_tokens=vision_tokens,
                cross_attention_masks=padded_masks,
            )
        )

        return (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask)

    def forward(
        self,
        position_ids: torch.Tensor,
        tokens: torch.Tensor,
        cross_attention_masks: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
    ) -> torch.Tensor:
        h = self.language_model.get_partially_trainable_embedding(tokens[:, position_ids])
        logits = self.language_model.forward(
            position_ids=position_ids,
            h=h,
            xattn_mask=cross_attention_masks[:, :, position_ids],
            full_text_row_masked_out_mask=full_text_row_masked_out_mask[
                :, :, position_ids
            ],
            xattn_caches=xattn_caches,
        )
        return logits


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # likely not needed here. 
    # TODO do we support multi-image per message at arbitrary positions? if yes, we can keep the merge method
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids
    # TODO add return docstrings
    # TODO add return type (MllamaCausalLMOutputWithPast)
    #@add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=MllamaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def llava_forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

        >>> model = MllamaForConditionalGeneration.from_pretrained("mllama-hf/mllama-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("mllama-hf/mllama-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Mllama + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MllamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
