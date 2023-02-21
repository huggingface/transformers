import collections.abc
import copy
import logging
import math
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from .configuration_udop import UdopConfig


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "UDOPConfig"
_TOKENIZER_FOR_DOC = "UDOPTokenizer"
# _CHECKPOINT_FOR_DOC = "t5-small"
UDOPDual_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UDOPDual_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

UDOPDual_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""
AUGMENTATION_RANGE = (0.80, 1.25)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, N_context, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, N_context, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, N_context, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_cross_attention=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
            )
            self.norm_ct = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if context is not None and self.use_cross_attention:
            x = x + self.drop_path(self.cross_attn(self.norm_ct(x), context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        ctx_embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        vocab_size=100,
        max_2d_position_embeddings=501,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, mask_ratio=None, ids_keep=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        if ids_keep is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W] pred: [N, L, p*p*3] mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(image_size, vocab_size, max_2d_position_embeddings, **kwargs):
    model = MaskedAutoencoderViT(
        img_size=image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size,
        max_2d_position_embeddings=max_2d_position_embeddings,
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(image_size, vocab_size, max_2d_position_embeddings, **kwargs):
    model = MaskedAutoencoderViT(
        img_size=image_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size,
        max_2d_position_embeddings=max_2d_position_embeddings,
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(image_size, vocab_size, max_2d_position_embeddings, **kwargs):
    model = MaskedAutoencoderViT(
        img_size=image_size,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size,
        max_2d_position_embeddings=max_2d_position_embeddings,
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


def mae_model(name, pretrained_weights, image_size, vocab_size, max_2d_position_embeddings, **kwargs):
    mae_models = {
        "mae_vit_base_patch16": mae_vit_base_patch16,
        "mae_vit_large_patch16": mae_vit_large_patch16,
        "mae_vit_huge_patch14": mae_vit_huge_patch14,
    }

    if name not in mae_models:
        raise RuntimeError(f"{name} is not available")

    model = mae_models[name](
        image_size=image_size, vocab_size=vocab_size, max_2d_position_embeddings=max_2d_position_embeddings
    )
    return model


def get_relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


def collate_vlembed(
    inputs_patches,
    inputs_embeds,
    seg_data,
    visual_segdata,
    vis_special_token=None,
    attention_mask=None,
    num_patches=14,
    max_len=0,
):

    L = num_patches
    ocr_points_x = torch.clip(torch.floor((seg_data[:, :, 0] + seg_data[:, :, 2]) / 2.0 * L).long(), 0, L - 1)
    ocr_points_y = torch.clip(torch.floor((seg_data[:, :, 1] + seg_data[:, :, 3]) / 2.0 * L).long(), 0, L - 1) * L
    ocr_points = ocr_points_x + ocr_points_y
    target_seg = (seg_data.mean(-1) == 0.0) | (seg_data.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(
        inputs_patches, 1, ocr_points.unsqueeze(-1).repeat(1, 1, inputs_patches.size(-1))
    )
    repeated_vision_embeds[target_seg] = 0.0
    inputs_embeds += repeated_vision_embeds

    patch_inds = torch.full_like(inputs_patches[:, :, 0], True).bool()
    ind = torch.cat(
        [
            torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
            ocr_points[:, :, None],
        ],
        -1,
    ).flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False

    input_vision_patches = [inputs_patches[i][patch_inds[i]] for i in range(len(patch_inds))]
    visual_segdata = [visual_segdata[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_segdata]

    if max_len == 0:
        max_len = inputs_patches.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(inputs_patches[0, 0])) for item in input_vision_patches]
    )
    visual_segdata = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(seg_data[0, 0])) for item in visual_segdata]
    )
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask]
        )

    if vis_special_token is not None:
        inputs_vision_patches += vis_special_token

    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    seg_data = torch.cat([seg_data, visual_segdata], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return inputs_embeds, seg_data, attention_mask


class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases :param num_heads: number of heads in lm model, it will create embeddings of size
    `num_heads`,
        which will be added to scores per each token pair
    :param relative_attention_num_buckets: pair token metric
        (distance in the sequence, distance in pixels etc.) will be bucketed, parameter is defining number of such
        buckets
    :param bidirectional: defining if for pair of tokens distance should be bidirecional,
        if bidirectional=False, then distance(tok1, tok2) == distance(tok2, tok1)
    :param scaling_factor: defining factor which will be used to scale relative distance :param max_distance: all
    distances above this value will end up in the one/same bucket :param augmentation: whether to multiple relative
    distances by random scalar :param expand: used for re-using pretrained model with subsequent addition of
    prefix_bucket
    """

    def __init__(
        self,
        num_heads=None,
        relative_attention_num_buckets=32,
        bidirectional=True,
        scaling_factor=1,
        max_distance=128,
        level="tokens",
        augmentation=False,
        prefix_bucket=False,
        expand=False,
    ):

        super(RelativePositionBiasBase, self).__init__()
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and not self.expand else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    def get_required_segment_levels(self) -> Sequence[str]:
        return [self.level]

    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Optional[Tensor] = None,
        seg_data: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        pass

    def get_bucket(self, attention_mask: Optional[Tensor] = None, seg_data: Optional[Dict[str, Any]] = None) -> Tensor:
        relative_position = self.prepare_input(attention_mask, seg_data)
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)

    def forward(self, attention_mask: Optional[Tensor] = None, seg_data: Optional[Dict[str, Any]] = None) -> Tensor:

        # re-using pretrained model with subsequent addition of prefix_bucket
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
            new_bias.weight.data[: self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets :] = 0.1
            self.relative_attention_bias = new_bias
            self.expand = False

        rp_bucket = self.get_bucket(attention_mask, seg_data)

        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
                rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
            # based on assumption that prefix bboxes are negative
            is_prefix = seg_data[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

        values: Tensor = self.relative_attention_bias(rp_bucket)
        assert values.dim() == 4, "Wrong dimension of values tensor"
        values = values.permute([0, 3, 1, 2])

        return values


class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is their distance in the sequence.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(
        self, attention_mask: Optional[Tensor] = None, seg_data: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        assert self.scaling_factor == 1, "No need to scale 1d features"
        relative_position = self.get_relative_position(
            torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)[None, :]
        )

        return relative_position


class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class will sums up computed biases :param modules: list of relative bias modules
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(
        self,
        attention_mask: Optional[Tensor] = None,
        seg_data: Optional[Dict[str, Any]] = None,
    ) -> Union[float, Tensor]:
        x = 0.0
        for bias in self.biases:  # type: ignore
            x = bias(attention_mask, seg_data) + x

        return x


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(
        self, attention_mask: Optional[Tensor] = None, seg_data: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        assert self.scaling_factor > 1.0, "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get x positions of left point of bbox
        assert seg_data is not None
        horizontal_position: Tensor = seg_data[:, :, [0, 2]].mean(dim=-1)

        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens. Parameters are the same as in base
        class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(
        self, attention_mask: Optional[Tensor] = None, seg_data: Optional[Dict[str, Any]] = None
    ) -> Tensor:
        assert self.scaling_factor > 1.0, "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get y positions of middle of bbox
        assert seg_data is not None
        vertical_position: Tensor = seg_data[:, :, [1, 3]].mean(dim=-1)

        return self.get_relative_position(vertical_position)


BIAS_CLASSES = {
    "1d": RelativePositionBias1D,
    "horizontal": RelativePositionBiasHorizontal,
    "vertical": RelativePositionBiasVertical,
}


def create_relative_bias(config: UdopConfig) -> Sequence[RelativePositionBiasBase]:
    """
    Creates empty list or one/multiple relative biases.

    :param config: Model's configuration :return: Sequence with created bias modules.
    """
    bias_list = []
    if hasattr(config, "relative_bias_args"):
        assert isinstance(config.relative_bias_args, list)
        for bias_kwargs_org in config.relative_bias_args:
            bias_kwargs = copy.deepcopy(bias_kwargs_org)
            bias_type = bias_kwargs.pop("type")
            model_num_heads = config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
            if "num_heads" in bias_kwargs:
                assert (
                    bias_kwargs["num_heads"] == model_num_heads
                ), "Number of heads must match num of heads in the model"
            else:
                bias_kwargs["num_heads"] = model_num_heads
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))  # type: ignore

    return bias_list


@dataclass
class BaseModelOutputWithVisionEmbeds(BaseModelOutput):
    """
    Args:
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model. If `past_key_values` is used only
            the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or
        when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`. Contains pre-computed hidden-states (key and values in the
            self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks)
            that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and
        `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_embeds: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    seg_data: torch.FloatTensor = None


@dataclass
class VisSeq2SeqLMOutput(BaseModelOutput):

    loss: Optional[torch.FloatTensor] = None
    image_output: Optional[Tuple[torch.FloatTensor]] = None
    image_target: Optional[Tuple[torch.FloatTensor]] = None
    image_mask_label: Optional[Tuple[torch.FloatTensor]] = None


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class UDOPDenseActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UDOPDenseGatedActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        if hidden_states.dtype != self.wo.weight.dtype:
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class UDOPDualLayerFF(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = UDOPDenseGatedActDense(config)
        else:
            self.DenseReluDense = UDOPDenseActDense(config)

        self.layer_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class UDOPAttention(nn.Module):
    def __init__(self, config: UdopConfig, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class UDOPDualLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = UDOPAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class UDOPDualLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = UDOPAttention(config, has_relative_attention_bias=False)
        self.layer_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class UDOPBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(UDOPDualLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(UDOPDualLayerCrossAttention(config))

        self.layer.append(UDOPDualLayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class UDOPLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the UDOP style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # UDOP uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class UDOPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UdopConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["UDOPDualBlock"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, UDOPLayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (UDOPDualForConditionalGeneration, UDOPUnimodelForConditionalGeneration),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, UDOPDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UDOPDenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, UDOPAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UDOPAttention, UDOPDualStack, UDOPUniStack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In UDOPDual it is usually set to the"
            " pad_token_id. See UDOPDual docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class CellEmbeddings(UDOPPreTrainedModel):
    def __init__(self, config):
        super(CellEmbeddings, self).__init__(config)
        self.ccat = config.ccat
        self.max_2d_position_embeddings = config.max_2d_position_embeddings
        if self.ccat:
            self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 4)
            self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 4)
        else:
            self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
            self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)

        self.post_init()

    def forward(self, bbox):
        bbox = torch.clip(bbox, 0.0, 1.0)
        bbox = (bbox * (self.max_2d_position_embeddings - 1)).long()
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        if self.ccat:
            embeddings = torch.cat(
                [
                    left_position_embeddings,
                    upper_position_embeddings,
                    right_position_embeddings,
                    lower_position_embeddings,
                ],
                dim=-1,
            )
        else:
            embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
            )

        return embeddings


class UDOPDualStack(UDOPPreTrainedModel):
    """
    Almost exact copy of transformers UDOPDualStack with the modification of passing `position_bias` in the forward
    method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        setattr(config, "output_attentions", True)
        if self.is_decoder:
            self.num_layers = (
                config.truncate_decoder_after_layer if config.truncate_decoder_after_layer else config.num_layers
            )
        else:
            self.num_layers = (
                config.truncate_encoder_after_layer if config.truncate_encoder_after_layer else config.num_layers
            )

        self.block = nn.ModuleList(
            [UDOPBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config)

        self.init_weights()

        if not self.is_decoder:
            self.vision_encoder = mae_model(
                config.mae_version,
                config.mae_checkpoint,
                config.image_size,
                config.vocab_size,
                config.max_2d_position_embeddings,
            )
        else:
            self.vision_fc = nn.Linear(config.d_model, config.d_model)
            self.vision_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        vision_embeds=None,
        head_mask=None,
        past_key_values=None,
        image=None,
        ids_keep=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_bias=None,  # modified line,
        seg_data=None,  # modified line
        cross_attn_head_mask=None,
        token_type_ids=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            True  # False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if image is not None:
            vision_embeds = self.vision_encoder.forward_encoder(image, ids_keep=ids_keep)
        else:
            vision_embeds = None
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data["tokens"]["bboxes"] = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning("Empty batch")
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
            if seg_data is not None:
                inputs_embeds += self.cell2dembedding(seg_data)
            if token_type_ids is not None:
                inputs_embeds += self.token_type_embedding(token_type_ids)

        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds
        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            # We share the position biases between the layers - the first layer store them

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            # import pdb; pdb.set_trace()

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    vision_embeds,
                ]
                if v is not None
            )

        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            vision_embeds=vision_embeds,
        )


# class UDOPModel(UDOPPreTrainedModel):
#     _keys_to_ignore_on_load_missing = [
#         r"encoder.embed_tokens.weight",
#         r"decoder.embed_tokens.weight",
#     ]
#     _keys_to_ignore_on_load_unexpected = [
#         r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
#     ]
#
#     def __init__(self, config: UdopConfig):
#         super().__init__(config)
#         self.shared = nn.Embedding(config.vocab_size, config.d_model)
#
#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = UDOPDualStack(encoder_config, self.shared)
#
#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = UDOPDualStack(decoder_config, self.shared)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#
#     @add_start_docstrings(PARALLELIZE_DOCSTRING)
#     def parallelize(self, device_map=None):
#         self.device_map = (
#             get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
#             if device_map is None
#             else device_map
#         )
#         assert_device_map(self.device_map, len(self.encoder.block))
#         self.encoder.parallelize(self.device_map)
#         self.decoder.parallelize(self.device_map)
#         self.model_parallel = True
#
#     @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
#     def deparallelize(self):
#         self.encoder.deparallelize()
#         self.decoder.deparallelize()
#         self.encoder = self.encoder.to("cpu")
#         self.decoder = self.decoder.to("cpu")
#         self.model_parallel = False
#         self.device_map = None
#         torch.cuda.empty_cache()
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, new_embeddings):
#         self.shared = new_embeddings
#         self.encoder.set_input_embeddings(new_embeddings)
#         self.decoder.set_input_embeddings(new_embeddings)
#
#     def get_encoder(self):
#         return self.encoder
#
#     def get_decoder(self):
#         return self.decoder
#
#     def _prune_heads(self, heads_to_prune):
#         """
# Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base # class
PreTrainedModel #"""
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)
#
#     @add_start_docstrings_to_model_forward(UDOPDual_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         decoder_inputs_embeds: Optional[torch.Tensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
#         r"""
# Returns: # # Example: # # ```python # >>> from transformers import T5Tokenizer, T5Model # # >>> tokenizer =
# T5Tokenizer.from_pretrained("t5-small") # >>> model = T5Model.from_pretrained("t5-small") # # >>> input_ids =
# tokenizer( # ... "Studies have been shown that owning a dog is good for you", return_tensors="pt" # ... ).input_ids #
# Batch size 1 # >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids # Batch size 1 # #
# >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model. # >>> # This is not needed
# for torch's T5ForConditionalGeneration as it does this internally using labels arg. # >>> decoder_input_ids =
# model._shift_right(decoder_input_ids) # # >>> # forward pass # >>> outputs = model(input_ids=input_ids,
# decoder_input_ids=decoder_input_ids) # >>> last_hidden_states = outputs.last_hidden_state # ```"""
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask
#
#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#
#         hidden_states = encoder_outputs[0]
#
#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
#
#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         if not return_dict:
#             return decoder_outputs + encoder_outputs
#
#         return Seq2SeqModelOutput(
#             last_hidden_state=decoder_outputs.last_hidden_state,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )


# class UDOPEncoderModel(UDOPPreTrainedModel):
#     _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]
#
#     def __init__(self, config: UdopConfig):
#         super().__init__(config)
#         self.shared = nn.Embedding(config.vocab_size, config.d_model)
#
#         encoder_config = copy.deepcopy(config)
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = UDOPDualStack(encoder_config, self.shared)
#
#         # Initialize weights and apply final processing
#         self.post_init()
#
#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None
#
#     @add_start_docstrings(PARALLELIZE_DOCSTRING)
#     def parallelize(self, device_map=None):
#         self.device_map = (
#             get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
#             if device_map is None
#             else device_map
#         )
#         assert_device_map(self.device_map, len(self.encoder.block))
#         self.encoder.parallelize(self.device_map)
#         self.model_parallel = True
#
#     @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
#     def deparallelize(self):
#         self.encoder.deparallelize()
#         self.encoder = self.encoder.to("cpu")
#         self.model_parallel = False
#         self.device_map = None
#         torch.cuda.empty_cache()
#
#     def get_input_embeddings(self):
#         return self.shared
#
#     def set_input_embeddings(self, new_embeddings):
#         self.shared = new_embeddings
#         self.encoder.set_input_embeddings(new_embeddings)
#
#     def get_encoder(self):
#         return self.encoder
#
#     def _prune_heads(self, heads_to_prune):
#         """
# Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base # class
PreTrainedModel #"""
#         for layer, heads in heads_to_prune.items():
#             self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)
#
#     @add_start_docstrings_to_model_forward(UDOPDual_ENCODER_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
#         r"""
# Returns: # # Example: # # ```python # >>> from transformers import T5Tokenizer, T5EncoderModel # # >>> tokenizer =
# T5Tokenizer.from_pretrained("t5-small") # >>> model = UDOPEncoderModel.from_pretrained("t5-small") # >>> input_ids =
# tokenizer( # ... "Studies have been shown that owning a dog is good for you", return_tensors="pt" # ... ).input_ids #
# Batch size 1 # >>> outputs = model(input_ids=input_ids) # >>> last_hidden_states = outputs.last_hidden_state # ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         encoder_outputs = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             inputs_embeds=inputs_embeds,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         return encoder_outputs


class UDOPDualForConditionalGeneration(UDOPPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.cell2dembedding.x_position_embeddings",
        r"decoder.cell2dembedding.y_position_embeddings",
        r"decoder.embed_tokens.weight",
        r"encoder.embed_tokens.weight",
        r"decoder.relative_bias.biases.0.relative_attention_bias.weight",
        r"encoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    def __init__(self, config):
        super(UDOPDualForConditionalGeneration, self).__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256
        self.config.decoder_start_token_id = self.config.pad_token_id
        self.model_dim = self.config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UDOPDualStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UDOPDualStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder

        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.encoder.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        self.post_init()

        self.model_parallel = False
        self.device_map = None

    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    @staticmethod
    def _get_relative_bias(config: UdopConfig) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def set_input_embeddings(self, value):
        self.shared = value

    def get_output_embeddings(self):
        return self.shared

    def get_input_embeddings(self):
        return self.shared

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        ids_keep: Optional[Tensor] = None,
        ids_restore: Optional[Tensor] = None,
        image_mask_label: Optional[Tensor] = None,
        mask_ratio: Optional[Tensor] = None,
        seg_data: Dict[str, Any] = None,
        labels: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        char_ids: Optional[Tensor] = None,
        char_seg_data: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_dict: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Tensor, Seq2SeqLMOutput]:

        if seg_data is not None:
            seg_data = torch.clip(seg_data, 0.0, 1.0)
        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs
        if attention_mask is None:
            batch_size, seq_length = input_ids.size()
            mask_seq_length = (
                past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
            )
            attention_mask = torch.ones(batch_size, mask_seq_length).to(input_ids.device)
        if encoder_outputs is None:
            # compute positional bias (can be aggregation of 1D and 2D biases)
            encoder_position_bias = self.relative_bias(attention_mask=attention_mask, seg_data=seg_data)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_bias=encoder_position_bias,
                seg_data=seg_data,
                image=image,
                ids_keep=ids_keep,
            )

        if encoder_outputs is None:
            return None

        if masked_lm_labels is not None:
            labels = masked_lm_labels

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if encoder_outputs.vision_embeds is not None:
            vision_embeds = self.decoder.vision_fc(encoder_outputs.vision_embeds)
            vision_embeds = self.decoder.vision_norm(vision_embeds)
            hidden_states = torch.cat([encoder_outputs.last_hidden_state, vision_embeds], 1)
            encoder_outputs.last_hidden_state = hidden_states
            encoder_outputs.vision_embeds = None

        # ugly hack for model to work as an encoder
        if decoder_input_ids is None and labels is None:
            return encoder_outputs

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(
                    encoder_outputs.last_hidden_state[
                        :, : encoder_outputs.last_hidden_state.size(1) - attention_mask.size(1), 0
                    ]
                ),
            ],
            1,
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_encoder(self):
        return self


class UDOPUniStack(UDOPPreTrainedModel):
    """
    Almost exact copy of transformers T5Stack with the modification of passing `position_bias` in the forward method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length

        setattr(config, "output_attentions", True)
        if self.is_decoder:
            self.num_layers = (
                config.truncate_decoder_after_layer if config.truncate_decoder_after_layer else config.num_layers
            )
        else:
            self.num_layers = (
                config.truncate_encoder_after_layer if config.truncate_encoder_after_layer else config.num_layers
            )

        self.block = nn.ModuleList(
            [UDOPBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = UDOPLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        self.init_weights()

    @staticmethod
    def _get_relative_bias(config: UdopConfig) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        ids_keep=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cross_attn_head_mask=None,
        position_bias=None,  # modified line,
        inputs_patches=None,  # modified line,
        seg_data=None,  # modified line,
        visual_seg_data=None,  # modified line,
        num_patches=None,  # modified line,
        special_vis_token=None,  # modified line,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            True  # False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ======================================================
        # input embeddings processing

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning("Empty batch")
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if inputs_patches is not None:
            # combine OCR text and visual embed
            inputs_embeds, seg_data, attention_mask = collate_vlembed(
                inputs_patches,
                inputs_embeds,
                seg_data,
                visual_seg_data,
                special_vis_token,
                attention_mask,
                num_patches,
                0,
            )
            input_shape = inputs_embeds.size()[:-1]

        if not self.is_decoder:
            inputs_embeds += self.cell2dembedding(seg_data)

        batch_size, seq_length = input_shape

        # ======================================================
        # input masking/pos embed processing

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(attention_mask=attention_mask, seg_data=seg_data)
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        # ======================================================
        # model inferencing

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    attention_mask,
                    seg_data,
                ]
            )

        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            seg_data=seg_data,
        )


class UDOPUnimodelForConditionalGeneration(UDOPPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.cell2dembedding.x_position_embeddings.weight",
        r"decoder.cell2dembedding.y_position_embeddings.weight",
        r"decoder.embed_tokens.weight",
        r"encoder.embed_tokens.weight",
        r"decoder.relative_bias.biases.0.relative_attention_bias.weight",
        r"encoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data. :param config: a
    `T5Config` instance
    """

    def __init__(self, config):
        super(UDOPUnimodelForConditionalGeneration, self).__init__(config)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256

        self.config.decoder_start_token_id = self.config.pad_token_id

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UDOPUniStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UDOPUniStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.model_dim = config.d_model
        # --------------------------------------------------------------------------
        # MAE encoder specifics

        mae_model_tmp = mae_model(
            config.mae_version,
            config.mae_checkpoint,
            config.image_size,
            config.vocab_size,
            config.max_2d_position_embeddings,
        )

        self.patch_embed = mae_model_tmp.patch_embed

    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def get_output_embeddings(self):
        return self.shared

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        ids_keep: Optional[Tensor] = None,
        ids_restore: Optional[Tensor] = None,
        seg_data: Dict[str, Any] = None,
        visual_seg_data: Dict[str, Any] = None,
        masked_lm_labels: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        input_dict: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Tensor, ...]:

        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs

        num_patches = None
        if encoder_outputs is None:
            inputs_patches = None
            if image is not None:
                assert visual_seg_data is not None
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                if ids_keep is not None:
                    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                    pad_tokens = self.pad_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                    x_padded = torch.cat([x, pad_tokens], dim=1)  # no cls token
                    x_padded = torch.gather(
                        x_padded, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_padded.shape[2])
                    )
                    inputs_patches = x_padded
                else:
                    inputs_patches = x

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                inputs_patches=inputs_patches,
                num_patches=num_patches,
                special_vis_token=None,
                ids_keep=ids_keep,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if encoder_outputs is None:
            return None

        if masked_lm_labels is not None and labels is None:
            labels = masked_lm_labels

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # ugly hack for model to work as an encoder
        if decoder_input_ids is None and masked_lm_labels is None:
            return encoder_outputs

        # outputs = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=encoder_outputs.attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     encoder_outputs=encoder_outputs,
        #     past_key_values=past_key_values,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     decoder_inputs_embeds=decoder_inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        attention_mask = encoder_outputs.attention_mask if return_dict else encoder_outputs[5]
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        return outputs  # type: ignore

    def get_encoder(self):
        return self
