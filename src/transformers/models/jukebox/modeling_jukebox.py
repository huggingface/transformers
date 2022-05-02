# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch Jukebox model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_jukebox import JukeboxConfig
import gc

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "huggingface/jukebox-dummy"
_CONFIG_FOR_DOC = "JukeboxConfig"
_TOKENIZER_FOR_DOC = "JukeboxTokenizer"

JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "huggingface/jukebox-dummy",
    # See all Jukebox models at https://huggingface.co/models?filter=jukebox
]



def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    



####################################################################
# Attention and scalable transformer
# Import FusedLayerNorm if we have apex, otherwise use regular LayerNorm
try:
    from apex.normalization import FusedLayerNorm

    print("Using apex FusedLayerNorm")
except ImportError:
    from torch.nn import LayerNorm as FusedLayerNorm

class JukeboxMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        init_scale = config.init_scale
        self.c_fc = Conv1D(intermediate_size, embed_dim, init_scale=init_scale)
        self.c_proj = Conv1D(embed_dim, intermediate_size, init_scale=init_scale)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class LayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(input.float(), self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            return super(LayerNorm, self).forward(input.float()).type_as(input)


def repeat(x, n, dim):
    if dim == -1:
        dim = len(x.shape) - 1
    return (
        x.view(int(np.prod(x.shape[: dim + 1])), 1, int(np.prod(x.shape[dim + 1 :])))
        .repeat(1, n, 1)
        .view(*x.shape[:dim], n * x.shape[dim], *x.shape[dim + 1 :])
    )


def get_mask(mask, q_l, kv_l, blocks, spread, device, sample, sample_t):
    # returns a mask of shape 1 x 1 x q_l x kv_l or None if masking is not needed.
    if mask is None or q_l == 1:
        return None
    offset = sample_t - q_l if sample else max(kv_l - q_l, 0)
    if mask == "autoregressive":
        # Masked dense
        mask = torch.ones(q_l, kv_l, device=device).tril(offset)
    elif mask == "summary":
        # Masked summary
        mask = (
            torch.nn.functional.pad(
                torch.ones(q_l, q_l, device=device).tril().view(q_l, blocks, q_l // blocks)[:, :-1, -kv_l // blocks :],
                (0, 0, 1, 0),
                value=1,
            )
            .contiguous()
            .view(q_l, kv_l)
        )
    elif mask == "prime":
        mask = torch.ones(q_l, kv_l, device=device).tril(offset)
    return mask.view(1, 1, q_l, kv_l)


class JukeboxAttention(nn.Module):
    # previously FactoredAttention
    def __init__(
        self,
        n_in,
        n_ctx,
        n_state,
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        checkpoint_attn=0,
        attn_func=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx  # NOTE: n_ctx could be different within operations. This is complete n_ctx
        self.n_state = n_state
        assert n_state % n_head == 0
        self.n_head = n_head
        self.scale = scale
        self.mask = mask
        if attn_func == 6:
            self.c_attn = Conv1D(n_in, n_state, init_scale=init_scale)
            self.c_enc_kv = Conv1D(n_in, n_state * 2, init_scale=init_scale)
        else:
            self.c_attn = Conv1D(n_in, n_state * 3, init_scale=init_scale)
        self.c_proj = Conv1D(n_state, n_in, zero_out, init_scale=init_scale)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else lambda x: x
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0.0 else lambda x: x

        # Sequence of length l is factored as [blocks, l // blocks]
        self.attn_func = attn_func
        self.qkv, self.attn, self.attn_mask = {
            0: (self.factored_qkv, self.dense_attn, "autoregressive"),  # Attend to all positions
            1: (self.factored_qkv, self.block_attn, "autoregressive"),  # Attend to your block
            2: (self.factored_qkv, self.transpose_block_attn, "autoregressive"),  # Attend to transpose block
            3: (self.factored_qkv, self.prev_block_attn, None),  # Attend to previous block
            4: (self.factored_qkv, self.summary_attn, "summary"),  # Attend to last position of each block
            5: (self.factored_qkv, self.summary_spread_attn, "summary"),
            6: (self.decode_qkv, self.decode_attn, None),
            7: (self.prime_qkv, self.prime_attn, "prime"),
        }[
            attn_func
        ]  # Attend to last k position of each block

        self.blocks = blocks
        self.spread = spread
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.checkpoint_attn = checkpoint_attn  # 0: None, 1: Attn after heads split, 2: Attn

        self.sample_t = 0
        self.cache = {}
        self.encoder_dims = encoder_dims
        self.prime_len = prime_len
        self.record_attn = False
        self.w = None

    def _attn(self, q, k, v, sample):
        scale = 1.0 / math.sqrt(math.sqrt(self.n_state // self.n_head))
        if self.training:
            w = torch.matmul(q * scale, k * scale)
        else:
            w = torch.matmul(q, k)
            w.mul_(scale * scale)
        wtype = w.dtype
        w = w.float()
        if self.mask:
            # Generate appropriate mask to mask out all positions before current
            # Might take up lot of memory for dense, so can cache it
            mask = get_mask(
                self.attn_mask, q.size(-2), k.size(-1), self.blocks, self.spread, w.device, sample, self.sample_t
            )
            if mask is not None:
                # print(mask)
                w = w * mask + -1e9 * (1 - mask)
            w = F.softmax(w, dim=-1).type(wtype)
        else:
            w = F.softmax(w, dim=-1).type(wtype)
        if self.record_attn:
            self.w = w  # .float().cpu().numpy()
            if self.attn_func == 7:
                # only keep music queries and lyrics keys/values
                self.w = self.w[:, :, self.prime_len :, : self.prime_len]
        w = self.attn_dropout(w)
        a = torch.matmul(w, v)
        return a

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = (*x.size()[:-2], x.size(-2) * x.size(-1))
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = (*x.size()[:-1], self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def dense_attn(self, query, key, value, sample):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        # if self.checkpoint_attn == 1 and not sample:
        #     a = checkpoint(lambda q,k,v,s=sample: self._attn(q,k,v,s), (query, key, value),
        #                (), True)
        # else:
        a = self._attn(query, key, value, sample)
        a = self.merge_heads(a)
        return a

    def block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert l == self._suff_cache_len(), f"{l} != {self._suff_cache_len()}"
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            if ql < l:
                l = ql
                k = k[:, -l:].contiguous()
                v = v[:, -l:].contiguous()
            k = k.view(bs * l // block_ctx, block_ctx, d)
            v = v.view(bs * l // block_ctx, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def transpose_block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            block_l = (l - 1) % block_ctx
            k = k[:, block_l::block_ctx, :]
            v = v[:, block_l::block_ctx, :]
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = (
                q.view(bs, ql // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, ql // block_ctx, d)
            )
            k = (
                k.view(bs, l // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, l // block_ctx, d)
            )
            v = (
                v.view(bs, l // block_ctx, block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs * block_ctx, l // block_ctx, d)
            )
            return (
                self.dense_attn(q, k, v, sample)
                .view(bs, block_ctx, ql // block_ctx, d)
                .transpose(1, 2)
                .contiguous()
                .view(bs, ql, d)
            )

    def prev_block_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert l == self._suff_cache_len(), f"{l} != {self._suff_cache_len()}"
            block = (l - 1) // block_ctx
            prev_l = (block - 1) * block_ctx
            if block > 0:
                assert prev_l == 0
                k = k[:, prev_l : prev_l + block_ctx, :]
                v = v[:, prev_l : prev_l + block_ctx, :]
            else:
                k = torch.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
                v = torch.zeros(bs, block_ctx, d, device=q.device, dtype=q.dtype)
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            ql = q.shape[1]
            q = q.view(bs * ql // block_ctx, block_ctx, d)
            k = torch.nn.functional.pad(
                k.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)
            ).view(bs * l // block_ctx, block_ctx, d)
            v = torch.nn.functional.pad(
                v.view(bs, l // block_ctx, block_ctx, d)[:, :-1, :, :], (0, 0, 0, 0, 1, 0)
            ).view(bs * l // block_ctx, block_ctx, d)
            if ql < l:
                qb = ql // block_ctx
                kb = l // block_ctx
                l = ql
                k = k.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
                v = v.view(bs, kb, block_ctx, d)[:, -qb:].contiguous().view(bs * qb, block_ctx, d)
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_attn(self, q, k, v, sample):
        blocks, block_ctx = (
            self.blocks,
            self.block_ctx,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            k = torch.nn.functional.pad(k[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :], (0, 0, 1, 0))
            v = torch.nn.functional.pad(v[:, block_ctx - 1 : blocks * block_ctx - 1 : block_ctx, :], (0, 0, 1, 0))
            return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = torch.nn.functional.pad(
                k.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0)
            )  # bs, blocks, d
            v = torch.nn.functional.pad(
                v.view(bs, blocks, l // blocks, d)[:, :-1, -1, :], (0, 0, 1, 0)
            )  # bs, blocks, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def summary_spread_attn(self, q, k, v, sample):
        blocks, block_ctx, spread = (
            self.blocks,
            self.block_ctx,
            self.spread,
        )  # block_ctx is l // blocks for complete l ie l = n_ctx. Sampling has less l
        bs, l, d = v.shape  # For sample, q_l = 1, k_l = v_l = sample_t
        if sample:
            assert False, "Not yet implemented"
            # k = torch.nn.functional.pad(k,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # v = torch.nn.functional.pad(v,(0,0,block_ctx,(-l)%block_ctx)).view(bs, -1, block_ctx, d)[:,:-1,-spread:,:].contiguous().view(bs, -1, d)
            # return self.dense_attn(q, k, v, sample).view(bs, 1, d)
        else:
            k = (
                torch.nn.functional.pad(k.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0))
                .contiguous()
                .view(bs, blocks * spread, d)
            )  # bs, blocks * spread, d
            v = (
                torch.nn.functional.pad(v.view(bs, blocks, l // blocks, d)[:, :-1, -spread:, :], (0, 0, 0, 0, 1, 0))
                .contiguous()
                .view(bs, blocks * spread, d)
            )  # bs, blocks * spread, d
            return self.dense_attn(q, k, v, sample).view(bs, l, d)

    def prime_attn(self, q, k, v, sample):
        prime_len = self._prime_len
        k = k[:, :prime_len]
        v = v[:, :prime_len]
        return self.dense_attn(q, k, v, sample)

    def decode_attn(self, q, k, v, sample):
        assert (
            k.shape[1] == v.shape[1] == self.encoder_dims
        ), f"k: {k.shape}, v: {v.shape}, enc_dims: {self.encoder_dims}"
        return self.dense_attn(q, k, v, sample)

    def factored_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            self.sample_t += curr_ctx
            key, value = self._append_cache(key, value)
            l_cache = self._suff_cache_len()
            if self._cache_len() > l_cache:
                self._slice_cache(-l_cache)
            if curr_ctx > 1:
                if self.attn_func != 0:
                    query = self._pad_to_block_ctx(query, query=True)
                    key = self._pad_to_block_ctx(key)
                    value = self._pad_to_block_ctx(value)
                    assert key.shape[1] % self.block_ctx == 0
                    assert query.shape[1] % self.block_ctx == 0
                assert key.shape[1] == value.shape[1]
                assert query.shape[1] <= key.shape[1]
                sample = False
            else:
                key = self.cache["key"]
                value = self.cache["value"]
        return query, key, value, sample

    def prime_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is None
        query, key, value = x.chunk(3, dim=2)
        if sample:
            if self._cache_len() < self._prime_len:
                self._append_cache(key, value)
            if self._cache_len() > self._prime_len:
                self._slice_cache(0, self._prime_len)
            key, value = self.cache["key"], self.cache["value"]
            self.sample_t += curr_ctx
            assert (
                key.shape[1] == value.shape[1] == self._suff_cache_len()
            ), f"k: {key.shape}, v: {value.shape}, prime_dims: {self._suff_cache_len()}"
        else:
            assert (
                key.shape[1] == value.shape[1] == self.n_ctx
            ), f"k: {key.shape}, v: {value.shape}, prime_dims: {self.n_ctx}"
        assert key.shape[0] == value.shape[0] == query.shape[0], f"k: {key.shape}, v: {value.shape}, q: {query.shape}"
        assert key.shape[2] == value.shape[2] == query.shape[2], f"k: {key.shape}, v: {value.shape}, q: {query.shape}"
        return query, key, value, sample

    def decode_qkv(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        assert encoder_kv is not None
        query = x
        if sample:
            if self.sample_t == 0:
                self.cache["key"], self.cache["value"] = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
            key, value = self.cache["key"], self.cache["value"]
            self.sample_t += curr_ctx
        else:
            key, value = self.c_enc_kv(encoder_kv.type_as(x)).chunk(2, dim=2)
        assert key.shape[0] == value.shape[0] == query.shape[0], f"k: {key.shape}, v: {value.shape}, q: {query.shape}"
        assert (
            key.shape[1] == value.shape[1] == self.encoder_dims
        ), f"k: {key.shape}, v: {value.shape}, enc_dims: {self.encoder_dims}"
        assert key.shape[2] == value.shape[2] == query.shape[2], f"k: {key.shape}, v: {value.shape}, q: {query.shape}"
        return query, key, value, sample

    def forward(self, x, encoder_kv=None, sample=False):
        curr_ctx = x.shape[1]
        x = self.c_attn(x)
        query, key, value, sample = self.qkv(x, encoder_kv=encoder_kv, sample=sample)
        # if self.checkpoint_attn == 2 and not sample:
        #     a = checkpoint(lambda q,k,v,s=sample: self.attn(q,k,v,s), (query, key, value), (), True)
        # else:
        a = self.attn(query, key, value, sample)
        if a.shape[1] != curr_ctx:
            offset = self._offset(curr_ctx)
            a = a[:, offset : offset + curr_ctx, :].contiguous()
        a = self.c_proj(a)
        return self.resid_dropout(a)

    @property
    def _prime_len(self):
        prime_len = self.prime_len
        assert prime_len is not None
        prime_blocks = (prime_len // self.blocks) + 1
        return prime_blocks * self.blocks

    def _offset(self, curr_ctx):
        if self.attn_func == 0:
            return 0
        return (self.sample_t - curr_ctx) % self.block_ctx

    def _pad_to_block_ctx(self, x, query=False):
        l = x.shape[1]
        offset = self._offset(l) if query else 0
        n_blocks = (l + offset + self.block_ctx - 1) // self.block_ctx
        pad = n_blocks * self.block_ctx - l - offset
        if pad == 0 and offset == 0:
            return x
        else:
            return F.pad(x, (0, 0, offset, pad))

    def _cache_len(self):
        return 0 if "key" not in self.cache else self.cache["key"].shape[1]

    def _suff_cache_len(self):
        """
        Precondition:
            key and value are appended with the current context and self.sample_t reflects the 1-indexed sample
            location in the context.
        """
        if self.attn_func == 0:
            return self.sample_t
        elif self.attn_func == 1:
            return (self.sample_t - 1) % self.block_ctx + 1
        elif self.attn_func == 2:
            return self.sample_t
        elif self.attn_func == 3:
            if self.sample_t <= self.block_ctx:
                return self.sample_t
            else:
                curr_block = (self.sample_t - 1) % self.block_ctx + 1
                prev_block = self.block_ctx
                return curr_block + prev_block
        elif self.attn_func == 6:
            return self.encoder_dims
        elif self.attn_func == 7:
            return min(self.sample_t, self._prime_len)
        else:
            raise NotImplementedError()

    def _slice_cache(self, start, end=None):
        self.cache["key"] = self.cache["key"][:, start:end]
        self.cache["value"] = self.cache["value"][:, start:end]

    def _append_cache(self, key, value):
        if "key" not in self.cache:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            old_key, old_value = key, value
            key = torch.cat([self.cache["key"], key], dim=1)
            value = torch.cat([self.cache["value"], value], dim=1)
            del self.cache["key"]
            del self.cache["value"]
            del old_key
            del old_value
            self.cache["key"] = key
            self.cache["value"] = value
        return self.cache["key"], self.cache["value"]

    def del_cache(self):
        self.sample_t = 0
        if "key" in self.cache:
            del self.cache["key"]
        if "value" in self.cache:
            del self.cache["value"]
        self.cache = {}

    def check(self):
        blocks = self.blocks or 1
        spread = self.spread or 1
        bs, l, d = (4, self.n_ctx, self.n_in)
        x = torch.randn(bs, l, d).cuda()
        x.requires_grad = True
        x_out = self.forward(x)  # bs, l, d
        loss = x_out.mean(dim=-1)  # bs, l
        pos = 60
        grad = torch.autograd.grad(loss[2, pos], x)[0]

        assert grad.shape == (bs, l, d)
        assert (grad[:2] == 0).all()
        assert (grad[3:] == 0).all()
        assert (grad[2, (pos + 1) :] == 0).all()
        pos_grad = (torch.sum(grad[2] ** 2, dim=-1) > 0).nonzero().view(-1).cpu()

        block_pos = pos - (pos % (l // blocks))
        exp_pos_grad = {
            0: torch.arange(pos),
            1: torch.arange(block_pos, pos),
            2: torch.arange(pos % (l // blocks), pos, l // blocks),
            3: torch.arange(block_pos - l // blocks, block_pos),
            4: torch.arange(l // blocks - 1, pos, l // blocks),
            5: ((torch.arange(pos) % (l // blocks) >= (l // blocks - spread)) & (torch.arange(pos) < block_pos))
            .nonzero()
            .view(-1),
        }[self.attn_func]
        exp_pos_grad = torch.cat([exp_pos_grad, torch.tensor([pos])], dim=-1)

        assert (len(pos_grad) == len(exp_pos_grad)) and (
            pos_grad == exp_pos_grad
        ).all(), f"Expected pos grad {exp_pos_grad} got {pos_grad} for attn_func {self.attn_func} pos {pos} l {l} blocks {blocks}"

    def check_cache(self, n_samples, sample_t, fp16):
        assert self.sample_t == sample_t, f"{self.sample_t} != {sample_t}"
        if sample_t == 0:
            assert self.cache == {}
        else:
            dtype = {True: torch.float16, False: torch.float32}[fp16]
            l_cache = self._suff_cache_len()
            assert self.cache["key"].shape == (n_samples, l_cache, self.n_state)
            assert self.cache["value"].shape == (n_samples, l_cache, self.n_state)
            assert self.cache["key"].dtype == dtype, f"Expected {dtype}, got {self.cache['key'].dtype}"
            assert self.cache["value"].dtype == dtype, f"Expected {dtype}, got {self.cache['value'].dtype}"

    def check_sample(self):
        torch.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        prime = 5
        x = torch.randn(bs, l, d).cuda()
        xs = torch.chunk(x, l, dim=1)
        assert self.sample_t == 0
        assert self.cache == {}

        with torch.no_grad():
            enc_l = self.encoder_dims
            encoder_kv = None
            if self.attn_func == 6:
                encoder_kv = torch.randn(bs, enc_l, d).cuda()

            # Normal path
            x_out_normal = self.forward(x, encoder_kv=encoder_kv)

            # Sampling path
            x_out_sample = torch.cat(
                [self.forward(xs[i], encoder_kv=encoder_kv, sample=True) for i in range(l)], dim=1
            )
        max_err = torch.max(torch.abs(x_out_sample - x_out_normal))
        assert (
            max_err < 1e-8
        ), f"Max sampling err is {max_err} {[i for i in range(l) if torch.max(torch.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

        with torch.no_grad():
            x_out_normal = x_out_normal[:, :prime, :]
            # Prime sampling path
            self.del_cache()
            x_out_sample = self.forward(x[:, :prime, :].contiguous(), encoder_kv=encoder_kv, sample=True)
            self.check_cache(bs, prime, False)

        max_err = torch.max(torch.abs(x_out_sample - x_out_normal))
        assert (
            max_err < 1e-8
        ), f"Max prime sampling err is {max_err} {[i for i in range(prime) if torch.max(torch.abs(x_out_sample - x_out_normal)[:,i,:]) > 1e-8]}"

    def check_chunks(self, chunk_size):
        torch.manual_seed(42)
        bs, l, d = (4, self.n_ctx, self.n_in)
        enc_l = self.encoder_dims
        assert l % chunk_size == 0
        n_chunks = l // chunk_size
        with torch.no_grad():
            encoder_kv = None
            x = torch.randn(bs, l, d).cuda()
            if self.attn_func == 6:
                encoder_kv = torch.randn(bs, enc_l, d).cuda()

            self.del_cache()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=False)
            self.del_cache()
            y_forw_sample = self.forward(x, encoder_kv=encoder_kv, sample=True)
            max_err = torch.max(torch.abs(y_forw - y_forw_sample))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(y_forw - y_forw_sample)[:, i, :]) > 1e-6]}"

            self.del_cache()
            x_chunks = torch.chunk(x, n_chunks, dim=1)
            y_chunks = []
            total_len = 0
            for x_chunk in x_chunks:
                y_chunk = self.forward(x_chunk.contiguous(), encoder_kv=encoder_kv, sample=True)
                total_len += x_chunk.shape[1]
                self.check_cache(bs, total_len, False)
                y_chunks.append(y_chunk)
            y_forw_in_chunks = torch.cat(y_chunks, dim=1)

            max_err = torch.max(torch.abs(y_forw - y_forw_in_chunks))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


class JukeboxBlock(nn.module):
    # previously ResAttnBlock
    def __init__(
        self,
        n_in,
        n_ctx,
        n_head,
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn="gelu",
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        res_scale=1.0,
        m_attn=0.25,
        m_mlp=1.0,
        checkpoint_attn=0,
        checkpoint_mlp=0,
        attn_func=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.attn = JukeboxAttention(
            n_in=n_in,
            n_ctx=n_ctx,
            n_state=int(m_attn * n_in),
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            scale=scale,
            mask=mask,
            zero_out=zero_out,
            init_scale=init_scale,
            checkpoint_attn=checkpoint_attn,
            attn_func=attn_func,
            blocks=blocks,
            spread=spread,
            encoder_dims=encoder_dims,
            prime_len=prime_len,
        )
        self.ln_0 = LayerNorm(n_in)
        self.mlp = JukeboxMLP(
            n_in=n_in,
            n_state=int(m_mlp * n_in),
            resid_dropout=resid_dropout,
            afn=afn,
            zero_out=zero_out,
            init_scale=init_scale,
        )
        self.ln_1 = LayerNorm(n_in)
        self.res_scale = res_scale

        self.checkpoint_attn = checkpoint_attn
        self.checkpoint_mlp = checkpoint_mlp
        self.n_in = n_in
        self.attn_func = attn_func

    def forward(self, x, encoder_kv, sample=False):
        if sample:
            a = self.attn(self.ln_0(x), encoder_kv, sample)
            m = self.mlp(self.ln_1(x + a))
        else:
            # if self.attn_func == 6:
            #     assert encoder_kv is not None
            #     a = checkpoint(lambda _x,_enc_kv,_s=sample: self.attn(self.ln_0(_x),_enc_kv,_s),
            #                    (x,encoder_kv),
            #                    (*self.attn.parameters(), *self.ln_0.parameters()),
            #                    self.checkpoint_attn == 3)  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            # else:
            #     assert encoder_kv is None
            #     a = checkpoint(lambda _x,_enc_kv=None,_s=sample: self.attn(self.ln_0(_x),_enc_kv,_s),
            #                    (x,),
            #                    (*self.attn.parameters(), *self.ln_0.parameters()),
            #                    self.checkpoint_attn == 3)  # 2 recomputes after the projections, and 1 recomputes after head splitting.
            # m = checkpoint(lambda _x: self.mlp(self.ln_1(_x)), (x + a,),
            #                (*self.mlp.parameters(), *self.ln_1.parameters()),
            #                self.checkpoint_mlp == 1)
            pass
        if self.res_scale == 1.0:
            h = x + a + m
        else:
            h = x + self.res_scale * (a + m)
        return h


class JukeboxTransformer(nn.Module):
    def __init__(
        self,
        n_in,
        n_ctx,
        n_head,
        n_depth,
        attn_dropout=0.0,
        resid_dropout=0.0,
        afn="gelu",
        scale=True,
        mask=False,
        zero_out=False,
        init_scale=1.0,
        res_scale=False,
        m_attn=0.25,
        m_mlp=1.0,
        checkpoint_attn=0,
        checkpoint_mlp=0,
        checkpoint_res=0,
        attn_order=0,
        blocks=None,
        spread=None,
        encoder_dims=None,
        prime_len=None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_ctx = n_ctx
        self.encoder_dims = encoder_dims
        self.blocks = blocks
        if blocks is not None:
            assert n_ctx % blocks == 0
            self.block_ctx = n_ctx // blocks
        self.prime_len = prime_len
        self.n_head = n_head

        res_scale = 1.0 / n_depth if res_scale else 1.0

        # Orders of attn_func
        attn_func = {
            0: lambda d: 0,  # Complete dense attn
            1: lambda d: [1, 2][d % 2],  # Alternate row and column attn
            2: lambda d: [1, 2, 3][d % 3],  # Alternate row, column and previous row attn
            3: lambda d: [1, 4][d % 2],  # Alternate row and last column
            4: lambda d: [1, 5][d % 2],  # Alternate row and last k columns
            5: lambda d: [1, 4, 1, 1][d % 4],  # Alternate row, last column, row, row
            6: lambda d: [1, 2, 3, 6][d % 4],
            7: lambda d: [*[1, 2, 3] * 5, 6][d % 16],
            8: lambda d: [1, 2, 3, 1, 2, 3, 1, 2, 3, 6][d % 10],  # Used by separated_enc_dec model with lyrics
            9: lambda d: [1, 2, 3, 0][d % 4],
            10: lambda d: [*[1, 2, 3, 1, 2, 3, 1, 2, 3], *[1, 2, 3, 1, 2, 3, 1, 2, 3, 6] * 7][
                d % 79
            ],  # Used by large separated_enc_dec model with lyrics
            11: lambda d: [6, 6, 0][d % 3] if d % 16 == 15 else [1, 2, 3][d % 3],
            12: lambda d: [7, 7, 0][d % 3]
            if d % 16 == 15
            else [1, 2, 3][d % 3],  # Used by single_enc_dec model with lyrics
        }[attn_order]

        attn_cycle = {0: 1, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4, 6: 4, 7: 16, 8: 10, 9: 4, 10: 79, 11: 16, 12: 16}[attn_order]
        # assert n_depth % attn_cycle == 0, f'Depth {n_depth} not a multiple of cycle {attn_cycle} for attn_order {attn_order}'

        attn_block = lambda d: JukeboxBlock(
            n_in=n_in,
            n_ctx=n_ctx,
            n_head=n_head,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            afn=afn,
            scale=scale,
            mask=mask,
            zero_out=zero_out if attn_func(d) != 6 else True,
            init_scale=init_scale,
            res_scale=res_scale,
            m_attn=m_attn,
            m_mlp=m_mlp,
            checkpoint_attn=checkpoint_attn,
            checkpoint_mlp=checkpoint_mlp,
            attn_func=attn_func(d),
            blocks=blocks,
            spread=spread,
            encoder_dims=encoder_dims,
            prime_len=prime_len,
        )

        self.checkpoint_res = checkpoint_res
        self._attn_mods = nn.ModuleList()
        for d in range(n_depth):
            self._attn_mods.append(attn_block(d))
        self.ws = []

    def set_record_attn(self, record_attn):
        """
        Arguments:
            record_attn (bool or set): Makes forward prop dump self-attention
                softmaxes to self.ws. Either a set of layer indices indicating which layers to store, or a boolean
                value indicating whether to dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn

        for i, l in enumerate(self._attn_mods):
            l.attn.record_attn = _should_record_attn(i)
        if record_attn:
            assert self.ws == []
            for l in self._attn_mods:
                assert l.attn.w == None
        else:
            self.ws = []
            for l in self._attn_mods:
                l.attn.w = None

    def forward(self, x, encoder_kv=None, sample=False, fp16=False, fp16_out=False):
        if fp16:
            x = x.half()

        # Blocks
        for i, l in enumerate(self._attn_mods):
            # if self.checkpoint_res == 1 and not sample:
            #     if l.attn_func == 6:
            #         assert encoder_kv is not None
            #         f = functools.partial(l, sample=sample)
            #         x = checkpoint(f, (x, encoder_kv), l.parameters(), True)
            #     else:
            #         f = functools.partial(l, encoder_kv=None, sample=sample)
            #         x = checkpoint(f, (x,), l.parameters(), True)
            # else:
            if l.attn_func == 6:
                x = l(x, encoder_kv=encoder_kv, sample=sample)
            else:
                x = l(x, encoder_kv=None, sample=sample)
            if l.attn.record_attn:
                self.ws.append(l.attn.w)
        if not fp16_out:
            x = x.float()
        return x

    def check_cache(self, n_samples, sample_t, fp16):
        for l in self._attn_mods:
            l.attn.check_cache(n_samples, sample_t, fp16)

    def del_cache(self):
        for l in self._attn_mods:
            l.attn.del_cache()

    def check_sample(self):
        bs, l, s, d = (4, self.n_ctx, self.encoder_dims, self.n_in)
        prime = 5
        with torch.no_grad():
            encoder_kv = torch.randn(bs, s, d).cuda()
            x = torch.randn(bs, l, d).cuda()
            y_forw = self.forward(x, encoder_kv=encoder_kv, sample=True)

            self.del_cache()
            x_chunks = torch.chunk(x, 4, dim=1)
            y_chunks = []
            n = 0
            for x_chunk in x_chunks:
                self.check_cache(bs, n, False)
                y_chunk = self.forward(x_chunk, encoder_kv=encoder_kv, sample=True)
                y_chunks.append(y_chunk)
                n += x_chunk.shape[1]
            self.check_cache(bs, n, False)
            y_forw_in_chunks = torch.cat(y_chunks, dim=1)

            max_err = torch.max(torch.abs(y_forw - y_forw_in_chunks))
            assert (
                max_err <= 1e-6
            ), f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(y_forw - y_forw_in_chunks)[:, i, :]) > 1e-6]}"


##################################
# Jukebox Prior ##################

def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    #assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))  # Safety check
    assert (top_k == 0) or (top_p == 0.0)
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        #indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def get_normal(*shape, std=0.01):
    w = torch.empty(shape)
    nn.init.normal_(w, std=std)
    return w

def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
    assert sum(chunk_sizes) == length
    return chunk_sizes

class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer('pos', torch.tensor(get_pos_idx(input_shape)).long())
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        if self.pos_init:
            pos_emb = sum([self._pos_embs[i](self.pos[:,i]) for i in range(len(self.input_shape))])
        else:
            pos_emb = self.pos_emb
        return pos_emb

class JukeboxPrior(nn.Module):
    # previously ConditionalAutoregressive2D, renamed it to prior
    def __init__(self, input_shape, bins,
                 width=128, depth=2, heads=1,
                 attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, mask=True,
                 zero_out=False, init_scale=1.0, res_scale=False, pos_init=False,
                 m_attn=0.25, m_mlp=1,
                 checkpoint_res=0, checkpoint_attn=0, checkpoint_mlp=0,
                 attn_order=0, blocks=None, spread=None, x_cond=False, y_cond=False,
                 encoder_dims=0, only_encode=False, merged_decoder=False, prime_len=None):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth

        self.x_emb = nn.Embedding(bins, width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond:
            self.start_token = nn.Parameter(get_normal(1, width, std=0.01 * init_scale))

        self.pos_emb = PositionEmbedding(input_shape=input_shape, width=width, init_scale=init_scale, pos_init=pos_init)
        self.pos_emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = JukeboxTransformer(n_in=width, n_ctx=input_dims, n_head=heads, n_depth=depth,
                                       attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                                       afn='relu', scale=True, mask=mask,
                                       zero_out=zero_out, init_scale=init_scale, res_scale=res_scale,
                                       m_attn=m_attn, m_mlp=m_mlp,
                                       checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, checkpoint_res=checkpoint_res,
                                       attn_order=attn_order, blocks=blocks, spread=spread,
                                       encoder_dims=encoder_dims, prime_len=prime_len)

        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            # Merged piped model uses this setup
            self.add_cond_after_transformer = False
            self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_x_emb_x_out = True

        if not only_encode:
            self.x_out = nn.Linear(width, bins, bias=False)
            if self.share_x_emb_x_out:
                self.x_out.weight = self.x_emb.weight
            self.loss = torch.nn.CrossEntropyLoss()

    def preprocess(self, x):
        # Input: x is NHWC and uint8. Converted to NL and long
        # Can include stuff like bitpacking, reordering here.
        N = x.shape[0]
        return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        # Convert back from NL and long to NHWC
        N = x.shape[0]
        assert (0 <= x).all() and (x < self.bins).all()
        if sample_tokens is None or sample_tokens==self.input_dims:
            return x.view(N, *self.input_shape)
        else:
            return x.view(N, -1)

    def forward(self, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False,
                encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        # Preprocess.
        with torch.no_grad():
            x = self.preprocess(x)

        N, D = x.shape
        assert isinstance(x, torch.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()

        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"{x_cond.shape} != {(N, D, self.width)} nor {(N, 1, self.width)}. Did you pass the correct --sample_length?"
        else:
            assert x_cond is None
            x_cond = torch.zeros((N, 1, self.width), device=x.device, dtype=torch.float)

        x_t = x # Target
        x = self.x_emb(x) # X emb
        x = roll(x, 1) # Shift by 1, and fill in start token
        if self.y_cond:
            x[:,0] = y_cond.view(N, self.width)
        else:
            x[:,0] = self.start_token

        x = self.x_emb_dropout(x) + self.pos_emb_dropout(self.pos_emb()) + x_cond # Pos emb and dropout

        x = self.transformer(x, encoder_kv=encoder_kv, fp16=fp16) # Transformer
        if self.add_cond_after_transformer: # Piped doesnt add x_cond
            x = x + x_cond

        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x) # Predictions

        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, :self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len:].reshape(-1, self.bins)

            prime_loss = F.cross_entropy(x_prime, x_t[:, :self.prime_len].reshape(-1)) / np.log(2.)
            gen_loss = F.cross_entropy(x_gen, x_t[:, self.prime_len:].reshape(-1)) / np.log(2.)

            loss = (prime_loss, gen_loss) # Note order! Prime is first
        else:
            loss = F.cross_entropy(x.view(-1, self.bins), x_t.view(-1)) / np.log(2.)  # Loss

        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, D = n_samples, self.input_dims
        if sample_t == 0:
            # Fill in start token
            # x = torch.empty(n_samples, 1, self.width).cuda()
            x = torch.empty(n_samples, 1, self.width).cpu()
            
            if self.y_cond:
                x[:, 0] = y_cond.view(N, self.width)
            else:
                x[:, 0] = self.start_token
        else:
            # assert isinstance(x, torch.cuda.LongTensor)
            assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape == (N, D, self.width):
            cond = x_cond[:, sample_t:sample_t + 1, :]
        else:
            cond = x_cond
        x = x + self.pos_emb()[sample_t:sample_t + 1] + cond  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def sample(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               get_preds=False, sample_tokens=None):
        assert self.training == False

        if sample_tokens is None: sample_tokens=self.input_dims
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = torch.zeros((N, 1, self.width), dtype=torch.float).cuda()

        with torch.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            # for sample_t in get_range(range(0, sample_tokens)):
            for sample_t in range(0, sample_tokens):
            
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16) # Transformer
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x.clone())
                # Adjust logits
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = torch.distributions.Categorical(logits=x).sample() # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x
            self.transformer.del_cache()

            x = torch.cat(xs, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def primed_sample(self, n_samples, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0,
                      top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None):
        assert self.training == False

        if sample_tokens is None: sample_tokens=self.input_dims
        # Preprocess.
        with torch.no_grad():
            x = self.preprocess(x)
        # assert isinstance(x, torch.cuda.LongTensor)
        assert (0 <= x).all() and (x < self.bins).all()
        assert x.shape[0] == n_samples
        xs = torch.split(x, 1, dim=1)
        xs = list(xs)
        assert len(xs) < sample_tokens

        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = torch.zeros((N, 1, self.width), dtype=torch.float).cuda()

        with torch.no_grad():
            if get_preds:
                preds = []

            # Fill up key/value cache for past context by runing forward pass.
            # We do so in chunks instead of doing the whole past in one forward pass to reduce max memory usage.
            if chunk_size is None:
                chunk_size = len(xs)
            #assert len(xs) % chunk_size == 0, f'expected {len(xs)} to be divisible by {chunk_size}'
            chunk_sizes = split_chunks(len(xs), chunk_size)
            x_primes = []
            start = 0
            x = None
            # for current_chunk_size in get_range(chunk_sizes):
            for current_chunk_size in chunk_sizes:
            
                xs_prime, conds_prime = [], []
                for sample_t in range(start, start + current_chunk_size):
                    x_prime, cond_prime = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                    x = xs[sample_t]
                    xs_prime.append(x_prime)
                    conds_prime.append(cond_prime)
                start = start + current_chunk_size

                x_prime, cond_prime = torch.cat(xs_prime, dim=1), torch.cat(conds_prime, dim=1)
                assert x_prime.shape == (n_samples, current_chunk_size, self.width)
                assert cond_prime.shape == (n_samples, current_chunk_size, self.width)
                del xs_prime
                del conds_prime
                if not get_preds:
                    del cond_prime
                x_prime = self.transformer(x_prime, encoder_kv=encoder_kv, sample=True, fp16=fp16)

                if get_preds:
                    if self.add_cond_after_transformer:
                        x_prime = x_prime + cond_prime
                    assert x_prime.shape == (n_samples, current_chunk_size, self.width)
                    del cond_prime
                    x_primes.append(x_prime)
                else:
                    del x_prime

            if get_preds:
                x_prime = torch.cat(x_primes, dim=1)
                assert x_prime.shape == (n_samples, len(xs), self.width)
                x_prime = self.x_out(x_prime)  # Predictions
                preds.append(x_prime)

            empty_cache()
            self.transformer.check_cache(n_samples, len(xs), fp16)

            x = xs[-1]
            assert x.shape == (n_samples, 1)
            empty_cache()
            # for sample_t in get_range(range(len(xs), sample_tokens)):
            for sample_t in range(len(xs), sample_tokens):
                
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16) # Transformer
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x)
                # Adjust logits
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = torch.distributions.Categorical(logits=x).sample() # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x
            self.transformer.del_cache()

            x = torch.cat(xs, dim=1)
            if get_preds:
                preds = torch.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def check_sample(self, chunk_size):
        bs, l, d = (4, self.input_dims, self.width)
        prime = int(self.input_dims//8*7)
        enc_l = self.encoder_dims
        with torch.no_grad():
            y_cond = torch.randn(bs, 1, d).cuda() if self.y_cond else None
            x_cond = torch.randn(bs, l, d).cuda() if self.x_cond else None
            encoder_kv = torch.randn(bs, enc_l, d).cuda()

            x, preds_sample = self.sample(bs, x_cond, y_cond, encoder_kv, get_preds=True)
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = torch.max(torch.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            x_prime = x.view(bs, -1)[:,:prime]
            # unchunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = torch.max(torch.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            # chunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True, chunk_size=chunk_size)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = torch.max(torch.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if torch.max(torch.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"


####################################################################
####################################################################
####################################################################
#                               VQ-VAE functions and class
####################################################################
####################################################################
####################################################################


class ResConvBlock(nn.Module):
    def __init__(self, n_in, n_state):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0),
        )

    def forward(self, x):
        return x + self.model(x)


class Resnet(nn.Module):
    def __init__(self, n_in, n_depth, m_conv=1.0):
        super().__init__()
        self.model = nn.Sequential(*[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)])

    def forward(self, x):
        return self.model(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, zero_out=False, res_scale=1.0):
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.model(x)


# def checkpoint(func, inputs, params, flag):
#     """
# Wrapper to checkpoint results of a forward path during training. # This should only be used to reproduce results as #
# it messes # with RNG. see https://pytorch.org/docs/stable/checkpoint.html for more details

# Args: # func (`_type_`): # _description_ # inputs (`_type_`): # _description_ # params (`_type_`): # _description_ #
# flag (`_type_`): # _description_ #"""
#     if flag:
#         args = inputs + tuple(params)
#         return CheckpointFunction.apply(func, len(inputs), *args)
#     else:
#         return func(*inputs)

# # TODO: I have no Idea what that function does
# class CheckpointFunction (torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, run_function, length, *args):
#         ctx.run_function = run_function
#         ctx.input_tensors = list(args[:length])
#         ctx.input_params = list(args[length:])
#         with torch.no_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         return output_tensors

#     @staticmethod
#     def backward(ctx, *output_grads):
#         for i in range(len(ctx.input_tensors)):
#             temp = ctx.input_tensors[i]
#             ctx.input_tensors[i] = temp.detach()
#             ctx.input_tensors[i].requires_grad = temp.requires_grad
#         with torch.enable_grad():
#             output_tensors = ctx.run_function(*ctx.input_tensors)
#         input_grads = torch.autograd.grad(output_tensors, ctx.input_tensors + ctx.input_params, output_grads, allow_unused=True)
#         del ctx.input_tensors
#         del output_tensors
#         return (None, None) + input_grads
class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        m_conv=1.0,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()

        def _get_depth(depth):
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        # if self.checkpoint_res == 1:
        #     # if dist.get_rank() == 0:
        #     #     print("Checkpointing convs")
        #     self.blocks = nn.ModuleList(blocks)
        # else:
        #    self.model = nn.Sequential(*blocks)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        # if self.checkpoint_res == 1:
        #     for block in self.blocks:
        #         x = checkpoint(block, (x, ), block.parameters(), True)
        #     return x
        # else:
        #     return self.model(x)
        return self.model(x)


class EncoderConvBlock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                    Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out, res_scale),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        down_t,
        stride_t,
        width,
        depth,
        m_conv,
        dilation_growth_rate=1,
        dilation_cycle=None,
        zero_out=False,
        res_scale=False,
        reverse_decoder_dilation=False,
        checkpoint_res=False,
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation,
                        checkpoint_res=checkpoint_res,
                    ),
                    nn.ConvTranspose1d(
                        width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t
                    ),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if "reverse_decoder_dilation" in block_kwargs_copy:
            del block_kwargs_copy["reverse_decoder_dilation"]
        level_block = lambda level, down_t, stride_t: EncoderConvBlock(
            input_emb_width if level == 0 else output_emb_width,
            output_emb_width,
            down_t,
            stride_t,
            **block_kwargs_copy,
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        xs = []

        # 64, 32, ...
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t**down_t)
            # assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        # assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t**down_t)
            # assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x


def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride**down for stride, down in zip(strides, downs)]


def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == "l1":
        return torch.mean(torch.abs(x_pred - x_target)) / hps.bandwidth["l1"]
    elif loss_fn == "l2":
        return torch.mean((x_pred - x_target) ** 2) / hps.bandwidth["l2"]
    elif loss_fn == "linf":
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = torch.topk(residual, hps.linf_k, dim=1)
        return torch.mean(values) / hps.bandwidth["l2"]
    elif loss_fn == "lmix":
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn("l1", x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn("l2", x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn("linf", x_target, x_pred, hps)
        return loss
    else:
        assert False, f"Unknown loss_fn {loss_fn}"


class BottleneckBlock(nn.Module):
    def __init__(self, k_bins, emb_width, mu):
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu
        self.reset_k()
        self.threshold = 1.0

    def reset_k(self):
        self.init = False
        self.k_sum = None
        self.k_elem = None
        # self.register_buffer('k',  torch.zeros(self.k_bins, self.emb_width).cuda())
        self.register_buffer("k", torch.zeros(self.k_bins, self.emb_width))

    def _tile(self, x):
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_k(self, x):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        # init k_w using random vectors from x
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][:k_bins]
        # dist.broadcast(_k_rand, 0)
        self.k = _k_rand
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k
        self.k_elem = torch.ones(k_bins, device=self.k.device)

    def restore_k(self, num_tokens=None, threshold=1.0):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        self.init = True
        assert self.k.shape == (k_bins, emb_width)
        self.k_sum = self.k.clone()
        self.k_elem = torch.ones(k_bins, device=self.k.device)
        if num_tokens is not None:
            expected_usage = num_tokens / k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)
        self.threshold = threshold

    def update_k(self, x, x_l):
        mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
        with torch.no_grad():
            # Calculate new centres
            x_l_onehot = torch.zeros(k_bins, x.shape[0], device=x.device)  # k_bins, N * L
            x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

            _k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
            _k_elem = x_l_onehot.sum(dim=-1)  # k_bins
            y = self._tile(x)
            _k_rand = y[torch.randperm(y.shape[0])][:k_bins]

            # dist.broadcast(_k_rand, 0)
            # dist.all_reduce(_k_sum)
            # dist.all_reduce(_k_elem)

            # Update centres
            old_k = self.k
            self.k_sum = mu * self.k_sum + (1.0 - mu) * _k_sum  # w, k_bins
            self.k_elem = mu * self.k_elem + (1.0 - mu) * _k_elem  # k_bins
            usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
            self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) + (1 - usage) * _k_rand
            _k_prob = _k_elem / torch.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
            entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))  # entropy ie how diverse
            used_curr = (_k_elem >= self.threshold).sum()
            usage = torch.sum(usage)
            dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        if x.shape[-1] == self.emb_width:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., : self.emb_width], x[..., self.emb_width :]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # Normalise
            x = x1 + x2
        else:
            assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
        return x, prenorm

    def postprocess(self, x_l, x_d, x_shape):
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x):
        # Calculate latent code x_l
        k_w = self.k.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)
        min_distance, x_l = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l):
        x = F.embedding(x_l, self.k)
        return x

    def encode(self, x):
        N, width, T = x.shape

        # Preprocess.
        x, prenorm = self.preprocess(x)

        # Quantise
        x_l, fit = self.quantise(x)

        # Postprocess.
        x_l = x_l.view(N, T)
        return x_l

    def decode(self, x_l):
        N, T = x_l.shape
        width = self.emb_width

        # Dequantise
        x_d = self.dequantise(x_l)

        # Postprocess
        x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
        return x_d

    def forward(self, x, update_k=True):
        N, width, T = x.shape

        # Preprocess
        x, prenorm = self.preprocess(x)

        # Init k if not inited
        if update_k and not self.init:
            self.init_k(x)

        # Quantise and dequantise through bottleneck
        x_l, fit = self.quantise(x)
        x_d = self.dequantise(x_l)

        # Update embeddings
        if update_k:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Loss
        commit_loss = torch.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))
        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):
    def __init__(self, l_bins, emb_width, mu, levels):
        super().__init__()
        self.levels = levels
        level_block = lambda level: BottleneckBlock(l_bins, emb_width, mu)
        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(level_block(level))

    def encode(self, xs):
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(self, zs, start_level=0, end_level=None):
        if end_level is None:
            end_level = self.levels
        xs_quantised = [
            level_block.decode(z) for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)
        ]
        return xs_quantised

    def forward(self, xs):
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics


def stft(sig, hps):
    return torch.stft(
        sig,
        hps.n_fft,
        hps.hop_length,
        win_length=hps.window_size,
        window=torch.hann_window(hps.window_size, device=sig.device),
    )


def spec(x, hps):
    return torch.norm(stft(x, hps), p=2, dim=-1)


class DefaultSTFTValues:
    def __init__(self, hps):
        self.sr = hps.sr
        self.n_fft = 2048
        self.hop_length = 256
        self.window_size = 6 * self.hop_length


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = torch.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f"Unknown input shape {x.shape}")
    return x


def spectral_loss(x_in, x_out, hps):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)
    return norm(spec_in - spec_out)


def spectral_convergence(x_in, x_out, hps, epsilon=2e-3):
    hps = DefaultSTFTValues(hps)
    spec_in = spec(squeeze(x_in.float()), hps)
    spec_out = spec(squeeze(x_out.float()), hps)

    gt_norm = norm(spec_in)
    residual_norm = norm(spec_in - spec_out)
    mask = (gt_norm > epsilon).float()
    return (residual_norm * mask) / torch.clamp(gt_norm, min=epsilon)


class STFTValues:
    def __init__(self, hps, n_fft, hop_length, window_size):
        self.sr = hps.sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size


def multispectral_loss(x_in, x_out, hps):
    losses = []
    assert len(hps.multispec_loss_n_fft) == len(hps.multispec_loss_hop_length) == len(hps.multispec_loss_window_size)
    args = [hps.multispec_loss_n_fft, hps.multispec_loss_hop_length, hps.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        hps = STFTValues(hps, n_fft, hop_length, window_size)
        spec_in = spec(squeeze(x_in.float()), hps)
        spec_out = spec(squeeze(x_out.float()), hps)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: sum(vals) / len(vals) for key, vals in metrics.items()}


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.sample_length:
            downsamples = calculate_strides(config.vq_vae_strides, config.vq_vae_downsampling)
            top_raw_to_tokens = np.prod(downsamples)
            config.sample_length = (
                config.sample_length_in_seconds * config.sampling_rate // top_raw_to_tokens
            ) * top_raw_to_tokens

        input_shape = (config.sample_length, 1)
        block_kwargs = dict(
            width=config.vq_vae_conv_block_width,
            depth=config.vq_vae_conv_block_depth,
            m_conv=config.vq_vae_m_conv,
            dilation_growth_rate=config.vq_vae_dilation_growth_rate,
            dilation_cycle=config.vq_vae_dilation_cycle,
            reverse_decoder_dilation=config.vq_vae_reverse_decoder_dilation,
        )

        multipliers = config.vq_vae_multipliers
        emb_width = config.vq_vae_emmbedding_width
        self.width = config.vq_vae_width
        self.depth = config.vq_vae_depth

        self.downs_t = downs_t = config.vq_vae_downsampling
        self.strides_t = strides_t = config.vq_vae_strides
        self.l_bins = l_bins = config.vq_vae_codebook_dimension
        self.commit = config.vq_vae_commit
        self.spectral = config.vq_vae_strides
        self.multispectral = config.vq_vae_strides

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.levels = levels = config.vq_vae_levels
        self.z_shapes = [(x_shape[0] // self.hop_lengths[level],) for level in range(levels)]

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        encoder = lambda level: Encoder(
            x_channels, emb_width, level + 1, downs_t[: level + 1], strides_t[: level + 1], **_block_kwargs(level)
        )
        decoder = lambda level: Decoder(
            x_channels, emb_width, level + 1, downs_t[: level + 1], strides_t[: level + 1], **_block_kwargs(level)
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        self.bottleneck = Bottleneck(l_bins, emb_width, config.vq_vae_lmu, levels)

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [torch.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return torch.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
        x_chunks = torch.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x_chunks:
            zs_i = self._encode(x_i, start_level=start_level, end_level=end_level)
            zs_list.append(zs_i)
        zs = [torch.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [torch.randint(0, self.l_bins, size=(n_samples, *z_shape), device="cuda") for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, hps, loss_fn="l1"):
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])

        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(xs)
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level : level + 1], all_levels=False)
            # assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)

        # Loss
        def _spectral_loss(x_target, x_out, hps):
            if hps.use_nonrelative_specloss:
                sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth["spec"]
            else:
                sl = spectral_convergence(x_target, x_out, hps)
            sl = torch.mean(sl)
            return sl

        def _multispectral_loss(x_target, x_out, hps):
            sl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth["spec"]
            sl = torch.mean(sl)
            return sl

        recons_loss = torch.zeros(()).to(x.device)
        spec_loss = torch.zeros(()).to(x.device)
        multispec_loss = torch.zeros(()).to(x.device)
        # x_target = audio_postprocess(x.float(), hps)
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            # x_out = audio_postprocess(x_out, hps)
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, hps)
            this_spec_loss = _spectral_loss(x_target, x_out, hps)
            this_multispec_loss = _multispectral_loss(x_target, x_out, hps)
            metrics[f"recons_loss_l{level + 1}"] = this_recons_loss
            metrics[f"spectral_loss_l{level + 1}"] = this_spec_loss
            metrics[f"multispectral_loss_l{level + 1}"] = this_multispec_loss
            recons_loss += this_recons_loss
            spec_loss += this_spec_loss
            multispec_loss += this_multispec_loss

        commit_loss = sum(commit_losses)
        loss = (
            recons_loss + self.spectral * spec_loss + self.multispectral * multispec_loss + self.commit * commit_loss
        )

        with torch.no_grad():
            sc = torch.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out, hps)
            linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(
            dict(
                recons_loss=recons_loss,
                spectral_loss=spec_loss,
                multispectral_loss=multispec_loss,
                spectral_convergence=sc,
                l2_loss=l2_loss,
                l1_loss=l1_loss,
                linf_loss=linf_loss,
                commit_loss=commit_loss,
                **quantiser_metrics,
            )
        )

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics


############################### 
# Conditioners


class Conditioner(nn.Module):
    def __init__(self, input_shape, bins, down_t, stride_t, out_width, init_scale, zero_out, res_scale, **block_kwargs):
        super().__init__()
        self.x_shape = input_shape

        # Embedding
        self.width = out_width
        self.x_emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)

        # Conditioner
        self.cond = DecoderConvBock(self.width, self.width, down_t, stride_t, **block_kwargs, zero_out=zero_out, res_scale=res_scale)
        self.ln = LayerNorm(self.width)

    def preprocess(self, x):
        x = x.permute(0,2,1) # NTC -> NCT
        return x

    def postprocess(self, x):
        x = x.permute(0,2,1) # NCT -> NTC
        return x

    def forward(self, x, x_cond=None):
        N = x.shape[0]
        # assert_shape(x, (N, *self.x_shape))
        if x_cond is not None:
            #assert_shape(x_cond, (N, *self.x_shape, self.width))
            pass
        else:
            x_cond = 0.0
        # Embed x
        x = x.long()
        x = self.x_emb(x)
        # assert_shape(x, (N, *self.x_shape, self.width))
        x = x + x_cond

        # Run conditioner
        x = self.preprocess(x)
        x = self.cond(x)
        x = self.postprocess(x)
        x = self.ln(x)
        return x

def flip(x):
    def _flip(x):
        return x.permute(0,2,1).contiguous()
    if isinstance(x, (list, tuple)):
        return [flip(z) for z in x]
    return _flip(x)

class SimpleEmbedding(nn.Module):
    def __init__(self, bins, out_width, init_scale):
        super().__init__()
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)

    def forward(self, y):
        assert len(y.shape) == 2, f"Expected shape with 2 dims, got {y.shape}"
        # assert isinstance(y, t.cuda.LongTensor), f"Expected dtype {t.cuda.LongTensor}, got {y.dtype}        assert (0 <= y).all() and (y < self.bins).all(), f"Bins {self.bins}, got label {y}"
        return self.emb(y)

class RangeEmbedding(nn.Module):
    # Interpolating
    # Interpolate so that [pos_start, pos_end] <-> position tensor of length n_ctx
    #
    # Binning
    # For each pos in position tensor, find its bin
    # [start,end) mapped to [0,1,...,bins-1]
    # [start,end) -> [0,1) -> [0, bins) -> floor -> [0,...,bins-1]
    # NOTE: Open ended interval on right, so start <= pos < end, not <= end
    def __init__(self, n_time, bins, range, out_width, init_scale, clamp=False):
        super().__init__()
        self.n_time = n_time
        self.bins = bins
        self.emb = nn.Embedding(bins, out_width)
        nn.init.normal_(self.emb.weight, std=0.01 * init_scale)
        self.pos_min, self.pos_max = range
        self.clamp = clamp

    def forward(self, pos_start, pos_end=None):
        # Check if [pos_start,pos_end] in [pos_min, pos_max)
        assert len(pos_start.shape) == 2, f"Expected shape with 2 dims, got {pos_start.shape}"
        assert (self.pos_min <= pos_start).all() and (pos_start < self.pos_max).all(), f"Range is [{self.pos_min},{self.pos_max}), got {pos_start}"
        pos_start = pos_start.float()
        if pos_end is not None:
            assert len(pos_end.shape) == 2, f"Expected shape with 2 dims, got {pos_end.shape}"
            if self.clamp:
                pos_end = pos_end.clamp(self.pos_min, self.pos_max)
            assert (self.pos_min <= pos_end).all() and (pos_end <= self.pos_max).all(), f"Range is [{self.pos_min},{self.pos_max}), got {pos_end}"
            pos_end = pos_end.float()
        # Interpolate so that [pos_start, ..., pos_end] <-> position tensor of length n_ctx
        n_time = self.n_time
        if n_time != 1:
            assert pos_end is not None
            interpolation  = (torch.arange(0, n_time, dtype=torch.float, device='cpu').view(1,n_time)/n_time)
            position = pos_start + (pos_end - pos_start)*interpolation
        else:
            position = pos_start

        # Bin each value to bins
        normalised_position = (position - self.pos_min) / (self.pos_max - self.pos_min) # [0,1)
        bins = (self.bins * normalised_position).floor().long().detach() # [0,1) -> [0,1..,bins) -> [0,1...,bins-1]
        return self.emb(bins)

class LabelConditioner(nn.Module):
    def __init__(self, y_bins, t_bins, sr, min_duration, max_duration, n_time, out_width, init_scale, max_bow_genre_size, include_time_signal):
        super().__init__()
        self.n_time = n_time
        self.out_width = out_width
        assert len(y_bins) == 2, f"Expecting (genre, artist) bins, got {y_bins}"
        bow_genre_bins, artist_bins = y_bins
        self.max_bow_genre_size = max_bow_genre_size
        self.bow_genre_emb = SimpleEmbedding(bow_genre_bins, out_width, init_scale)
        self.artist_emb = SimpleEmbedding(artist_bins, out_width, init_scale)
        self.include_time_signal = include_time_signal
        if self.include_time_signal:
            t_ranges = ((min_duration * sr, max_duration * sr),  # Total length
                        (0.0, max_duration * sr),                # Absolute pos
                        (0.0, 1.0))                              # Relative pos
            assert len(t_ranges) == 3, f"Expecting (total, absolute, relative) ranges, got {t_ranges}"
            total_length_range, absolute_pos_range, relative_pos_range = t_ranges
            self.total_length_emb = RangeEmbedding(1, t_bins, total_length_range, out_width, init_scale)
            self.absolute_pos_emb = RangeEmbedding(n_time, t_bins, absolute_pos_range, out_width, init_scale)
            self.relative_pos_emb = RangeEmbedding(n_time, t_bins, relative_pos_range, out_width, init_scale, clamp=True)

    def forward(self, y):
        assert len(y.shape) == 2, f"Expected shape with 2 dims, got {y.shape}"
        assert y.shape[-1] == 4 + self.max_bow_genre_size, f"Expected shape (N,{4 + self.max_bow_genre_size}), got {y.shape}"
        # assert isinstance(y, t.cuda.LongTensor), f"Expected dtype {t.cuda.LongTensor}, got {y.dtype}"
        N = y.shape[0]
        total_length, offset, length, artist, genre = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4], y[:,4:]

        # Start embedding of length 1
        artist_emb = self.artist_emb(artist)
        # Empty genre slots are denoted by -1. We mask these out.
        mask = (genre >= 0).float().unsqueeze(2)
        genre_emb = (self.bow_genre_emb(genre.clamp(0)) * mask).sum(dim=1, keepdim=True)
        start_emb = genre_emb + artist_emb
        # assert_shape(start_emb, (N, 1, self.out_width))

        # Pos embedding of length n_ctx
        if self.include_time_signal:
            start, end = offset, offset + length
            total_length, start, end = total_length.float(), start.float(), end.float()
            pos_emb = self.total_length_emb(total_length) + self.absolute_pos_emb(start, end) + self.relative_pos_emb(start/total_length, end/total_length)
            # assert_shape(pos_emb, (N, self.n_time, self.out_width))
        else:
            pos_emb = None
        return start_emb, pos_emb
    
####################################################################
####################################################################
# def load_tf_weights_in_jukebox(model, config, jukebox_checkpoint_path):
#     """Load tf checkpoints in a pytorch model"""
#     try:
#         import re

#         import tensorflow as tf
#     except ImportError:
#         logger.error(
#             "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
#             "https://www.tensorflow.org/install/ for installation instructions."
#         )
#         raise
#     tf_path = os.path.abspath(jukebox_checkpoint_path)
#     logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
#     # Load weights from TF model
#     init_vars = tf.train.list_variables(tf_path)
#     names = []
#     arrays = []
#     for name, shape in init_vars:
#         logger.info(f"Loading TF weight {name} with shape {shape}")
#         array = tf.train.load_variable(tf_path, name)
#         names.append(name)
#         arrays.append(array.squeeze())

#     for name, array in zip(names, arrays):
#         name = name[6:]  # skip "model/"
#         name = name.split("/")
#         pointer = model
#         for m_name in name:
#             if re.fullmatch(r"[A-Za-z]+\d+", m_name):
#                 scope_names = re.split(r"(\d+)", m_name)
#             else:
#                 scope_names = [m_name]
#             if scope_names[0] == "w" or scope_names[0] == "g":
#                 pointer = getattr(pointer, "weight")
#             elif scope_names[0] == "b":
#                 pointer = getattr(pointer, "bias")
#             elif scope_names[0] == "wpe" or scope_names[0] == "wte":
#                 pointer = getattr(pointer, scope_names[0])
#                 pointer = getattr(pointer, "weight")
#             else:
#                 pointer = getattr(pointer, scope_names[0])
#             if len(scope_names) >= 2:
#                 num = int(scope_names[1])
#                 pointer = pointer[num]
#         try:
#             assert (
#                 pointer.shape == array.shape
#             ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
#         except AssertionError as e:
#             e.args += (pointer.shape, array.shape)
#             raise
#         logger.info(f"Initialize PyTorch weight {name}")
#         pointer.data = torch.from_numpy(array)
#     return model


class JukeboxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = JukeboxConfig
    # load_tf_weights = load_tf_weights_in_jukebox
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the Jukebox Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, JukeboxModel):
            module.gradient_checkpointing = value


@dataclass
class JukeboxDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            JukeboxAttentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


JUKEBOX_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`JukeboxConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

JUKEBOX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`JukeboxTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
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
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the jukebox models have the
            following number of attention modules:

                - jukebox: 12
                - jukebox-medium: 24
                - jukebox-large: 36
                - jukebox-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using jukebox-xl, which has a total of 48 attention modules:
    model = JukeboxLMHeadModel.from_pretrained("jukebox-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with jukebox-large:
    model = JukeboxLMHeadModel.from_pretrained("jukebox-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare JUKEBOX Model transformer outputting raw hidden-states without any specific head on top.",
    JUKEBOX_START_DOCSTRING,
)
class JukeboxModel(JukeboxPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([JukeboxBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(JUKEBOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # JukeboxAttention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
