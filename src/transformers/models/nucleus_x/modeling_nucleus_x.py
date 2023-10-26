# This code was adapted from Sehyun Choi's https://github.com/syncdoth/retnet
# licensed under the MIT License. Above repository builds on Microsoft's
# torchscale library and the RetNet implementations in this library, also
# licensed under the MIT License. It has been modified from its original forms
# to accommodate minor architectural differences compared to NucleusX used by
# the NucleusAI team that trained the model.

# MIT License
#
# Copyright (c) 2023  NucleusAI and The HuggingFace Inc. team and Sehyun Choi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" PyTorch NucleusX model."""


import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import drop_path
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import get_activation
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)


try:
    from apex.normalization import FusedRMSNorm

    SUPPORT_APEX = True
except ImportError:
    SUPPORT_APEX = False

from .configuration_nucleus_x import NucleusXConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "NucleusXConfig"

NUCLEUS_X_PRETRAINED_MODEL_ARCHIVE_LIST = []


def split_heads(tensors, bsz, seqlen, num_heads):
    assert isinstance(tensors, (tuple, list))
    return [x.view(bsz, seqlen, num_heads, -1).transpose(1, 2) for x in tensors]


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def build_rms_norm(dim, eps=1e-6, elementwise_affine=True):
    if SUPPORT_APEX:
        return FusedRMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
    return NculeusXRMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)


class NculeusXRMSNorm(nn.Module):
    """RMSNorm from torchscale implementation.
    Copied from https://github.com/microsoft/torchscale/blob/main/torchscale/component/rms_norm.py
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class NucleusXRelPos(nn.Module):
    """Relative Position Embedding for NucleusX. Includes xpos and retention decay.
    Adapted from https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py
    """

    def __init__(self, config: NucleusXConfig):
        super().__init__()
        self.config = config
        num_heads = config.decoder_retention_heads

        angle = 1.0 / (10000 ** torch.linspace(0, 1, config.decoder_embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # decay (gamma)
        if config.use_lm_decay:
            # NOTE: alternative way described in the paper
            s = torch.log(torch.tensor(1 / 32))
            e = torch.log(torch.tensor(1 / 512))
            decay = torch.log(1 - torch.exp(torch.linspace(s, e, num_heads)))  # [h,]
        else:
            decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = config.recurrent_chunk_size

    def forward(
        self, slen, forward_mode="parallel", recurrent_chunk_size=None, retention_mask=None, get_decay_scale=True
    ):
        if forward_mode == "recurrent":
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.view(1, -1, 1, 1).exp())
        elif forward_mode == "chunkwise":
            if recurrent_chunk_size is None:
                recurrent_chunk_size = self.recurrent_chunk_size
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(recurrent_chunk_size, recurrent_chunk_size)).to(self.decay)
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            # TODO: need to handle retention_mask
            # scaling
            value_inner_decay = mask[:, :, -1] / mask[:, :, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[None, :, None, None]
            query_inner_decay = query_inner_decay[None, :, :, None] / (
                scale / mask[:, :, -1].sum(dim=-1)[:, :, None, None]
            )
            # decay_scale (used for kv cache)
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask)
            else:
                decay_scale = None
            retention_rel_pos = (
                (sin, cos),
                (inner_mask, cross_decay, query_inner_decay, value_inner_decay, decay_scale),
            )
        else:  # parallel
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen)).to(self.decay)
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask.unsqueeze(0)  # [1, h, t, t]
            if retention_mask is not None:
                # this is required for left padding
                mask = mask * retention_mask.float().view(-1, 1, 1, slen).to(mask)

            # scaling
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            mask = torch.nan_to_num(mask, nan=0.0)
            # decay_scale (used for kv cache)
            if get_decay_scale:
                decay_scale = self.compute_decay_scale(slen, retention_mask)
            else:
                decay_scale = None
            # mask processing for intra decay
            if retention_mask is not None:
                max_non_zero = torch.cumsum(retention_mask, dim=-1).max(dim=-1).indices  # [b,]
                intra_decay = mask[range(mask.shape[0]), :, max_non_zero]
            else:
                intra_decay = mask[:, :, -1]

            retention_rel_pos = ((sin, cos), (mask, intra_decay, decay_scale))

        return retention_rel_pos

    def compute_decay_scale(self, slen, retention_mask=None):
        exponent = torch.arange(slen, device=self.decay.device).float()
        decay_scale = self.decay.exp().view(-1, 1) ** exponent.view(1, -1)  # [h, t]
        if retention_mask is not None:
            seqlen = retention_mask.sum(dim=-1)  # [b,]
            bsz = seqlen.size(0)
            decay_scale = decay_scale.unsqueeze(0).repeat(bsz, 1, 1)  # [b, h, t]
            for i, pos in enumerate(seqlen):
                # the formula for decay_scale is `sum(gamma^i) for i in [0, slen).`
                # Since the retention_mask is 0 for padding, we can set the decay_scale
                # to 0 for the padding positions.
                decay_scale[i, :, pos.item() :] = 0
        else:
            bsz = 1
        decay_scale = decay_scale.sum(-1).view(bsz, -1, 1, 1)  # [b, h, 1, 1]
        return decay_scale


class NucleusXMultiScaleRetention(nn.Module):
    """Multi-Scale Retention for NucleusX.
    Adapted from https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py
    """

    def __init__(
        self,
        config: NucleusXConfig,
        gate_fn="swish",
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.value_dim = config.decoder_value_embed_dim
        self.num_heads = config.decoder_retention_heads
        self.head_dim = self.value_dim // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim**-0.5

        self.gate_fn = get_activation(gate_fn)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.value_dim, bias=False)
        self.g_proj = nn.Linear(self.embed_dim, self.value_dim, bias=False)

        self.out_proj = nn.Linear(self.value_dim, self.embed_dim, bias=False)

        self.group_norm = build_rms_norm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)

    def reset_parameters(self, gain=2**-2.5):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def parallel_retention(self, q, k, v, decay_mask, use_cache=False):
        """
        q, # bsz * num_head * len * qk_dim k, # bsz * num_head * len * qk_dim v, # bsz * num_head * len * v_dim
        decay_mask, # (1 or bsz) * num_head * len * len
        """
        decay_mask, intra_decay, scale = decay_mask

        # [b, h, t, t]
        retention = q @ k.transpose(-1, -2)  # (scaled dot-product)
        retention = retention * decay_mask

        # invariant after normalization
        retention = retention / retention.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)

        output = retention @ v  # [b, h, t, v_dim / h]
        output = output.transpose(1, 2)  # [b, t, h, v_dim / h]

        if not use_cache:  # skip cache
            return output, None, retention

        # kv cache: [b, h, t, v_dim, qk_dim]
        current_kv = k.unsqueeze(-2) * v.unsqueeze(-1)
        intra_decay = intra_decay[:, :, :, None, None]  # [b, h, t, 1, 1]
        current_kv = (current_kv * intra_decay).sum(2)  # [b, h, v_dim, qk_dim]

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache, retention

    def recurrent_retention(self, q, k, v, decay, past_key_value=None, retention_mask=None):
        """
        q, k, v, # bsz * num_head * 1 * qkv_dim past_key_value:
            - "prev_key_value" # bsz * num_head * v_dim * qk_dim
            - "scale" # (1 or bsz) * num_head * 1 * 1
        decay # (1 or bsz) * num_head * 1 * 1 retention_mask # bsz * 1
        """
        if retention_mask is not None:
            retention_mask = retention_mask.float().view(-1, 1, 1, 1).to(decay)
        else:
            retention_mask = torch.ones(k.size(0), 1, 1, 1).to(decay)
        # (b, h, v_dim, qk_dim)
        current_kv = k * v.transpose(-1, -2) * retention_mask

        if past_key_value is not None and "prev_key_value" in past_key_value:
            prev_kv = past_key_value["prev_key_value"]
            prev_scale = past_key_value["scale"]
            scale = torch.where(retention_mask == 0, prev_scale, prev_scale * decay + 1)
            # connect prev_kv and current_kv
            # how much to decay prev_kv
            decay_amount = prev_scale.sqrt() * decay / scale.sqrt()
            decay_amount = torch.where(retention_mask == 0, 1, decay_amount)
            prev_kv = prev_kv * decay_amount  # decay prev_kv
            current_kv = current_kv / scale.sqrt()  # scale current_kv
            current_kv = torch.nan_to_num(current_kv, nan=0.0)  # remove nan, scale might be 0

            current_kv = prev_kv + current_kv
        else:
            scale = torch.ones_like(decay)
            # when retention_mask is 0 at the beginning, setting scale to 1 will
            # make the first retention to use the padding incorrectly. Hence,
            # setting it to 0 here. This is a little ugly, so we might want to
            # change this later. TODO: improve
            scale = torch.where(retention_mask == 0, torch.zeros_like(decay), scale)

        output = torch.sum(q * current_kv, dim=3).unsqueeze(1)  # (b, 1, h, d_v)

        cache = {"prev_key_value": current_kv, "scale": scale}
        return output, cache

    def chunkwise_retention(self, q, k, v, decay_mask):
        """
        q, k, v, # bsz * num_head * seqlen * qkv_dim past_key_value:
            - "prev_key_value" # bsz * num_head * v_dim * qk_dim
            - "scale" # (1 or bsz) * num_head * 1 * 1
        decay_mask, # 1 * num_head * chunk_size * chunk_size cross_decay, # 1 * num_head * 1 * 1 inner_decay, # 1 *
        num_head * chunk_size * 1
        """
        decay_mask, cross_decay, query_inner_decay, value_inner_decay, decay_scale = decay_mask
        bsz, _, tgt_len, _ = v.size()
        chunk_len = decay_mask.size(-1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len

        # [b, n_c, h, t_c, qkv_dim]
        q = q.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        k = k.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, self.num_heads, num_chunks, chunk_len, self.head_dim).transpose(1, 2)

        k_t = k.transpose(-1, -2)

        qk_mat = q @ k_t  # [b, n_c, h, t_c, t_c]
        qk_mat = qk_mat * decay_mask.unsqueeze(1)
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        # [b, n_c, h, t_c, v_dim]
        inner_output = torch.matmul(qk_mat, v)

        # reduce kv in one chunk
        # [b, n_c, h, qk_dim, v_dim]
        kv = k_t @ (v * value_inner_decay)
        # kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (q * query_inner_decay.unsqueeze(1)) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        output = output.transpose(2, 3)  # [b, n_c, t_c, h, v_dim]

        cache = {"prev_key_value": kv_state.transpose(-2, -1), "scale": decay_scale}
        return output, cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        forward_mode: str = "parallel",
        output_retentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        B, T, H = hidden_states.size()
        (sin, cos), decay_mask = rel_pos
        # projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)
        # multi-head
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        k *= self.scaling  # for scaled dot product
        # rotate
        # NOTE: theta_shift has bug with mps device.
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        # retention
        if forward_mode == "parallel":
            retention_out, curr_kv, retention_weights = self.parallel_retention(
                qr, kr, v, decay_mask, use_cache=use_cache
            )
        elif forward_mode == "recurrent":
            retention_out, curr_kv = self.recurrent_retention(
                qr, kr, v, decay_mask, past_key_value=past_key_value, retention_mask=retention_mask
            )
        elif forward_mode == "chunkwise":
            retention_out, curr_kv = self.chunkwise_retention(qr, kr, v, decay_mask)
        else:
            raise ValueError(f"forward_mode {forward_mode} not supported.")

        # concaat heads
        normed = self.group_norm(retention_out).reshape(B, T, self.value_dim)
        # out gate & proj
        out = self.gate_fn(g) * normed
        out = self.out_proj(out)

        outputs = (out, curr_kv)
        if output_retentions:
            outputs += (retention_weights,) if forward_mode == "parallel" else (None,)
        return outputs


class NucleusXGLU(nn.Module):
    """Gated Linear Unit from torchscale implementation.
    Copied from https://github.com/microsoft/torchscale/blob/main/torchscale/component/gate_linear_unit.py
    """

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation(activation_fn)
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.dropout_module = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class NucleusXDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Coped from https://github.com/microsoft/torchscale/blob/main/torchscale/component/droppath.py"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class NucleusXDecoderLayer(nn.Module):
    """Decoder layer for NucleusX.
    Adapted from https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py
    """

    def __init__(self, config: NucleusXConfig, depth: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.decoder_embed_dim
        self.dropout_module = nn.Dropout(config.dropout)

        if config.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, config.drop_path_rate, config.decoder_layers)[depth]
            self.drop_path = NucleusXDropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.retention = NucleusXMultiScaleRetention(config)

        self.normalize_before = config.decoder_normalize_before

        self.retention_layer_norm = build_rms_norm(self.embed_dim, eps=config.rms_norm_eps)

        self.ffn_dim = config.decoder_ffn_embed_dim

        self.ffn = self.build_ffn()

        self.final_layer_norm = build_rms_norm(self.embed_dim, eps=config.rms_norm_eps)

        if config.deepnorm:
            self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self):
        return NucleusXGLU(
            self.embed_dim,
            self.ffn_dim,
            self.config.activation_fn,
            self.config.dropout,
            self.config.activation_dropout,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        hidden_states: torch.Tensor,
        retention_rel_pos: Tuple[Tuple[torch.Tensor]],
        retention_mask: Optional[torch.Tensor] = None,
        forward_mode: str = "parallel",
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_retentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        msr_outs = self.retention(
            hidden_states,
            retention_rel_pos,
            retention_mask=retention_mask,
            past_key_value=past_key_value,
            forward_mode=forward_mode,
            output_retentions=output_retentions,
            use_cache=use_cache,
        )
        hidden_states = msr_outs[0]
        curr_kv = msr_outs[1]

        hidden_states = self.dropout_module(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.retention_layer_norm(hidden_states)

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.ffn(hidden_states)

        if self.drop_path is not None:
            hidden_states = self.drop_path(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, curr_kv)

        if output_retentions:
            outputs += (msr_outs[2],)
        return outputs


class NucleusXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NucleusXConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NucleusXDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        gain = self.config.initializer_factor

        if isinstance(module, NucleusXForCausalLM):
            module.reset_parameters()
        elif isinstance(module, NucleusXMultiScaleRetention):
            module.reset_parameters(gain=gain)
        if isinstance(module, NucleusXGLU):
            module.reset_parameters()
        elif isinstance(module, nn.Embedding):  # copied from LlamaPretrainedModel
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if self.config.deepnorm and isinstance(module, NucleusXDecoderLayer):
            init_scale = math.pow(8.0 * self.config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.div_(init_scale)

        # TODO: in torchscale implementation, when using GLU, the normalization for
        # subln is ignored completely. We might want to do the same.
        if self.config.subln and isinstance(module, NucleusXMultiScaleRetention):
            init_scale = math.sqrt(math.log(self.config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if "out_proj" in name or "v_proj" in name:
                    p.data.div_(init_scale)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NucleusXModel):
            module.gradient_checkpointing = value


@dataclass
class NucleusXOutputWithPast(ModelOutput):
    """
    class for NucleusX model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, decoder_embed_dim)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            decoder_embed_dim)` is output.
        past_key_values (`Tuple(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks) that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_retentions=True` is passed or when `config.output_retentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NucleusXCausalLMOutputWithPast(ModelOutput):
    """
    class for NucleusX causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Tuple(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks) that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NucleusXClassifierOutputWithPast(ModelOutput):
    """
    class for NucleusX sequence classifier model outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Tuple(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks) that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, decoder_embed_dim)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        retentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Retentions weights, used for visualization.

        attentions (`tuple(torch.FloatTensor)`, *optional*, for backward compatibility. Same as retentions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    retentions: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


NUCLEUS_X_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~NucleusXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NUCLEUS_X_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`NucleusXTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        forward_mode (`str`, *optional*, defaults to `"parallel"`):
            The mode for the forward pass. Can be one of `"parallel"`, `"recurrent"`, `"chunkwise"`.
        recurrent_chunk_size (`int`, *optional*, defaults to `None`):
            The chunk size for chunkwise forward mode. Only used when `forward_mode="chunkwise"`.
        retention_rel_pos (`Tuple[torch.Tensor]`, *optional*):
            The relative position encoding for the retention blocks. If not provided, it will be computed on the fly.
            It is advised to pre-compute it during pre-training, since the sequence length is fixed typically.
"""


@add_start_docstrings(
    "The bare NucleusX Model transformer outputting raw hidden-states without any specific head on top."
    " Based on RetNet Architecture (https://arxiv.org/abs/2307.08621),"
    " adapted from https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py",
    NUCLEUS_X_START_DOCSTRING,
)
class NucleusXModel(NucleusXPreTrainedModel):
    def __init__(self, config: NucleusXConfig, embed_tokens: nn.Embedding = None):
        super().__init__(config)
        self.config = config

        self.dropout_module = nn.Dropout(config.dropout)

        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)

        if embed_tokens is None:
            embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim, config.pad_token_id)
        self.embed_tokens = embed_tokens

        if config.rms_norm_embedding:
            self.rms_norm_embedding = build_rms_norm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.rms_norm_embedding = None

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(NucleusXDecoderLayer(config, depth=i))

        self.decoder_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = build_rms_norm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.layer_norm = None

        self.rel_pos = NucleusXRelPos(config)
        self.recurrent_chunk_size = config.recurrent_chunk_size

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_embedding(
        self,
        input_ids=None,
        inputs_embeds=None,
        forward_mode="parallel",
    ):
        # if past_key_values is not None:
        if forward_mode == "recurrent":
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]
            else:
                input_ids = input_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed = self.embed_scale * inputs_embeds

        if self.rms_norm_embedding is not None:
            embed = self.rms_norm_embedding(embed)

        embed = self.dropout_module(embed)

        return embed

    @add_start_docstrings_to_model_forward(NUCLEUS_X_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_mode: Optional[str] = "parallel",
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple, NucleusXOutputWithPast]:
        r"""
        past_key_values (`Tuple(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks) that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # NOTE: internal renaming of attention -> retention
        output_retentions = output_attentions
        retention_mask = attention_mask

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        # embed tokens
        inputs_embeds = self.forward_embedding(
            input_ids=input_ids, inputs_embeds=inputs_embeds, forward_mode=forward_mode
        )

        if retention_mask is not None and forward_mode == "recurrent":
            retention_mask = retention_mask[:, -1:]

        hidden_states = inputs_embeds

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = forward_mode == "chunkwise" and seq_length % recurrent_chunk_size != 0
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        if retention_rel_pos is None:
            retention_rel_pos = self.rel_pos(
                slen,
                forward_mode=forward_mode,
                recurrent_chunk_size=recurrent_chunk_size,
                retention_mask=retention_mask,
                get_decay_scale=use_cache,
            )

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_retentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    retention_rel_pos,
                    retention_mask,
                    forward_mode,
                    past_key_value,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    retention_rel_pos,
                    retention_mask=retention_mask,
                    forward_mode=forward_mode,
                    past_key_value=past_key_value,
                    output_retentions=output_retentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_retentions:
                all_retentions += (layer_outputs[2],)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_retentions] if v is not None)
        return NucleusXOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
            attentions=all_retentions,
        )


@add_start_docstrings(
    """NucleusX Model with a `language modeling` head on top for CLM fine-tuning.""", NUCLEUS_X_START_DOCSTRING
)
class NucleusXForCausalLM(NucleusXPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: NucleusXConfig, embed_tokens: nn.Embedding = None) -> None:
        super().__init__(config)
        self.model = NucleusXModel(config, embed_tokens=embed_tokens)
        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)

        self.post_init()

    def reset_parameters(self):
        nn.init.normal_(self.lm_head.weight, mean=0, std=self.config.decoder_embed_dim**-0.5)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(NUCLEUS_X_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NucleusXCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_mode: Optional[str] = None,
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple, NucleusXCausalLMOutputWithPast]:
        r"""
        past_key_values (`Tuple(Dict(str, torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            - "prev_key_value": shape=(bsz * num_head * v_dim * qk_dim)
            - "scale": shape=((1 or bsz) * num_head * 1 * 1)

            Contains pre-computed hidden-states (key and values in the multi-scale retention blocks) that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        forward_mode (`str`, *optional*, defaults to `"parallel"`):
            The mode for the forward pass. Can be one of `"parallel"`, `"recurrent"`, `"chunkwise"`.
        retention_rel_pos (`Tuple[torch.Tensor]`, *optional*):
            The relative position encoding for the retention blocks. If not provided, it will be computed on the fly.
            It is advised to pre-compute it during pre-training, since the sequence length is fixed typically.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:

        Example:

        ```python
        >>> from transformers import NucleusXTokenizer, NucleusXForCausalLM, NucleusXConfig
        >>> import torch

        >>> tokenizer = NucleusXTokenizer.from_pretrained("nucleus_x")
        >>> config = NucleusXConfig.from_pretrained("nucleus_x")
        >>> config.is_decoder = True
        >>> model = NucleusXForCausalLM.from_pretrained("nucleus_x", config=config)

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        forward_mode = forward_mode if forward_mode is not None else self.config.forward_mode
        recurrent_chunk_size = (
            recurrent_chunk_size if recurrent_chunk_size is not None else self.config.recurrent_chunk_size
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forward_mode=forward_mode,
            use_cache=use_cache,
            recurrent_chunk_size=recurrent_chunk_size,
            retention_rel_pos=retention_rel_pos,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if self.config.z_loss_coeff > 0:
                # z_loss from PaLM paper
                # z_loss = 1e-4 * log(log(z)), where z = sum(exp(logits))
                z_loss = torch.logsumexp(shift_logits, dim=-1).log().mean()
                loss += self.config.z_loss_coeff * z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return NucleusXCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            retentions=outputs.retentions,
            attentions=outputs.attentions,
        )

    def _crop_past_key_values(model, past_key_values, maximum_length):
        """Since NucleusX's kv do not have length, no need to crop. Just return"""
        return past_key_values

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            raise NotImplementedError(
                "Generation from `inputs_embeds` is not implemented for NucleusX yet."
                " The obstacle is to set the correct seqlen for the relative position encoding."
            )
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        forward_mode = kwargs.get("forward_mode", "parallel")
        if past_key_values is not None:
            # NOTE: when we have past_key_values, using recurrent mode will be faster.
            forward_mode = "recurrent"

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "forward_mode": forward_mode,
            }
        )
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:  # dict
            layer_past_kv = layer_past["prev_key_value"]  # [b, h, v_dim / h, qk_dim]
            layer_past_scale = layer_past["scale"]  # [b, h, 1, 1]
            if layer_past_scale.size(0) > 1:
                # this means that retention_mask is not None, so the scale for
                # each batch is different. We need to select the correct scale then.
                # NOTE: during huggingface generate, it will generate attention_mask
                # if it is None, so this linke will always be true. Still, having
                # this line here for safety.
                layer_past_scale = layer_past_scale.index_select(0, beam_idx)
            reordered_past += (
                {
                    "prev_key_value": layer_past_kv.index_select(0, beam_idx),
                    "scale": layer_past_scale,
                },
            )
        return reordered_past


@add_start_docstrings(
    """NucleusX Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    NUCLEUS_X_START_DOCSTRING,
)
class NucleusXForSequenceClassification(NucleusXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = NucleusXModel(config)
        self.score = nn.Linear(config.decoder_embed_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(NUCLEUS_X_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_mode: Optional[str] = None,
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple, NucleusXClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        forward_mode = forward_mode if forward_mode is not None else self.config.forward_mode
        recurrent_chunk_size = (
            recurrent_chunk_size if recurrent_chunk_size is not None else self.config.recurrent_chunk_size
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forward_mode=forward_mode,
            use_cache=use_cache,
            recurrent_chunk_size=recurrent_chunk_size,
            retention_rel_pos=retention_rel_pos,
        )

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NucleusXClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            retentions=outputs.retentions,
            attentions=outputs.attentions,
        )
