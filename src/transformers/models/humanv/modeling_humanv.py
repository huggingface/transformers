# Copyright 2025 The HumanV Team. All rights reserved.
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

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig

logger = logging.get_logger(__name__)


def _get_dtype_from_str(name: str, fallback: torch.dtype) -> torch.dtype:
    name = (name or "").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    return fallback


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(dtype=input_dtype)


class HumanVScaleNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(()))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.to(torch.float32)
        denom = torch.linalg.vector_norm(x_f, ord=2, dim=-1, keepdim=True).clamp_min(self.eps)
        y = x_f * (self.g * (self.hidden_size**0.5)) / denom
        return y.to(dtype=dtype)


def _make_norm(config: HumanVConfig) -> type[nn.Module]:
    backend = getattr(config, "norm_backend", None)
    backend = (backend or "rmsnorm").lower()
    if backend in {"layernorm", "ln"}:
        return nn.LayerNorm
    if backend in {"scalenorm", "sn"}:
        return HumanVScaleNorm
    return HumanVRMSNorm


def _make_activation_fn(config: HumanVConfig):
    backend = getattr(config, "activation_backend", None)
    if backend is None:
        return ACT2FN[config.hidden_act]
    backend = backend.lower()
    if backend in ACT2FN:
        return ACT2FN[backend]
    if backend in {"sqrelu", "relu2", "squared_relu"}:

        def _sqrelu(x: torch.Tensor) -> torch.Tensor:
            return torch.square(torch.relu(x))

        return _sqrelu
    raise ValueError(f"Unknown activation_backend={backend}")


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        self.dim = int(config.head_dim)
        self.base = float(getattr(config, "rope_theta", None) or 10000.0)
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].to(device=x.device, dtype=torch.float32).expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].to(device=x.device, dtype=torch.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=getattr(config, "mlp_bias", False))
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=getattr(config, "mlp_bias", False))
        self.act_fn = _make_activation_fn(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _make_causal_mask(
    input_shape: tuple[int, int],
    past_key_values_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bsz, tgt_len = input_shape
    src_len = tgt_len + past_key_values_length
    mask = torch.full((tgt_len, src_len), -1e9, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1 + past_key_values_length)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, src_len)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    bsz, src_len = mask.shape
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = (1.0 - expanded_mask) * -1e9
    return inverted


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=getattr(config, "attention_bias", False))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=getattr(config, "attention_bias", False))

        self.use_selective_rope = bool(getattr(config, "use_selective_rope", False))
        self.selective_rope_scale = float(getattr(config, "selective_rope_scale", 0.25))
        if self.use_selective_rope:
            self.rope_phase = nn.Linear(config.hidden_size, self.num_heads, bias=False)

    def _get_layer_type(self) -> str:
        layer_types = getattr(self.config, "layer_types", None)
        if isinstance(layer_types, list) and self.layer_idx < len(layer_types):
            return str(layer_types[self.layer_idx])
        use_sliding = bool(getattr(self.config, "use_sliding_window", False))
        if use_sliding:
            max_window_layers = int(getattr(self.config, "max_window_layers", 0))
            if self.layer_idx >= max_window_layers:
                return "sliding_attention"
        return "full_attention"

    def _maybe_apply_selective_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hidden_states: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_selective_rope:
            return cos, sin
        phase = self.rope_phase(hidden_states).transpose(1, 2).unsqueeze(-1)
        phase = phase.to(dtype=torch.float32)
        delta = torch.tanh(phase) * self.selective_rope_scale
        cos_d = torch.cos(delta).to(dtype=dtype, device=device)
        sin_d = torch.sin(delta).to(dtype=dtype, device=device)

        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        cos_new = cos_b * cos_d - sin_b * sin_d
        sin_new = sin_b * cos_d + cos_b * sin_d
        return cos_new, sin_new

    def _full_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_implementation = str(getattr(self.config, "attn_implementation", "eager") or "eager").lower()
        if attn_implementation in {"sdpa", "flash_attention_2", "flash_attention"} and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask_4d,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            return attn_output, torch.empty(0, device=query_states.device, dtype=query_states.dtype)

        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_scores = attn_scores.to(torch.float32)

        if attention_mask_4d is not None:
            attn_scores = attn_scores + attention_mask_4d.to(dtype=torch.float32)

        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.attention_dropout, training=self.training)

        attn_probs_v = attn_probs.to(dtype=query_states.dtype)
        attn_output = torch.matmul(attn_probs_v, value_states)
        return attn_output, attn_probs_v

    def _local_global_block_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = q.shape
        block_size = int(getattr(self.config, "sparse_block_size", 64))
        local_num_blocks = int(getattr(self.config, "sparse_local_num_blocks", 4))
        global_num_blocks = int(getattr(self.config, "sparse_global_num_blocks", 2))
        global_stride = int(getattr(self.config, "sparse_global_block_stride", 4))
        window = int(getattr(self.config, "sparse_attention_window", 0) or 0)
        if window > 0:
            local_num_blocks = min(local_num_blocks, max(1, (window + block_size - 1) // block_size))
        prefill_chunk_blocks = int(getattr(self.config, "sparse_prefill_chunk_blocks", 0) or 0)

        if block_size <= 0:
            raise ValueError("sparse_block_size must be > 0")

        k_len = k.shape[2]
        if q_len != k_len:
            if q_len != 1:
                attn_mask_4d = None
                if attention_mask_2d is not None:
                    attn_mask_4d = _expand_mask(attention_mask_2d, dtype=torch.float32, tgt_len=q_len)
                out, _ = self._full_attention(q, k, v, attn_mask_4d)
                return out
            pos = k_len - 1
            block = pos // block_size
            offset = pos % block_size

            block_start = block * block_size
            block_end = block_start + block_size
            kv_pad = (block_end - k_len)
            if kv_pad > 0:
                k = torch.nn.functional.pad(k, (0, 0, 0, kv_pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, kv_pad))
                if attention_mask_2d is not None:
                    attention_mask_2d = torch.nn.functional.pad(attention_mask_2d, (0, kv_pad), value=0)
                k_len = k.shape[2]

            local_blocks = torch.arange(max(0, block - local_num_blocks + 1), block + 1, device=q.device, dtype=torch.long)
            global_blocks = torch.arange(0, block + 1, global_stride, device=q.device, dtype=torch.long)
            if global_num_blocks > 0 and global_blocks.numel() > global_num_blocks:
                global_blocks = global_blocks[-global_num_blocks:]
            sel_blocks = torch.unique(torch.cat([local_blocks, global_blocks], dim=0), sorted=True)

            k_sel = []
            v_sel = []
            m_sel = []
            for b in sel_blocks.tolist():
                s = b * block_size
                e = s + block_size
                k_sel.append(k[:, :, s:e, :])
                v_sel.append(v[:, :, s:e, :])
                if attention_mask_2d is None:
                    m_sel.append(torch.ones((bsz, block_size), device=q.device, dtype=torch.float32))
                else:
                    m_sel.append(attention_mask_2d[:, s:e].to(dtype=torch.float32))
            k_sel = torch.cat(k_sel, dim=2)
            v_sel = torch.cat(v_sel, dim=2)
            km = torch.cat(m_sel, dim=1)

            scores = torch.matmul(q.to(torch.float32), k_sel.transpose(2, 3).to(torch.float32)) * self.scaling
            if sel_blocks.numel() > 0:
                cur_in_sel = (sel_blocks == block).nonzero(as_tuple=False)
                if cur_in_sel.numel() > 0:
                    cur_idx = int(cur_in_sel[0].item())
                    start = cur_idx * block_size
                    end = start + block_size
                    causal = torch.arange(block_size, device=q.device) > offset
                    scores[:, :, :, start:end] = scores[:, :, :, start:end] + causal.to(dtype=torch.float32).view(
                        1, 1, 1, block_size
                    ) * -1e9

            scores = scores + (1.0 - km.view(bsz, 1, 1, -1)) * -1e9
            probs = torch.softmax(scores, dim=-1).to(dtype=v_sel.dtype)
            probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
            out = torch.matmul(probs, v_sel)
            return out

        pad_len = (block_size - (q_len % block_size)) % block_size
        if pad_len:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
            if attention_mask_2d is not None:
                attention_mask_2d = torch.nn.functional.pad(attention_mask_2d, (0, pad_len), value=0)

        seq_len = q.shape[2]
        n_blocks = seq_len // block_size

        q_b = q.view(bsz, num_heads, n_blocks, block_size, head_dim)
        k_b = k.view(bsz, num_heads, n_blocks, block_size, head_dim)
        v_b = v.view(bsz, num_heads, n_blocks, block_size, head_dim)

        if attention_mask_2d is None:
            attn_mask_blocks = torch.ones((bsz, n_blocks, block_size), device=q.device, dtype=torch.float32)
        else:
            attn_mask_blocks = attention_mask_2d.view(bsz, n_blocks, block_size).to(dtype=torch.float32)

        t = torch.arange(n_blocks, device=q.device)
        local_offsets = torch.arange(local_num_blocks - 1, -1, -1, device=q.device)
        local_idx = t[:, None] - local_offsets[None, :]
        local_valid = local_idx >= 0
        local_idx = local_idx.clamp_min(0)

        global_idx = torch.arange(global_num_blocks, device=q.device)[None, :] * global_stride
        global_idx = global_idx.expand(n_blocks, -1)
        global_valid = global_idx <= t[:, None]

        out_num_blocks = global_num_blocks
        out_offsets = torch.arange(out_num_blocks, device=q.device)
        out_idx = (t[:, None] - local_num_blocks) - out_offsets[None, :] * global_stride
        out_valid = out_idx >= 0
        out_idx = out_idx.clamp_min(0)

        sel_idx = torch.cat([local_idx, global_idx], dim=1)
        sel_valid = torch.cat([local_valid, global_valid], dim=1)

        sel_count = sel_idx.shape[1]

        gather_idx = sel_idx.view(1, 1, n_blocks, sel_count, 1, 1).expand(bsz, num_heads, n_blocks, sel_count, block_size, head_dim)
        k_sel = torch.gather(
            k_b.unsqueeze(3).expand(bsz, num_heads, n_blocks, sel_count, block_size, head_dim),
            dim=2,
            index=gather_idx,
        )
        v_sel = torch.gather(
            v_b.unsqueeze(3).expand(bsz, num_heads, n_blocks, sel_count, block_size, head_dim),
            dim=2,
            index=gather_idx,
        )

        km_gather_idx = sel_idx.view(1, n_blocks, sel_count, 1).expand(bsz, n_blocks, sel_count, block_size)
        km_sel = torch.gather(
            attn_mask_blocks.unsqueeze(2).expand(bsz, n_blocks, sel_count, block_size),
            dim=1,
            index=km_gather_idx,
        )
        km_sel = km_sel * sel_valid.view(1, n_blocks, sel_count, 1).to(dtype=km_sel.dtype)

        intra = torch.triu(torch.full((block_size, block_size), -1e9, device=q.device, dtype=torch.float32), diagonal=1)
        same_block = sel_idx == t[:, None]
        causal = (same_block[:, :, None, None].to(dtype=torch.float32) * intra[None, None, :, :])
        causal = causal.permute(0, 2, 1, 3).reshape(n_blocks, block_size, sel_count * block_size)

        k_flat = k_sel.reshape(bsz, num_heads, n_blocks, sel_count * block_size, head_dim)
        v_flat = v_sel.reshape(bsz, num_heads, n_blocks, sel_count * block_size, head_dim)
        km_flat = km_sel.reshape(bsz, n_blocks, sel_count * block_size)

        k_pool = k_b.mean(dim=3)
        v_pool = v_b.mean(dim=3)
        kp_idx = out_idx
        kp_valid = out_valid
        kp_gather = kp_idx.view(1, 1, n_blocks, out_num_blocks, 1).expand(bsz, num_heads, n_blocks, out_num_blocks, head_dim)
        k_pool_sel = torch.gather(
            k_pool.unsqueeze(3).expand(bsz, num_heads, n_blocks, out_num_blocks, head_dim),
            dim=2,
            index=kp_gather,
        )
        v_pool_sel = torch.gather(
            v_pool.unsqueeze(3).expand(bsz, num_heads, n_blocks, out_num_blocks, head_dim),
            dim=2,
            index=kp_gather,
        )

        km_pool_idx = kp_idx.view(1, n_blocks, out_num_blocks).expand(bsz, n_blocks, out_num_blocks)
        km_pool = torch.gather(attn_mask_blocks.amax(dim=2), dim=1, index=km_pool_idx)
        km_pool = km_pool * kp_valid.view(1, n_blocks, out_num_blocks).to(dtype=km_pool.dtype)

        def _sparse_attend(q_chunk: torch.Tensor, start_block: int, end_block: int) -> torch.Tensor:
            qb = q_chunk[:, :, start_block:end_block]
            qf = qb.to(torch.float32)
            kf = k_flat[:, :, start_block:end_block].to(torch.float32)
            vf = v_flat[:, :, start_block:end_block]
            scores = torch.einsum("bhntqd,bhntkd->bhntqk", qf, kf) * self.scaling
            scores = scores + causal[start_block:end_block].view(1, 1, -1, block_size, sel_count * block_size)
            scores = scores + (1.0 - km_flat[:, start_block:end_block].view(bsz, 1, -1, 1, sel_count * block_size)) * -1e9

            kpf = k_pool_sel[:, :, start_block:end_block].to(torch.float32)
            sp = torch.einsum("bhntqd,bhntpd->bhntqp", qf, kpf) * self.scaling
            sp = sp + (1.0 - km_pool[:, start_block:end_block].view(bsz, 1, -1, 1, out_num_blocks)) * -1e9

            scores = torch.cat([scores, sp], dim=-1)
            probs = torch.softmax(scores, dim=-1)
            probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)

            p_full = probs[..., : sel_count * block_size].to(dtype=vf.dtype)
            p_pool = probs[..., sel_count * block_size :].to(dtype=vf.dtype)

            out_full = torch.einsum("bhntqk,bhntkd->bhntqd", p_full, vf)
            out_pool = torch.einsum(
                "bhntqp,bhntpd->bhntqd", p_pool, v_pool_sel[:, :, start_block:end_block].to(dtype=vf.dtype)
            )
            return out_full + out_pool

        if prefill_chunk_blocks and q_len > 1:
            outs = []
            for s in range(0, n_blocks, prefill_chunk_blocks):
                e = min(n_blocks, s + prefill_chunk_blocks)
                outs.append(_sparse_attend(q_b, s, e))
            out_b = torch.cat(outs, dim=2)
        else:
            out_b = _sparse_attend(q_b, 0, n_blocks)

        out = out_b.reshape(bsz, num_heads, seq_len, head_dim)
        if pad_len:
            out = out[:, :, :q_len, :]
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos, sin = self._maybe_apply_selective_rope(cos, sin, hidden_states, query_states.dtype, query_states.device)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            kv_dtype = _get_dtype_from_str(getattr(self.config, "kv_cache_dtype", "bf16"), key_states.dtype)
            key_states = key_states.to(dtype=kv_dtype)
            value_states = value_states.to(dtype=kv_dtype)
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        layer_type = self._get_layer_type()
        use_sparse = bool(getattr(self.config, "use_sparse_attention", False)) and layer_type == "sliding_attention"
        sparse_impl = str(getattr(self.config, "sparse_attention_impl", "local_global_block") or "local_global_block")

        if use_sparse and sparse_impl == "local_global_block":
            attn_output = self._local_global_block_sparse(query_states, key_states, value_states, attention_mask_2d)
            attn_weights = torch.empty(0, device=attn_output.device, dtype=attn_output.dtype)
        else:
            attn_output, attn_weights = self._full_attention(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), attn_weights


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        norm_cls = _make_norm(config)
        if norm_cls is nn.LayerNorm:
            self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx)
        self.mlp = HumanVMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HumanVPreTrainedModel(PreTrainedModel):
    config_class = HumanVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HumanVDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module: nn.Module):
        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HumanVModel(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        norm_cls = _make_norm(config)
        if norm_cls is nn.LayerNorm:
            self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _prepare_attention_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len = input_shape
        device = inputs_embeds.device
        dtype = torch.float32

        if attention_mask is None:
            attention_mask_2d = torch.ones((bsz, tgt_len + past_key_values_length), device=device, dtype=torch.long)
        else:
            attention_mask_2d = attention_mask.to(device=device)
            if attention_mask_2d.dim() == 2 and attention_mask_2d.shape[1] == tgt_len and past_key_values_length > 0:
                prefix = torch.ones((bsz, past_key_values_length), device=device, dtype=attention_mask_2d.dtype)
                attention_mask_2d = torch.cat([prefix, attention_mask_2d], dim=-1)

        causal_mask = _make_causal_mask((bsz, tgt_len), past_key_values_length, dtype=dtype, device=device)
        padding_mask = _expand_mask(attention_mask_2d, dtype=dtype, tgt_len=tgt_len)
        attention_mask_4d = causal_mask + padding_mask
        return attention_mask_4d, attention_mask_2d

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = bool(use_cache if use_cache is not None else getattr(self.config, "use_cache", True))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch_size, seq_length = inputs_embeds.shape[:2]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        max_pos = int(getattr(self.config, "max_position_embeddings", 0) or 0)
        if max_pos > 0 and (past_length + seq_length) > max_pos:
            raise ValueError(f"Sequence length {past_length + seq_length} exceeds max_position_embeddings={max_pos}")

        position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_4d, attention_mask_2d = self._prepare_attention_masks(attention_mask, (batch_size, seq_length), inputs_embeds, past_length)

        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "use_cache": kwargs.get("use_cache", True),
        }


class HumanVForSequenceClassification(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.model = HumanVModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state
        pooled = hidden[:, -1, :]
        logits = self.score(pooled).to(torch.float32)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1).to(logits.dtype))
            else:
                loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HumanVForTokenClassification(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.model = HumanVModel(config)
        self.dropout = nn.Dropout(float(getattr(config, "classifier_dropout", 0.0)))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output).to(torch.float32)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class HumanVForQuestionAnswering(HumanVPreTrainedModel):
    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = self.qa_outputs(outputs.last_hidden_state).to(torch.float32)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HumanVForCausalLM",
    "HumanVForQuestionAnswering",
    "HumanVForSequenceClassification",
    "HumanVForTokenClassification",
    "HumanVModel",
    "HumanVPreTrainedModel",
]
