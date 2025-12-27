from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig

logger = logging.get_logger(__name__)

_NEG_INF = -1e9


class HumanVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class HumanVTorchRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(hidden_size, eps=eps)
        else:
            self.norm = HumanVRMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        self._seq_len_cached: int = 0
        self._device_cached: Optional[torch.device] = None
        self._dtype_cached: Optional[torch.dtype] = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = x.dtype
        seq_len = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 0

        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._device_cached != device
            or self._dtype_cached != dtype
        ):
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            self._device_cached = device
            self._dtype_cached = dtype

            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype=dtype)
            self._sin_cached = emb.sin().to(dtype=dtype)

        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        return cos, sin


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", None) or config.num_attention_heads)

        if self.num_kv_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {self.num_kv_heads}")
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_kv_heads}) cannot exceed num_attention_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads for grouped attention")

        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = float(getattr(config, "attention_scaling", 1.0)) / (self.head_dim ** 0.5)
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.rope_partial_rotary_factor = float(getattr(config, "rope_partial_rotary_factor", 1.0))

        # Sparse knobs
        self.use_sparse_attention = bool(getattr(config, "use_sparse_attention", False))
        self.sparse_attention_impl = str(getattr(config, "sparse_attention_impl", "local_global_block"))
        self.sparse_block_size = int(getattr(config, "sparse_block_size", 64))
        self.sparse_prefill_chunk_blocks = int(getattr(config, "sparse_prefill_chunk_blocks", 0) or 0)
        self.sparse_local_num_blocks = int(getattr(config, "sparse_local_num_blocks", 8))
        self.sparse_global_num_blocks = int(getattr(config, "sparse_global_num_blocks", 1))
        self.sparse_attention_window = int(getattr(config, "sparse_attention_window", 0) or 0)

        # Backend knobs
        self.kv_cache_dtype = str(getattr(config, "kv_cache_dtype", "auto"))
        self.attn_backend = str(getattr(config, "attn_backend", "gqa_matmul")).lower().strip()
        if self.attn_backend not in ("gqa_matmul", "sdpa"):
            self.attn_backend = "gqa_matmul"

        # Sparse caches (per-device, per-n_blocks)
        self._sparse_tables_cache = {}
        self._intra_causal_cache = {}

    def _kv_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_dtype == "auto":
            return x
        if self.kv_cache_dtype in ("bf16", "bfloat16"):
            return x.to(torch.bfloat16)
        if self.kv_cache_dtype in ("fp16", "float16"):
            return x.to(torch.float16)
        if self.kv_cache_dtype in ("fp32", "float32"):
            return x.to(torch.float32)
        return x

    def _reshape_q_grouped(self, q: torch.Tensor) -> torch.Tensor:
        bsz, h, t, d = q.shape
        return q.contiguous().view(bsz, self.num_kv_heads, self.num_kv_groups, t, d)

    def _get_intra_causal(self, block_size: int, device: torch.device) -> torch.Tensor:
        key = (block_size, device)
        m = self._intra_causal_cache.get(key)
        if m is not None:
            return m
        # intra[i,j] = True if j > i (future) within a block
        i = torch.arange(block_size, device=device)
        j = torch.arange(block_size, device=device)
        intra = j[None, :] > i[:, None]
        self._intra_causal_cache[key] = intra
        return intra

    def _get_sparse_tables(
        self, n_blocks: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return:
        - idx_table: (n_blocks, max_sel) block indices selected per query-block
        - idx_valid: (n_blocks, max_sel) whether the selection entry is valid
        - self_pos : (n_blocks,) index in selection where idx_table == i, else -1
        """
        key = (n_blocks, device)
        cached = self._sparse_tables_cache.get(key)
        if cached is not None:
            return cached

        # local blocks: include current + previous (sparse_local_num_blocks - 1)
        # global blocks: include earliest blocks [0 .. sparse_global_num_blocks-1]
        local_k = max(1, int(self.sparse_local_num_blocks))
        glob_k = max(0, int(self.sparse_global_num_blocks))

        max_sel = local_k + glob_k
        idx_table = torch.zeros((n_blocks, max_sel), dtype=torch.long, device=device)
        idx_valid = torch.zeros((n_blocks, max_sel), dtype=torch.bool, device=device)
        self_pos = torch.full((n_blocks,), -1, dtype=torch.long, device=device)

        for i in range(n_blocks):
            sel = []
            # globals first
            for g in range(glob_k):
                if g < n_blocks:
                    sel.append(g)
            # locals last: (i-local_k+1 .. i)
            start = max(0, i - (local_k - 1))
            for b in range(start, i + 1):
                sel.append(b)

            # unique, keep order
            seen = set()
            li = []
            for x in sel:
                if x not in seen:
                    li.append(x)
                    seen.add(x)

            vals = torch.tensor(li, dtype=torch.long, device=device)
            idx_table[i, : vals.numel()] = vals
            idx_valid[i, : vals.numel()] = True
            match = (vals == i).nonzero(as_tuple=False)
            if match.numel() > 0:
                self_pos[i] = int(match[0].item())

        self._sparse_tables_cache[key] = (idx_table, idx_valid, self_pos)
        return idx_table, idx_valid, self_pos

    def _pad_to_blocks(self, x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len <= 0:
            return x
        pad = torch.zeros((*x.shape[:-2], pad_len, x.shape[-1]), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-2)

    def _pad_mask_to_blocks(self, mask: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len <= 0:
            return mask
        pad = torch.zeros((mask.shape[0], pad_len), device=mask.device, dtype=mask.dtype)
        return torch.cat([mask, pad], dim=-1)

    def _apply_partial_rope(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.rope_partial_rotary_factor
        if f >= 0.999999:
            return _apply_rotary(q, k, cos, sin)

        rotary_dim = int(self.head_dim * f)
        if rotary_dim <= 0:
            return q, k

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        q_rot, k_rot = _apply_rotary(q_rot, k_rot, cos[..., :rotary_dim], sin[..., :rotary_dim])
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def _sdpa_mha_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """SDPA attention (MHA only).

        Notes:
        - If `attn_mask` is provided, we must set `is_causal=False` and fold causality into the mask.
        - If `attn_mask` is a float mask/bias, its dtype must match the query dtype (PyTorch requirement).
        """
        dropout_p = self.attention_dropout if self.training else 0.0
        if attention_mask_4d is None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
            return out

        if attention_mask_4d.dtype is not torch.bool and attention_mask_4d.dtype != q.dtype:
            attention_mask_4d = attention_mask_4d.to(dtype=q.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask_4d, dropout_p=dropout_p, is_causal=False
        )
        return out

    def _grouped_dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, _, q_len, d = q.shape
        qg = self._reshape_q_grouped(q)

        scores = torch.matmul(
            qg.to(torch.float32),
            k.unsqueeze(2).transpose(-2, -1).to(torch.float32),
        ) * self.scaling

        if attention_mask_4d is not None:
            m = attention_mask_4d[:, 0].to(dtype=torch.float32)
            scores = scores + m[:, None, None, :, :]

        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs.to(v.dtype), v.unsqueeze(2))
        out = out.to(dtype=q.dtype).reshape(bsz, self.num_heads, q_len, d)
        return out

    def _local_global_block_sparse_prefill_grouped(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, h, q_len, d = q.shape
        if h != self.num_heads:
            raise ValueError(f"Expected q heads={self.num_heads}, got {h}")

        seqlen = k.shape[-2]
        b = self.sparse_block_size
        if b <= 0:
            raise ValueError("sparse_block_size must be > 0")

        # pad seq to blocks
        n_blocks = (seqlen + b - 1) // b
        pad_len = n_blocks * b - seqlen
        if pad_len > 0:
            k = self._pad_to_blocks(k, pad_len)
            v = self._pad_to_blocks(v, pad_len)
            if attention_mask_2d is not None:
                attention_mask_2d = self._pad_mask_to_blocks(attention_mask_2d, pad_len)
            seqlen = n_blocks * b

        # key-valid mask (bsz, seqlen) float32 for arithmetic
        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, seqlen), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
            if key_valid.shape[1] != seqlen:
                raise ValueError(f"attention_mask_2d has wrong length: {key_valid.shape[1]} vs {seqlen}")

        # reshape into blocks
        q_blocks = q.view(bsz, self.num_kv_heads, self.num_kv_groups, n_blocks, b, d)
        k_blocks = k.view(bsz, self.num_kv_heads, n_blocks, b, d)
        v_blocks = v.view(bsz, self.num_kv_heads, n_blocks, b, d)
        key_valid_blocks = key_valid.view(bsz, n_blocks, b)

        idx_table, idx_valid, self_pos = self._get_sparse_tables(n_blocks, q.device)
        intra = self._get_intra_causal(b, q.device)

        chunk = self.sparse_prefill_chunk_blocks if self.sparse_prefill_chunk_blocks > 0 else n_blocks
        chunk = max(1, int(chunk))

        out = torch.zeros((bsz, self.num_kv_heads, self.num_kv_groups, n_blocks, b, d), device=q.device, dtype=q.dtype)

        max_sel = idx_table.shape[1]
        offsets = torch.arange(b, device=q.device).repeat(max_sel)
        idx_valid_token = idx_valid[..., None].expand(n_blocks, max_sel, b).reshape(n_blocks, max_sel * b)

        for start in range(0, n_blocks, chunk):
            end = min(n_blocks, start + chunk)
            clen = end - start

            qb = q_blocks[:, :, :, start:end]
            idx_c = idx_table[start:end]
            idxv_c = idx_valid_token[start:end]
            selfpos_c = self_pos[start:end]

            k_sel = k_blocks[:, :, idx_c]
            v_sel = v_blocks[:, :, idx_c]

            k_flat = k_sel.reshape(bsz, self.num_kv_heads, clen, max_sel * b, d)
            v_flat = v_sel.reshape(bsz, self.num_kv_heads, clen, max_sel * b, d)

            scores = torch.matmul(
                qb.to(torch.float32),
                k_flat.unsqueeze(2).transpose(-2, -1).to(torch.float32),
            ) * self.scaling

            kv_mask_sel = key_valid_blocks[:, idx_c].reshape(bsz, clen, max_sel * b)
            kv_mask_sel = kv_mask_sel * idxv_c[None, :, :].to(kv_mask_sel.dtype)
            scores = scores + (1.0 - kv_mask_sel[:, None, None, :, None, :]) * _NEG_INF

            causal_mask = torch.zeros((clen, b, max_sel * b), device=q.device, dtype=torch.bool)
            active = selfpos_c >= 0
            if active.any():
                pos = torch.clamp(selfpos_c, min=0)
                onehot = F.one_hot(pos, num_classes=max_sel).to(dtype=torch.bool)
                onehot = onehot & active[:, None]
                for s in range(max_sel):
                    if not onehot[:, s].any():
                        continue
                    causal_mask[:, :, s * b : (s + 1) * b] |= (onehot[:, s][:, None, None] & intra[None, :, :])

            scores = scores.masked_fill(causal_mask[None, None, None, :, :, :], _NEG_INF)

            if self.sparse_attention_window > 0:
                max_ctx = int(self.sparse_attention_window)
                key_abs = idx_c.repeat_interleave(b, dim=1) * b + offsets[None, :]
                q_block_ids = torch.arange(start, end, device=q.device)
                q_abs = q_block_ids[:, None] * b + torch.arange(b, device=q.device)[None, :]
                min_allowed = (q_abs[:, :, None] - (max_ctx - 1)).clamp(min=0)
                window_mask = key_abs[:, None, :] < min_allowed
                scores = scores.masked_fill(window_mask[None, None, None, :, :, :], _NEG_INF)

            probs = torch.softmax(scores, dim=-1)
            probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

            o = torch.matmul(probs.to(v_flat.dtype), v_flat.unsqueeze(2)).to(dtype=q.dtype)
            out[:, :, :, start:end] = o

        out = out.reshape(bsz, self.num_heads, n_blocks * b, d)
        return out[:, :, :seqlen, :]

    def _local_global_block_sparse_decode_one_grouped(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, h, q_len, d = q.shape
        if q_len != 1:
            raise ValueError("decode_one expects q_len == 1")
        if h != self.num_heads:
            raise ValueError(f"Expected q heads={self.num_heads}, got {h}")

        seqlen = k.shape[-2]
        b = self.sparse_block_size
        if b <= 0:
            raise ValueError("sparse_block_size must be > 0")

        n_blocks = (seqlen + b - 1) // b
        pad_len = n_blocks * b - seqlen
        if pad_len > 0:
            k = self._pad_to_blocks(k, pad_len)
            v = self._pad_to_blocks(v, pad_len)
            if attention_mask_2d is not None:
                attention_mask_2d = self._pad_mask_to_blocks(attention_mask_2d, pad_len)
            seqlen = n_blocks * b

        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, seqlen), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
            if key_valid.shape[1] != seqlen:
                raise ValueError(f"attention_mask_2d has wrong length: {key_valid.shape[1]} vs {seqlen}")

        qg = self._reshape_q_grouped(q)

        k_blocks = k.view(bsz, self.num_kv_heads, n_blocks, b, d)
        v_blocks = v.view(bsz, self.num_kv_heads, n_blocks, b, d)
        key_valid_blocks = key_valid.view(bsz, n_blocks, b)

        idx_table, idx_valid, self_pos = self._get_sparse_tables(n_blocks, q.device)

        q_pos = seqlen - 1
        q_block = q_pos // b
        q_intra = q_pos % b

        idx_row = idx_table[q_block]
        idxv_row = idx_valid[q_block]

        max_sel = idx_row.numel()
        k_sel = k_blocks[:, :, idx_row]
        v_sel = v_blocks[:, :, idx_row]
        k_flat = k_sel.reshape(bsz, self.num_kv_heads, max_sel * b, d)
        v_flat = v_sel.reshape(bsz, self.num_kv_heads, max_sel * b, d)

        scores = torch.matmul(
            qg.to(torch.float32),
            k_flat.unsqueeze(2).transpose(-2, -1).to(torch.float32),
        ) * self.scaling

        kv_mask_sel = key_valid_blocks[:, idx_row].reshape(bsz, max_sel * b)
        idxv_tok = idxv_row[:, None].expand(max_sel, b).reshape(max_sel * b).to(kv_mask_sel.dtype)
        kv_mask_sel = kv_mask_sel * idxv_tok[None, :]
        scores = scores + (1.0 - kv_mask_sel[:, None, None, None, :]) * _NEG_INF

        self_p = int(self_pos[q_block].item())
        if self_p >= 0:
            s = self_p * b
            e = s + b
            mask = torch.arange(b, device=q.device) > q_intra
            scores[:, :, :, 0, s:e] = scores[:, :, :, 0, s:e].masked_fill(mask[None, None, None, :], _NEG_INF)

        if self.sparse_attention_window > 0:
            max_ctx = int(self.sparse_attention_window)
            offsets = torch.arange(b, device=q.device).repeat(max_sel)
            key_abs = idx_row.repeat_interleave(b) * b + offsets
            min_allowed = max(0, q_pos - (max_ctx - 1))
            scores = scores.masked_fill(key_abs[None, None, None, None, :] < min_allowed, _NEG_INF)

        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs.to(v_flat.dtype), v_flat.unsqueeze(2)).to(dtype=q.dtype)
        out = out.reshape(bsz, self.num_heads, 1, d)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_partial_rope(q, k, cos, sin)

        if past_key_values is not None:
            k = self._kv_dtype(k)
            v = self._kv_dtype(v)
            k, v = past_key_values.update(k, v, self.layer_idx)

        k_len = k.shape[-2]
        use_sparse = (
            self.use_sparse_attention
            and self.layer_type == "sliding_attention"
            and self.sparse_attention_impl == "local_global_block"
        )

        if use_sparse:
            if q_len == k_len:
                attn_out = self._local_global_block_sparse_prefill_grouped(q, k, v, attention_mask_2d)
            elif q_len == 1:
                attn_out = self._local_global_block_sparse_decode_one_grouped(q, k, v, attention_mask_2d)
            else:
                attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)
        else:
            if self.attn_backend == "sdpa" and self.num_kv_heads == self.num_heads:
                attn_out = self._sdpa_mha_attention(q, k, v, attention_mask_4d)
            else:
                attn_out = self._grouped_dense_attention(q, k, v, attention_mask_4d)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)
        return attn_out, None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_type = "full_attention"
        else:
            layer_type = str(layer_types[layer_idx])

        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx, layer_type=layer_type)
        self.mlp = HumanVMLP(config)

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "rmsnorm")).lower()
        if norm_backend in ("torch_rmsnorm", "rmsnorm_torch", "fused_rmsnorm"):
            Norm = HumanVTorchRMSNorm
        elif norm_backend in ("layernorm", "ln"):
            Norm = nn.LayerNorm
        else:
            Norm = HumanVTorchRMSNorm

        if Norm is nn.LayerNorm:
            self.input_layernorm = Norm(config.hidden_size, eps=eps)
            self.post_attention_layernorm = Norm(config.hidden_size, eps=eps)
        else:
            self.input_layernorm = Norm(config.hidden_size, eps=eps)
            self.post_attention_layernorm = Norm(config.hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask_4d=attention_mask_4d,
            attention_mask_2d=attention_mask_2d,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + attn_out

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
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(int(config.num_hidden_layers))])

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "rmsnorm")).lower()
        if norm_backend in ("layernorm", "ln"):
            self.norm = nn.LayerNorm(config.hidden_size, eps=eps)
        else:
            self.norm = HumanVTorchRMSNorm(config.hidden_size, eps=eps)

        rope_base = float(getattr(config, "rope_theta", 10000.0))
        self.rotary_emb = HumanVRotaryEmbedding(
            dim=int(config.head_dim),
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 2048)),
            base=rope_base,
        )

        self._causal_cache = {}
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_causal_mask(
        self, q_len: int, src_len: int, past_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Cache key includes dtype to avoid SDPA dtype mismatches and repeated casts.
        key = (q_len, src_len, past_len, device, dtype)
        m = self._causal_cache.get(key)
        if m is not None:
            return m

        m = torch.triu(
            torch.full((q_len, src_len), _NEG_INF, device=device, dtype=dtype),
            diagonal=1 + past_len,
        )
        m = m[None, None, :, :]
        self._causal_cache[key] = m
        return m

    def _prepare_attention_masks(
        self,
        attention_mask_2d: torch.Tensor,
        q_len: int,
        past_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build additive attention bias mask of shape (bsz, 1, q_len, src_len).

        - 0.0 for tokens that can be attended to
        - large negative for masked tokens (padding and causal future)
        """
        device = attention_mask_2d.device
        src_len = int(attention_mask_2d.shape[1])

        causal = self._get_causal_mask(
            q_len=q_len, src_len=src_len, past_len=past_len, device=device, dtype=dtype
        )

        key_valid = attention_mask_2d.to(dtype=torch.bool)
        pad_bias = (~key_valid)[:, None, None, :].to(dtype=dtype) * torch.tensor(_NEG_INF, device=device, dtype=dtype)
        return causal + pad_bias

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, q_len = inputs_embeds.shape[:2]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask_2d = torch.ones((bsz, past_len + q_len), device=inputs_embeds.device, dtype=torch.bool)
        else:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (bsz, seq)")
            attention_mask_2d = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
            # Generation often provides only current-step mask (bsz, q_len)
            if attention_mask_2d.shape[1] == q_len and past_len > 0:
                pad = torch.ones((bsz, past_len), device=inputs_embeds.device, dtype=torch.bool)
                attention_mask_2d = torch.cat([pad, attention_mask_2d], dim=-1)

        position_ids = torch.arange(past_len, past_len + q_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_4d = self._prepare_attention_masks(
            attention_mask_2d, q_len=q_len, past_len=past_len, dtype=inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    attention_mask_2d,
                    position_embeddings,
                    past_key_values,
                    output_attentions,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask_4d=attention_mask_4d,
                    attention_mask_2d=attention_mask_2d,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class HumanVForCausalLM(HumanVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: HumanVConfig):
        super().__init__(config)
        self.model = HumanVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
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
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel"]
