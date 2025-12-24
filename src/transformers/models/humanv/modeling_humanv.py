from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_humanv import HumanVConfig

logger = logging.get_logger(__name__)


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
        return (self.weight * hidden_states).to(input_dtype)


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        dim = int(config.head_dim)
        rope_theta = 10000.0
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters:
            rope_theta = float(rope_parameters.get("rope_theta", rope_theta))
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
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
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, n_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv * n_rep, slen, head_dim)


class HumanVMLP(nn.Module):
    def __init__(self, config: HumanVConfig):
        super().__init__()
        mlp_bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class HumanVAttention(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.attention_bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=self.attention_bias)

    def _get_layer_type(self) -> str:
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            return "full_attention"
        if self.layer_idx >= len(layer_types):
            return layer_types[-1]
        return layer_types[self.layer_idx]

    def _get_backend(self) -> str:
        backend = getattr(self.config, "attn_backend", None)
        if backend is not None:
            return str(backend)
        impl = getattr(self.config, "attn_implementation", None)
        if impl is None:
            return "sdpa"
        impl = str(impl)
        if impl in {"sdpa", "flash_attention_2"}:
            return "sdpa"
        return "matmul"

    def _dense_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask_4d: Optional[torch.Tensor]):
        dropout_p = self.attention_dropout if self.training else 0.0
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        try:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_4d, dropout_p=dropout_p, is_causal=False)
        except TypeError:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_4d, dropout_p=dropout_p)

    def _dense_matmul_fp32(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask_4d: Optional[torch.Tensor]
    ):
        scores = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-2, -1)) * self.scaling
        if attn_mask_4d is not None:
            scores = scores + attn_mask_4d.to(dtype=torch.float32)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0.0:
            probs = F.dropout(probs, p=self.attention_dropout, training=True)
        return torch.matmul(probs, v.to(torch.float32)).to(dtype=q.dtype)

    def _sliding_mask_4d(self, q_len: int, k_len: int, past_len: int, device: torch.device) -> torch.Tensor:
        window = int(getattr(self.config, "sliding_window", 0) or 0)
        if window <= 0:
            window = int(getattr(self.config, "sparse_attention_window", 0) or 0)
        if window <= 0:
            return torch.zeros((1, 1, q_len, k_len), device=device, dtype=torch.float32)
        q_pos = torch.arange(past_len, past_len + q_len, device=device)
        k_pos = torch.arange(0, k_len, device=device)
        dist = q_pos[:, None] - k_pos[None, :]
        invalid = (dist < 0) | (dist >= window)
        mask = torch.zeros((q_len, k_len), device=device, dtype=torch.float32)
        mask = mask.masked_fill(invalid, -1e9)
        return mask[None, None, :, :]

    def _local_global_block_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
        past_len: int,
        block_size: int,
        local_num_blocks: int,
        global_num_blocks: int,
        global_stride: int,
        window_tokens: int,
    ) -> torch.Tensor:
        bsz, nheads, q_len, d = q.shape
        _, _, k_len, _ = k.shape

        if (
            q_len % block_size != 0
            or k_len % block_size != 0
            or past_len % block_size != 0
            or q_len < block_size
            or k_len < block_size
        ):
            return self._local_global_gather_sparse(
                q=q,
                k=k,
                v=v,
                attention_mask_2d=attention_mask_2d,
                past_len=past_len,
                block_size=block_size,
                global_num_blocks=global_num_blocks,
                global_stride=global_stride,
                window_tokens=window_tokens,
            )

        q_blocks = q_len // block_size
        k_blocks = k_len // block_size
        base_q_block = past_len // block_size

        q_block_ids = base_q_block + torch.arange(q_blocks, device=q.device, dtype=torch.long)
        local_offsets = torch.arange(local_num_blocks, device=q.device, dtype=torch.long)
        local_idx = q_block_ids[:, None] - (local_num_blocks - 1 - local_offsets[None, :])
        local_idx = local_idx.clamp(min=0, max=max(k_blocks - 1, 0))

        if global_num_blocks > 0:
            g = torch.arange(global_num_blocks, device=q.device, dtype=torch.long) * max(int(global_stride), 1)
            g = g.clamp(min=0, max=max(k_blocks - 1, 0))
            global_idx = g[None, :].expand(q_blocks, -1)
        else:
            global_idx = torch.zeros((q_blocks, 0), device=q.device, dtype=torch.long)

        sel_idx = torch.cat([global_idx, local_idx], dim=1)
        sel_len = sel_idx.shape[1]

        valid = sel_idx <= q_block_ids[:, None]
        if window_tokens and window_tokens > 0:
            window_blocks = (int(window_tokens) + block_size - 1) // block_size
            min_k = (q_block_ids - (window_blocks - 1)).clamp(min=0)
            valid = valid & (sel_idx >= min_k[:, None])

        if sel_len > 1:
            eq = sel_idx[:, :, None].eq(sel_idx[:, None, :])
            lower = torch.tril(torch.ones((sel_len, sel_len), device=q.device, dtype=torch.bool), diagonal=-1)
            has_prev = (eq & lower[None, :, :]).any(dim=-1)
            keep = (~has_prev) & valid
        else:
            keep = valid

        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, k_len), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
            if key_valid.shape[-1] != k_len:
                key_valid = key_valid[..., :k_len]

        k_blk = k.reshape(bsz, nheads, k_blocks, block_size, d)
        v_blk = v.reshape(bsz, nheads, k_blocks, block_size, d)
        key_valid_blk = key_valid.reshape(bsz, k_blocks, block_size)

        flat_sel = sel_idx.reshape(-1)

        k_sel = torch.index_select(k_blk, dim=2, index=flat_sel).reshape(bsz, nheads, q_blocks, sel_len, block_size, d)
        v_sel = torch.index_select(v_blk, dim=2, index=flat_sel).reshape(bsz, nheads, q_blocks, sel_len, block_size, d)

        kv_sel = torch.index_select(key_valid_blk, dim=1, index=flat_sel).reshape(bsz, q_blocks, sel_len, block_size)
        kv_sel = kv_sel[:, None, :, :, :].expand(bsz, nheads, q_blocks, sel_len, block_size)

        keep_tok = keep[:, :, None].expand(q_blocks, sel_len, block_size).reshape(q_blocks, sel_len * block_size)
        keep_tok = keep_tok[None, None, :, :].expand(bsz, nheads, q_blocks, sel_len * block_size)

        kv_flat = kv_sel.reshape(bsz, nheads, q_blocks, sel_len * block_size) * keep_tok.to(dtype=torch.float32)

        q_blk = q.reshape(bsz, nheads, q_blocks, block_size, d)

        q2 = q_blk.to(torch.float32).reshape(bsz * nheads * q_blocks, block_size, d)
        k2 = k_sel.to(torch.float32).reshape(bsz * nheads * q_blocks, sel_len * block_size, d)
        v2 = v_sel.to(torch.float32).reshape(bsz * nheads * q_blocks, sel_len * block_size, d)

        scores = torch.matmul(q2, k2.transpose(1, 2)) * self.scaling
        mask = (1.0 - kv_flat).reshape(bsz * nheads * q_blocks, 1, sel_len * block_size) * -1e9
        scores = scores + mask

        same = sel_idx.eq(q_block_ids[:, None])
        if same.any():
            scores6 = scores.reshape(bsz, nheads, q_blocks, block_size, sel_len, block_size)
            intra = torch.triu(
                torch.full((block_size, block_size), -1e9, device=q.device, dtype=torch.float32), diagonal=1
            )
            same_m = same[None, None, :, None, :, None].to(dtype=torch.float32)
            scores6 = scores6 + intra[None, None, None, :, None, :] * same_m
            scores = scores6.reshape(bsz * nheads * q_blocks, block_size, sel_len * block_size)

        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0.0:
            probs = F.dropout(probs, p=self.attention_dropout, training=True)

        out = torch.matmul(probs, v2).to(dtype=q.dtype)
        out = out.reshape(bsz, nheads, q_blocks, block_size, d).reshape(bsz, nheads, q_len, d)
        return out

    def _local_global_gather_sparse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
        past_len: int,
        block_size: int,
        global_num_blocks: int,
        global_stride: int,
        window_tokens: int,
    ) -> torch.Tensor:
        bsz, nheads, q_len, d = q.shape
        _, _, k_len, _ = k.shape

        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, k_len), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
            if key_valid.shape[-1] != k_len:
                key_valid = key_valid[..., :k_len]

        window = int(window_tokens) if window_tokens and window_tokens > 0 else k_len
        t0 = past_len
        q_pos = t0 + torch.arange(q_len, device=q.device, dtype=torch.long)

        k_indices_all = []
        for i in range(q_len):
            t = int(q_pos[i].item())
            local_start = max(0, t - window + 1)
            local_idx = torch.arange(local_start, t + 1, device=q.device, dtype=torch.long)

            if global_num_blocks > 0:
                g_blocks = torch.arange(global_num_blocks, device=q.device, dtype=torch.long) * max(int(global_stride), 1)
                g_blocks = g_blocks.clamp(min=0, max=max((k_len - 1) // max(block_size, 1), 0))
                g_tok = []
                for b in g_blocks.tolist():
                    s = b * block_size
                    e = min((b + 1) * block_size, t + 1, k_len)
                    if s < e:
                        g_tok.append(torch.arange(s, e, device=q.device, dtype=torch.long))
                global_idx = torch.cat(g_tok, dim=0) if len(g_tok) else torch.empty((0,), device=q.device, dtype=torch.long)
            else:
                global_idx = torch.empty((0,), device=q.device, dtype=torch.long)

            idx = torch.cat([global_idx, local_idx], dim=0)
            idx, _ = torch.sort(idx)
            idx = torch.unique_consecutive(idx)
            k_indices_all.append(idx)

        max_k = max(int(x.numel()) for x in k_indices_all) if k_indices_all else 0
        if max_k == 0:
            max_k = 1

        idx_mat = torch.full((q_len, max_k), 0, device=q.device, dtype=torch.long)
        valid_mat = torch.zeros((q_len, max_k), device=q.device, dtype=torch.float32)

        for i, idx in enumerate(k_indices_all):
            n = int(idx.numel())
            idx_mat[i, :n] = idx
            valid_mat[i, :n] = 1.0

        k_sel = k.index_select(dim=2, index=idx_mat.reshape(-1)).reshape(bsz, nheads, q_len, max_k, d)
        v_sel = v.index_select(dim=2, index=idx_mat.reshape(-1)).reshape(bsz, nheads, q_len, max_k, d)

        kv = key_valid.index_select(dim=1, index=idx_mat.reshape(-1)).reshape(bsz, q_len, max_k)
        kv = kv[:, None, :, :].expand(bsz, nheads, q_len, max_k)
        kv = kv * valid_mat[None, None, :, :]

        qf = q.to(torch.float32)
        kf = k_sel.to(torch.float32)
        vf = v_sel.to(torch.float32)

        scores = torch.einsum("bhqd,bhqkd->bhqk", qf, kf) * self.scaling
        scores = scores + (1.0 - kv) * -1e9

        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0.0:
            probs = F.dropout(probs, p=self.attention_dropout, training=True)

        out = torch.einsum("bhqk,bhqkd->bhqd", probs, vf).to(dtype=q.dtype)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        bsz, q_len, _ = hidden_states.size()
        layer_type = self._get_layer_type()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_key_values is not None:
            past_len = int(past_key_values.get_seq_length())
            k, v = past_key_values.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        use_sparse = bool(getattr(self.config, "use_sparse_attention", False))
        sparse_impl = str(getattr(self.config, "sparse_attention_impl", "local_global_block"))

        if use_sparse and layer_type == "sliding_attention" and sparse_impl == "local_global_block":
            block_size = int(getattr(self.config, "sparse_block_size", 64))
            local_num_blocks = int(getattr(self.config, "sparse_local_num_blocks", 4))
            global_num_blocks = int(getattr(self.config, "sparse_global_num_blocks", 2))
            global_stride = int(getattr(self.config, "sparse_global_block_stride", 4))
            window_tokens = int(getattr(self.config, "sparse_attention_window", 0) or 0)

            attn_out = self._local_global_block_sparse(
                q=q,
                k=k,
                v=v,
                attention_mask_2d=attention_mask_2d,
                past_len=past_len,
                block_size=block_size,
                local_num_blocks=local_num_blocks,
                global_num_blocks=global_num_blocks,
                global_stride=global_stride,
                window_tokens=window_tokens,
            )
        else:
            backend = self._get_backend()
            attn_mask = attention_mask_4d
            if layer_type == "sliding_attention":
                slide = self._sliding_mask_4d(q_len=q_len, k_len=k.shape[-2], past_len=past_len, device=q.device)
                attn_mask = slide if attn_mask is None else (attn_mask.to(torch.float32) + slide)
            if backend == "matmul":
                attn_out = self._dense_matmul_fp32(q, k, v, attn_mask)
            else:
                attn_out = self._dense_sdpa(q, k, v, attn_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), None


class HumanVDecoderLayer(nn.Module):
    def __init__(self, config: HumanVConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HumanVAttention(config=config, layer_idx=layer_idx)
        self.mlp = HumanVMLP(config)
        self.input_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor] = None,
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
            attention_mask_4d=attention_mask_4d,
            attention_mask_2d=attention_mask_2d,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
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

    def _init_weights(self, module):
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
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = HumanVRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HumanVRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def _normalize_attention_mask_2d(
        self, attention_mask_2d: Optional[torch.Tensor], bsz: int, total_len: int, device: torch.device
    ) -> torch.Tensor:
        if attention_mask_2d is None:
            return torch.ones((bsz, total_len), device=device, dtype=torch.float32)
        m = attention_mask_2d.to(device=device, dtype=torch.float32)
        if m.dim() != 2:
            m = m.view(bsz, -1)
        if m.shape[-1] == total_len:
            return m
        if m.shape[-1] < total_len:
            pad = torch.ones((bsz, total_len - m.shape[-1]), device=device, dtype=torch.float32)
            return torch.cat([pad, m], dim=-1)
        return m[:, -total_len:]

    def _build_attention_mask_4d(self, attention_mask_2d_full: torch.Tensor, q_len: int, past_len: int) -> torch.Tensor:
        bsz, k_len = attention_mask_2d_full.shape
        device = attention_mask_2d_full.device
        causal = torch.triu(
            torch.full((q_len, k_len), -1e9, device=device, dtype=torch.float32),
            diagonal=1 + past_len,
        )
        causal = causal[None, None, :, :]
        expanded = attention_mask_2d_full[:, None, None, :].to(dtype=torch.float32)
        inverted = (1.0 - expanded) * -1e9
        return causal + inverted

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        bsz, seq_len = inputs_embeds.shape[:2]
        past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
        total_len = past_len + seq_len

        max_pos = int(getattr(self.config, "max_position_embeddings", 0) or 0)
        if max_pos > 0 and total_len > max_pos:
            raise ValueError(f"Sequence length {total_len} exceeds max_position_embeddings={max_pos}.")

        position_ids = torch.arange(past_len, total_len, dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids[:, -seq_len:])
        position_embeddings = (cos, sin)

        attn_2d_full = self._normalize_attention_mask_2d(attention_mask, bsz, total_len, inputs_embeds.device)
        attn_4d = self._build_attention_mask_4d(attn_2d_full, q_len=seq_len, past_len=past_len)

        hidden_states = inputs_embeds

        use_sparse = bool(getattr(self.config, "use_sparse_attention", False))
        block_size = int(getattr(self.config, "sparse_block_size", 64))
        chunk_blocks = int(getattr(self.config, "sparse_prefill_chunk_blocks", 0) or 0)
        chunk_tokens = chunk_blocks * block_size if chunk_blocks > 0 else 0

        if use_cache and use_sparse and chunk_tokens > 0 and seq_len > chunk_tokens:
            outs = []
            start = 0
            while start < seq_len:
                end = min(seq_len, start + chunk_tokens)
                hs = inputs_embeds[:, start:end, :]

                past_len_i = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
                total_len_i = past_len_i + (end - start)

                pos_i = torch.arange(past_len_i, total_len_i, dtype=torch.long, device=hs.device)
                pos_i = pos_i.unsqueeze(0).expand(bsz, -1)
                cos_i, sin_i = self.rotary_emb(hs, pos_i[:, -hs.shape[1] :])
                pos_emb_i = (cos_i, sin_i)

                attn_2d_i = attn_2d_full[:, :total_len_i]
                attn_4d_i = self._build_attention_mask_4d(attn_2d_i, q_len=hs.shape[1], past_len=past_len_i)

                for layer in self.layers:
                    if self.gradient_checkpointing and self.training:
                        hs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hs,
                            attn_4d_i,
                            attn_2d_i,
                            pos_emb_i,
                            past_key_values,
                            use_cache,
                        )
                    else:
                        hs = layer(
                            hs,
                            attention_mask_4d=attn_4d_i,
                            attention_mask_2d=attn_2d_i,
                            position_embeddings=pos_emb_i,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                        )

                outs.append(hs)
                start = end

            hidden_states = torch.cat(outs, dim=1)
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_4d,
                    attn_2d_full,
                    position_embeddings,
                    past_key_values,
                    use_cache,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask_4d=attn_4d,
                    attention_mask_2d=attn_2d_full,
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
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

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
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


__all__ = ["HumanVForCausalLM", "HumanVModel", "HumanVPreTrainedModel"]
