from __future__ import annotations

from typing import Optional

import torch
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


class HumanVLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class HumanVRotaryEmbedding(nn.Module):
    def __init__(self, config: HumanVConfig, device=None):
        super().__init__()
        self.head_dim = int(config.head_dim)
        rope_params = getattr(config, "rope_parameters", None)
        base = 10000.0
        if isinstance(rope_params, dict):
            base = float(rope_params.get("rope_theta", base))
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].to(device=x.device, dtype=torch.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, n_kv, slen, hd = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, n_kv, n_rep, slen, hd)
    return x.reshape(bsz, n_kv * n_rep, slen, hd)


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
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        attn_bias = bool(getattr(config, "attention_bias", False))

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=attn_bias)

        self.use_sparse_attention = bool(getattr(config, "use_sparse_attention", False))
        self.sparse_attention_impl = str(getattr(config, "sparse_attention_impl", "local_global_block"))
        self.sparse_block_size = int(getattr(config, "sparse_block_size", 64))
        self.sparse_prefill_chunk_blocks = int(getattr(config, "sparse_prefill_chunk_blocks", 0) or 0)
        self.sparse_local_num_blocks = int(getattr(config, "sparse_local_num_blocks", 4))
        self.sparse_global_num_blocks = int(getattr(config, "sparse_global_num_blocks", 2))
        self.sparse_global_block_stride = int(getattr(config, "sparse_global_block_stride", 4))
        self.sparse_attention_window = int(getattr(config, "sparse_attention_window", 0) or 0)

        self.rope_partial_rotary_factor = float(getattr(config, "rope_partial_rotary_factor", 1.0))
        self.kv_cache_dtype = str(getattr(config, "kv_cache_dtype", "auto"))

        self._tri_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _kv_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_cache_dtype == "auto":
            return x
        if self.kv_cache_dtype == "bf16":
            return x.to(torch.bfloat16)
        if self.kv_cache_dtype == "fp16":
            return x.to(torch.float16)
        if self.kv_cache_dtype == "fp32":
            return x.to(torch.float32)
        return x

    def _get_intra_causal(self, b: int, device: torch.device) -> torch.Tensor:
        key = (b, device)
        m = self._tri_cache.get(key)
        if m is not None:
            return m
        m = torch.triu(torch.ones((b, b), device=device, dtype=torch.bool), diagonal=1)
        self._tri_cache[key] = m
        return m

    def _select_key_blocks(self, q_block_idx: int, n_blocks: int) -> torch.Tensor:
        local_nb = self.sparse_local_num_blocks
        g_nb = self.sparse_global_num_blocks
        stride = max(1, self.sparse_global_block_stride)

        start = max(0, q_block_idx - local_nb + 1)
        local = list(range(start, q_block_idx + 1))

        global_candidates = list(range(0, q_block_idx + 1, stride))
        if g_nb is not None and g_nb > 0 and len(global_candidates) > g_nb:
            global_candidates = global_candidates[-g_nb:]
        if 0 not in global_candidates:
            global_candidates = [0] + global_candidates

        idx = sorted(set(local + global_candidates))
        idx = [i for i in idx if 0 <= i < n_blocks and i <= q_block_idx]
        return torch.tensor(idx, dtype=torch.long)

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
        rotary_dim = rotary_dim - (rotary_dim % 2)
        if rotary_dim <= 0:
            return q, k
        q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
        k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]
        cos1, sin1 = cos[..., :rotary_dim], sin[..., :rotary_dim]
        q1, k1 = _apply_rotary(q1, k1, cos1, sin1)
        return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32)) * self.scaling
        if attention_mask_4d is not None:
            scores = scores + attention_mask_4d.to(dtype=torch.float32)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
        out = torch.matmul(probs.to(v.dtype), v).to(dtype=q.dtype)
        return out

    def _local_global_block_sparse_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, n_heads, seqlen, d = q.shape
        b = self.sparse_block_size
        n_blocks = (seqlen + b - 1) // b
        pad_len = n_blocks * b - seqlen

        q = self._pad_to_blocks(q, pad_len)
        k = self._pad_to_blocks(k, pad_len)
        v = self._pad_to_blocks(v, pad_len)

        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, seqlen), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
        key_valid = self._pad_mask_to_blocks(key_valid, pad_len)

        q_blocks = q.view(bsz, n_heads, n_blocks, b, d)
        k_blocks = k.view(bsz, n_heads, n_blocks, b, d)
        v_blocks = v.view(bsz, n_heads, n_blocks, b, d)
        key_valid_blocks = key_valid.view(bsz, n_blocks, b)

        intra = self._get_intra_causal(b, q.device)
        chunk = self.sparse_prefill_chunk_blocks if self.sparse_prefill_chunk_blocks > 0 else n_blocks

        out_chunks = []
        for chunk_start in range(0, n_blocks, chunk):
            chunk_end = min(n_blocks, chunk_start + chunk)
            o_chunk = torch.zeros((bsz, n_heads, chunk_end - chunk_start, b, d), device=q.device, dtype=q.dtype)

            for bi in range(chunk_start, chunk_end):
                qb = q_blocks[:, :, bi]
                idx = self._select_key_blocks(bi, n_blocks).to(device=q.device)

                kb = k_blocks.index_select(dim=2, index=idx)
                vb = v_blocks.index_select(dim=2, index=idx)
                kb_flat = kb.reshape(bsz, n_heads, -1, d)
                vb_flat = vb.reshape(bsz, n_heads, -1, d)

                scores = torch.matmul(qb.to(torch.float32), kb_flat.transpose(-2, -1).to(torch.float32)) * self.scaling

                kv_mask = key_valid_blocks.index_select(dim=1, index=idx).reshape(bsz, -1)
                scores = scores + (1.0 - kv_mask[:, None, None, :]) * -1e9

                same = (idx == bi).nonzero(as_tuple=False)
                if same.numel() > 0:
                    p = int(same[0].item())
                    s = p * b
                    e = s + b
                    scores[:, :, :, s:e] = scores[:, :, :, s:e].masked_fill(intra[None, None, :, :], -1e9)

                if self.sparse_attention_window > 0:
                    max_ctx = self.sparse_attention_window
                    abs_key_pos = idx.repeat_interleave(b) * b + (torch.arange(idx.numel() * b, device=q.device) % b)
                    abs_q_pos = bi * b + torch.arange(b, device=q.device)
                    min_allowed = (abs_q_pos[:, None] - (max_ctx - 1)).clamp(min=0)
                    scores = scores.masked_fill(abs_key_pos[None, :] < min_allowed, -1e9)

                probs = torch.softmax(scores, dim=-1)
                probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
                o = torch.matmul(probs.to(vb_flat.dtype), vb_flat).to(dtype=q.dtype)
                o_chunk[:, :, bi - chunk_start] = o

            out_chunks.append(o_chunk)

        out = torch.cat(out_chunks, dim=2).reshape(bsz, n_heads, n_blocks * b, d)
        return out[:, :, :seqlen, :]

    def _local_global_block_sparse_decode_one(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, n_heads, q_len, d = q.shape
        if q_len != 1:
            raise ValueError("decode_one expects q_len == 1")

        seqlen = k.shape[-2]
        b = self.sparse_block_size
        n_blocks = (seqlen + b - 1) // b
        pad_len = n_blocks * b - seqlen

        k = self._pad_to_blocks(k, pad_len)
        v = self._pad_to_blocks(v, pad_len)

        if attention_mask_2d is None:
            key_valid = torch.ones((bsz, seqlen), device=q.device, dtype=torch.float32)
        else:
            key_valid = attention_mask_2d.to(device=q.device, dtype=torch.float32)
        key_valid = self._pad_mask_to_blocks(key_valid, pad_len)

        k_blocks = k.view(bsz, n_heads, n_blocks, b, d)
        v_blocks = v.view(bsz, n_heads, n_blocks, b, d)
        key_valid_blocks = key_valid.view(bsz, n_blocks, b)

        q_pos = seqlen - 1
        q_block = q_pos // b
        q_intra = q_pos % b

        idx = self._select_key_blocks(q_block, n_blocks).to(device=q.device)

        kb = k_blocks.index_select(dim=2, index=idx)
        vb = v_blocks.index_select(dim=2, index=idx)
        kb_flat = kb.reshape(bsz, n_heads, -1, d)
        vb_flat = vb.reshape(bsz, n_heads, -1, d)

        scores = torch.matmul(q.to(torch.float32), kb_flat.transpose(-2, -1).to(torch.float32)) * self.scaling

        kv_mask = key_valid_blocks.index_select(dim=1, index=idx).reshape(bsz, -1)
        scores = scores + (1.0 - kv_mask[:, None, None, :]) * -1e9

        same = (idx == q_block).nonzero(as_tuple=False)
        if same.numel() > 0:
            p = int(same[0].item())
            s = p * b
            e = s + b
            mask = torch.arange(b, device=q.device) > q_intra
            scores[:, :, 0, s:e] = scores[:, :, 0, s:e].masked_fill(mask[None, None, :], -1e9)

        if self.sparse_attention_window > 0:
            max_ctx = self.sparse_attention_window
            abs_key_pos = idx.repeat_interleave(b) * b + (torch.arange(idx.numel() * b, device=q.device) % b)
            min_allowed = max(0, q_pos - (max_ctx - 1))
            scores = scores.masked_fill(abs_key_pos[None, None, None, :] < min_allowed, -1e9)

        probs = torch.softmax(scores, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
        out = torch.matmul(probs.to(vb_flat.dtype), vb_flat).to(dtype=q.dtype)
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

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        k_len = k.shape[-2]
        use_sparse = (
            self.use_sparse_attention
            and self.layer_type == "sliding_attention"
            and self.sparse_attention_impl == "local_global_block"
        )

        if use_sparse:
            if q_len == k_len:
                attn_out = self._local_global_block_sparse_prefill(q, k, v, attention_mask_2d)
            elif q_len == 1:
                attn_out = self._local_global_block_sparse_decode_one(q, k, v, attention_mask_2d)
            else:
                attn_out = self._dense_attention(q, k, v, attention_mask_4d)
        else:
            attn_out = self._dense_attention(q, k, v, attention_mask_4d)

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
        Norm = HumanVRMSNorm if norm_backend == "rmsnorm" else HumanVLayerNorm

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
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([HumanVDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        norm_backend = str(getattr(config, "norm_backend", "rmsnorm")).lower()
        self.norm = (HumanVRMSNorm if norm_backend == "rmsnorm" else HumanVLayerNorm)(config.hidden_size, eps=eps)

        self.rotary_emb = HumanVRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_attention_masks(
        self,
        attention_mask_2d: torch.Tensor,
        q_len: int,
        past_len: int,
    ) -> torch.Tensor:
        bsz, src_len = attention_mask_2d.shape
        device = attention_mask_2d.device
        dtype = torch.float32
        causal = torch.triu(
            torch.full((q_len, src_len), -1e9, device=device, dtype=dtype),
            diagonal=1 + past_len,
        )
        causal = causal[None, None, :, :]
        expanded = attention_mask_2d[:, None, None, :].to(dtype=dtype)
        inverted = (1.0 - expanded) * -1e9
        return causal + inverted

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
            attention_mask_2d = torch.ones((bsz, past_len + q_len), device=inputs_embeds.device, dtype=torch.float32)
        else:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be 2D (bsz, seq)")
            attention_mask_2d = attention_mask.to(device=inputs_embeds.device, dtype=torch.float32)

        position_ids = torch.arange(past_len, past_len + q_len, device=inputs_embeds.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        attention_mask_4d = self._prepare_attention_masks(attention_mask_2d, q_len=q_len, past_len=past_len)

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
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
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
            attentions=outputs.attentions,
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