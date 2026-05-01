# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...integrations import use_experts_implementation
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..glm.modeling_glm import rotate_half
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaMLP, LlamaModel
from ..mixtral.modeling_mixtral import MixtralExperts, MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from .configuration_deepseek_v4 import DeepseekV4Config


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)





class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RotaryEmbedding(LagunaRotaryEmbedding):
    """
    Multi-layer-type rotary embedding (Laguna pattern: partial rotary on top of
    Gemma3's per-layer-type buffers), specialised for V4's *interleaved* RoPE.
    Interleaved RoPE: one `θ_i` per pair (`rope_head_dim // 2` entries),
    DIFF no end-to-end duplication. Same shape as `inv_freq @ position_ids`.
    """

    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    r"""Cache layer for HCA blocks (paper §2.3.2). Holds the long-range compressor's
    buffer / pool / count on top of the sliding-window K=V branch. HCA uses
    *non-overlapping* windows, so there is *no* overlap state, and HCA has *no*
    indexer either.

    Fields on top of :class:`DynamicSlidingWindowLayer`:

      * `compressor_pool` — the running list of compressed KV entries emitted so
        far (one every `compress_rate_hca` source tokens; the long-range KVs the
        attention concatenates onto its sliding-window keys / values).
      * `compressor_buffer_kv` / `compressor_buffer_gate` — source tokens that
        arrived between two full windows; once the buffer hits `compress_rate_hca`
        tokens the compressor closes a window, emits one pooled entry, and drains
        the buffer.
      * `compressor_pool_count` — number of compressed entries emitted so far,
        so `compressor_pool_count * compress_rate_hca` is the absolute position
        of the *next* window's first source token.

    The reason we store `compressor_pool_count` is to account for prefill -> decode -> prefill
    cases, where it is not obvious from the current `position_ids` alone how many compressed entries
    have been emitted so far, and thus what the absolute position of the next window is.
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.compressor_buffer_kv: torch.Tensor | None = None
        self.compressor_buffer_gate: torch.Tensor | None = None
        self.compressor_states: torch.Tensor | None = None
        self.compressor_pool_count = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """
        Shared sliding-window K=V update body. V4 uses shared-KV MQA, so `keys` and
        `values` point to the same storage on every layer.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""
        Stores the new projected `(kv, gate)` (paper §2.3.2 eqs. 20–21:
        `C = H·W^{KV}`, `Z = H·W^Z`) with the buffered kv and gate and
        return the longest window-aligned chunk that's ready to pool, plus the
        absolute source-token position of that chunk's first window. The returned
        chunk is softmax-pooled by the compressor with `position_bias` to emit one
        compressed entry per window of `compress_rate_hca` tokens.
        """
        first_pool_position = self.compressor_pool_count * self.compress_rate
        if self.compressor_buffer_kv is not None and self.compressor_buffer_kv.shape[1]:
            kv = torch.cat([self.compressor_buffer_kv, kv], dim=1)
            gate = torch.cat([self.compressor_buffer_gate, gate], dim=1)

        # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer for next time
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv, chunk_gate = kv[:, :usable], gate[:, :usable]
        self.compressor_buffer_kv, self.compressor_buffer_gate = kv[:, usable:], gate[:, usable:]
        return chunk_kv, chunk_gate, first_pool_position

    def update_compressor_states(self, compressed: torch.Tensor) -> torch.Tensor:
        r"""
        Append freshly emitted compressed entries to `compressor_states`
        (`C^{Comp}`, paper §2.3.2 eq. 23) and return the full pool. Bumps
        `compressor_pool_count` so the next `update_compressor` call knows the
        absolute source-token position of its first window.
        """
        if self.compressor_states is None:
            self.compressor_states = compressed.new_zeros((compressed.shape[0], 0, compressed.shape[-1]))
            self.compressor_pool_count += compressed.shape[1]
        else:
            if compressed.shape[1] > 0:
                self.compressor_states = torch.cat([self.compressor_states, compressed], dim=1)
        return self.compressor_states


class DeepseekV4CSACache(DeepseekV4HCACache):
    r"""Cache layer for CSA blocks (paper §2.3.1). Holds two parallel sets of
    buffer / pool / count / overlap state on top of the sliding-window K=V branch:

      * *compressor* — the main-branch `head_dim` pool (the long-range KVs
        the attention concatenates after top-k indexer selection).
      * *indexer* — the Lightning Indexer's smaller `index_head_dim` pool
        (the keys `K^{IComp}` that queries score against to pick the top-k blocks).
        Kept separate from the compressor pool because the head dim
        differs.

    Even if there is an overlap, because Ci uses the index of Cb and Ci-1 the indexes of Ca, the compression
    becomes 1/m, but we have to keep the overlap buffers.

    Both compressor and indexer use *overlapping* windows of stride `compress_rate_csa` and width
    `2 * compress_rate_csa`, thus we keep 2 buffers of KV and gate, holding the last full window's
    projected `(kv, gate)` so the next forward call's first window can stitch in
    its low-channel slice as the prior contribution.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        # Compressor overlap
        self.compressor_overlap_kv: torch.Tensor | None = None
        self.compressor_overlap_gate: torch.Tensor | None = None
        # Indexer side (parallel state at `index_head_dim`)
        self.indexer_buffer_kv: torch.Tensor | None = None
        self.indexer_buffer_gate: torch.Tensor | None = None
        self.indexer_pool: torch.Tensor | None = None
        self.indexer_pool_count = 0

        # indexer overlap
        self.indexer_overlap_kv: torch.Tensor | None = None
        self.indexer_overlap_gate: torch.Tensor | None = None

    def get_compressor_overlap(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.compressor_overlap_kv, self.compressor_overlap_gate

    def set_compressor_overlap(self, kv: torch.Tensor, gate: torch.Tensor) -> None:
        self.compressor_overlap_kv = kv
        self.compressor_overlap_gate = gate

    def update_indexer(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_pool_position = self.indexer_pool_count * self.compress_rate
        chunk_kv, chunk_gate, self.indexer_buffer_kv, self.indexer_buffer_gate = _update_window_buffer(
            self.indexer_buffer_kv, self.indexer_buffer_gate, kv, gate, self.compress_rate
        )
        return chunk_kv, chunk_gate, first_pool_position

    def update_indexer_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        self.indexer_pool = _append_to_pool(self.indexer_pool, new_pooled)
        self.indexer_pool_count += new_pooled.shape[1]
        return self.indexer_pool

    def get_indexer_overlap(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.indexer_overlap_kv, self.indexer_overlap_gate

    def set_indexer_overlap(self, kv: torch.Tensor, gate: torch.Tensor) -> None:
        self.indexer_overlap_kv = kv
        self.indexer_overlap_gate = gate


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by the grouped output projection
    The core attention's stacked output is `num_attention_heads* head_dim`-dim,
    which is *very* large (V4-Flash: 32768; V4-Pro: 65536). A direct
    `num_attention_heads*head_dim → hidden_size` projection would dominate the per-token cost.

    The paper sidesteps that by splitting the heads into `g` groups, projecting
    each `num_attention_heads * head_dim/g`-dim group independently to a `d_g`-dim intermediate output
    (with `d_g < num_attention_heads * head_dim/g`), and then mixing the resulting `g·d_g` vector to
    `hidden_size` through a single follow-up linear (`self_attn.o_b_proj`). This
    module owns the per-group block (`self_attn.o_a_proj`).

    For V4-Flash (num_attention_heads=64, head_dim=512, o_groups=8, o_lora_rank=1024,
    hidden_size=4096), g=8 groups of 4096-dim each are projected to 1024-dim, then
    mixed to 4096-dim; for V4-Pro (num_attention_heads=128, head_dim=512, o_groups=16,
    o_lora_rank=1024, hidden_size=7168), g=16 groups of 4096-dim each are projected
    to 1024-dim, then mixed to 7168-dim.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        d_in = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, d_in).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, d_in).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


def _overlap_pool(
    chunk_kv: torch.Tensor,
    chunk_gate: torch.Tensor,
    prior_kv: torch.Tensor | None,
    prior_gate: torch.Tensor | None,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    TODO this IS what I don't fully understand, probably notations
    Expand `[B, n_win, ratio, 2*head_dim]` chunks into `[B, n_win, 2*ratio, head_dim]`
    by stitching each window's *low-channel* slice onto the *high-channel* slice of the
    prior window.

    Each pooled output thus mixes `ratio` *current* source tokens (high half of the
    learned 2d split) with `ratio` *previous* source tokens (low half), so windows
    have width `2*ratio` but stride `ratio` (paper §2.3.1). For window 0, the prior
    half is filled with zero (kv) / `-inf` (gate, so its softmax weight is exactly 0),
    unless `prior_kv` / `prior_gate` carry the last full window from a previous
    forward call — in which case its low-channel slice slots into row `[0, :ratio]`.

    """
    batch, n_windows, ratio, _ = chunk_kv.shape
    new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, head_dim))
    new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, head_dim), float("-inf"))
    new_kv[:, :, ratio:] = chunk_kv[..., head_dim:]
    new_gate[:, :, ratio:] = chunk_gate[..., head_dim:]

    if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :head_dim]
    if prior_kv is not None and prior_gate is not None:
        # if prior KV is always stored in new_kv[:, 0, :ratio], then we can directly read from it instead of having separate state variables for the prior window
        new_kv[:, 0, :ratio] = prior_kv[..., :head_dim].to(new_kv.dtype)
        new_gate[:, 0, :ratio] = prior_gate[..., :head_dim].to(new_gate.dtype)
    return new_kv, new_gate


def _rope_pool(pooled: torch.Tensor, rotary_emb: nn.Module, positions: torch.Tensor, layer_type: str) -> torch.Tensor:
    """Apply RoPE to the trailing rope slice of each pooled entry at its deterministic
    absolute position. Used by both the indexer pool and the HCA / CSA compressor pools."""
    cos, sin = rotary_emb(pooled, position_ids=positions, layer_type=layer_type)
    return apply_rotary_pos_emb(pooled.unsqueeze(1), cos, sin).squeeze(1)



class DeepseekV4HCACompressor(nn.Module):
    """
    Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
    every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
    entry.

    Each closed window of m' tokens produces one pooled entry:
    `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
    `rope_head_dim` slice is applied at the deterministic absolute position
    `i * compress_rate_hca + first_pool_position` so cross-call concatenation stays
    causality-correct. Returns the current pool `[B, 1, T, head_dim]`. TODO only? not all?

    When `past_key_values is None` runs in stateless single-shot mode: compress every complete
    window from `hidden_states` and discard the remainder (instead of caching it)
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)   # Wkv
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False) # Wz
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        _q_residual: torch.Tensor,
        _position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, _, _ = hidden_states.shape
        cache_layer:DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.update_compressor(kv, gate)

        if chunk_kv.shape[1] > 0: # there were at least self.compress_rate tokens
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            compressed= self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_pool_position).unsqueeze(0).expand(batch, -1)

            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states(compressed)
        return compressed.unsqueeze(1)


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-k compressed KV blocks per query.

    Each query will only attend to these top-k (`k=?`) thus reduction attention
    complexity by a factor of ???

    The indexer
    runs its own scaled-down compressor at `index_head_dim` over the same windows
    as the outer CSA compressor, then scores queries against the pooled keys with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and keeps the top `index_topk`
    indices.

    Class-attribute `rope_layer_type` selects which inv_freq buffer of the shared
    :class:`DeepseekV4RotaryEmbedding` to use; the indexer always reads
    `"compress"` (paired with `compress_rope_theta`).

    The indexer has its own rotary because it applies RoPE to two sets of tensors:

      * *pool keys* at deterministic positions `i * compress_rate + first_pool_position`,
      * *queries* at the model's current `position_ids` (variable per forward).

    Both must use the same theta as the outer compressor (`compress_rope_theta`) so
    query/key inner products are translation-invariant in the standard rope sense — if
    they used different thetas the score `q · k` would carry a residual position-
    dependent skew. We can't precompute cos/sin once at init because the query
    positions vary per call, so the indexer owns a rotary embedding and calls it with
    `layer_type=self.rope_layer_type` twice per forward (once for pool keys, once for queries).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.n_heads**-0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None

        # --- Pool side: same overlapping windows as the outer CSA compressor, at index_head_dim ---
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
            prior_kv, prior_gate = None, None
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.update_indexer(kv, gate)
            prior_kv, prior_gate = cache_layer.get_indexer_overlap()
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            if cache_layer is not None:
                cache_layer.set_indexer_overlap(chunk_kv[:, -1].clone(), chunk_gate[:, -1].clone())
            chunk_kv, chunk_gate = _overlap_pool(chunk_kv, chunk_gate, prior_kv, prior_gate, self.head_dim)
            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            new_pooled = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            new_pooled = _rope_pool(new_pooled, self.rotary_emb, positions, self.rope_layer_type)
        else:
            new_pooled = chunk_kv.new_zeros((batch, 0, self.head_dim))
        pooled_kv = new_pooled if cache_layer is None else cache_layer.update_indexer_pool(new_pooled)


        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        # --- Score: ReLU(q·kᵀ) * weights, then top-k ---
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        topk = min(self.index_topk, pooled_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices



class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses every
    `compress_rate_csa` (m=4) source tokens and runs a
    Lightning Indexer on top of the compressed KV that scores queries with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
    entries per query before they reach core attention.

    * `kv_proj` / `gate_proj` / `position_bias` project to *2 × head_dim*. TODO spend some time
    explaining, 2 different series of compressions, mixed.

    * The cache layer's `compressor_overlap_*` state carries the last full
    window across forward calls.

    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # `2 * head_dim` because windows overlap: each compressed entry is a softmax-gated
        # convex combination of `compress_rate_csa` *current* tokens
        # mixed with `compress_rate_csa` *previous* tokens. This allows to compute 2 different series
        # of KV entries, that are then combined.
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        prior_kv, prior_gate = None, None
        if cache_layer is None: # only valid chunks can be used as cache, thus we slice the proj
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_pool_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_pool_position = cache_layer.store_compression_weights(kv, gate)
            prior_kv, prior_gate = cache_layer.get_compressor_overlap()

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            if cache_layer is not None:
                # Persist the *raw* last full window (gate already biased) so the next
                # forward call's first window can read its low-channel slice as prior.
                cache_layer.set_compressor_overlap(chunk_kv[:, -1].clone(), chunk_gate[:, -1].clone())
            chunk_kv, chunk_gate = _overlap_pool(chunk_kv, chunk_gate, prior_kv, prior_gate, self.head_dim)
            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            new_pooled = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            new_pooled = _rope_pool(new_pooled, self.rotary_emb, positions, self.rope_layer_type)
        else:
            new_pooled = chunk_kv.new_zeros((batch, 0, self.head_dim))
        pooled = (
            new_pooled.unsqueeze(1)
            if cache_layer is None
            else cache_layer.update_compressor_pool(new_pooled).unsqueeze(1)
        )
        # Lightning Indexer: gather top-`index_topk` pool entries per query.
        topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
        expanded = pooled.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        return torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    r"""
    Diff with classic attentions:
    * Shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects
      directly to that single KV head and the same tensor is read as both key and
      value.
    * Partial RoPE on the first `rope_head_dim` of each head ("Partial Rotary
      Positional Embedding"). RoPE is also applied with position `-i` to the
      attention output's rope slice, so the contribution of each KV entry stays a
      function of the *relative* distance to the query.
    * Per-head learnable attention sink like gpt OSS.
    * Grouped low-rank output projection for perfs.
    * 3 different cache mechanisms, sliding, sliding+CSA, sliding+HCA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_head_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        compressor_cls = COMPRESSOR_CLASSES[self.layer_type]
        self.compressor = compressor_cls(config) if compressor_cls is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        q_residual = self.q_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_head_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            compressed_kv = self.compressor(hidden_states, q_residual, position_ids, past_key_values, self.layer_idx)
            kv = torch.cat([kv, compressed_kv], dim=2)

        if attention_mask is not None and kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # Since K==V, V at this point had rope applied to it! By transposing we aim to only
        # apply rope on the partial rotary dim corresponding to V.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, -1).view(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (`fn`, `base`, `scale`)
    parameters that turn the incoming `hc_mult` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — `B` = batch, `S` = seq, `H` = hc_mult, `D` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        × scale[0]                × scale[1]                                × scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        σ() + eps                 σ() + eps                                 σ() + eps
                        │                         │                                         │
                        pre                        post                                     Sinkhorn(iters)
                        (stream collapse weights)  (block-output placement)                 row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        project it onto the manifold of doubly stochastic matrices M.
        This is achieved by the Sinkhorn-Knopp algorithm, which first applies an exponential function
        ˜
        to
        𝐵𝑙 to ensure positivity, getting 𝑀(0) = exp(˜
        𝐵𝑙), and then iteratively performs column and row
        normalization:
        𝑀(𝑡) = T𝑟(T𝑐(𝑀(𝑡−1))), (8)
        where T𝑟 and T𝑐 denote row and column normalization, respectively.
        """
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc)
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            # Collapse the `hc_mult` parallel streams down to a single sequence using
            # the `pre` weights (Manifold-Constrained input projection): one weighted
            # sum across the stream axis, ready for the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by `DeepseekV4Model` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP(LlamaMLP):
    pass


@use_experts_implementation
class DeepseekV4Experts(MixtralExperts):
    # GPT OSS style, no bias

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.limit = config.swiglu_limit

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate, up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            current = self.act_fn(gate) * up
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(MixtralTopKRouter):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    r"""
    Hash routing for the first `mlp_layer_types == "hash_moe"` MoE layers (paper
    §2.1). Expert selection is determined by a fixed `tid2eid[input_ids]` lookup —
    a frozen token-id → expert-id table — instead of a learned argmax. The learned
    gate `weight` still produces the per-expert scores that weight the selected
    experts' activations; only the *which-experts* selection is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

    The residual is a stack of `hc_mult` parallel streams kept in shape
    `[B, S, hc_mult, D]` throughout the block, mixed in and out via two
    :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
    Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
    the residual transform to the manifold of doubly-stochastic matrices via the
    Sinkhorn-Knopp projection — making signal propagation non-expansive across
    deep stacks.

    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden].
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        dtype = hidden_states.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    # V4 ships eager-only: the compressor / indexer paths weren't validated against
    # SDPA / FlashAttention / FlexAttention kernels — leaving these `False` makes
    # `set_attn_implementation` reject those backends instead of silently routing
    # through them.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressor's rolling-window buffer / pool / overlap state lives on the
    # per-layer cache (:class:`DeepseekV4HCACache` / :class:`DeepseekV4CSACache`)
    # and isn't compatible with :class:`StaticCache` — that path would hand the
    # compressor a :class:`StaticSlidingWindowLayer` with no `update_compressor`
    # method. Disabling fullgraph compile keeps generation tests on the dynamic
    # cache build that does dispatch to V4's own cache layers.
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["attn_hc", "ffn_hc"]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    # ``_is_stateful`` opts out of generation modes that need to roll the cache
    # back across drafts (assisted generation, prompt lookup, contrastive search).
    # The compressor's running-window state isn't rewindable, so `generate`
    # raises a clear error early instead of failing deep in the compressor with
    # a missing-method `AttributeError`.
    _is_stateful = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV4Model(LlamaModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        return_cache = past_key_values if use_cache else None
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length()
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main")

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=return_cache)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    pass


__all__ = [
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
