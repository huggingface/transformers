# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
    """V4 interleaved RoPE applied to the *trailing* rope slice of `x`.

    `cos` / `sin` come in half-sized (one entry per interleaved pair, from
    `DeepseekV4RotaryEmbedding`); we expand them to the full rope dim with
    `repeat_interleave`, then rotate the last `2 * cos.shape[-1]` channels of `x`
    with the standard `x*cos + rotate_half(x)*sin` formula in fp32 and leave the
    leading nope channels untouched. V4-Flash lays each head out as `[nope | rope]`,
    matching the reference's `x[..., -rd:]` indexing.
    """
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

    V4 deliberately decouples its architecture `layer_types`
    (`sliding_attention` / `compressed_sparse_attention` /
    `heavily_compressed_attention`) from its rope-type labels (`main` /
    `compress`) — the latter live as keys in `config.rope_parameters` and
    only differ in their `rope_theta` base. So this override replaces
    Laguna's `set(config.layer_types)` iteration with `rope_parameters.keys()`
    when building the per-type inv_freq buffers.
    """

    def __init__(self, config: DeepseekV4Config):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # Only the nested per-rope-type sub-dicts are real layer types — the top-level
        # `rope_type` key that ``convert_rope_params_to_dict`` may leave on
        # ``config.rope_parameters`` is a flat-shape leftover, not a layer.
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    def forward(self, x, position_ids, layer_type=None):
        # Key difference vs Laguna's forward: no `torch.cat([freqs, freqs], dim=-1)`
        # duplication. V4's interleaved RoPE pairs consecutive channels, so we only need
        # `rope_head_dim // 2` unique θ entries — the `apply_rotary_pos_emb` helper does
        # the `repeat_interleave(2)` next to the rotation math, where the link between
        # the doubled dim and `rotate_half` is local and obvious.
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
    buffer / running compressed entries / count on top of the sliding-window K=V
    branch. HCA uses *non-overlapping* windows, so there is *no* overlap state,
    and HCA has *no* indexer either.

    State is dict-keyed by entry name — HCA only uses `"compressor"`, but
    :class:`DeepseekV4CSACache` adds `"indexer"` to the same dicts so a single
    set of methods (`store_compression_weights` / `update_compressor_states`)
    serves both:

      * `compressed_kv[name]` — the running list of compressed KV entries
        emitted so far (one every `compress_rate` source tokens; the long-range
        KVs the attention concatenates onto its sliding-window keys / values).
      * `buffer_kv[name]` / `buffer_gate[name]` — source tokens that arrived
        between two full windows; once the buffer hits `compress_rate` tokens
        the compressor closes a window, emits one entry, and drains the buffer.
      * `entry_count[name]` — number of compressed entries emitted so far, so
        `entry_count[name] * compress_rate` is the absolute position of the
        *next* window's first source token. Tracked separately from
        `position_ids` so prefill -> decode -> prefill stays consistent.
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

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

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""
        Concatenate the new projected `(kv, gate)` (paper §2.3.2 eqs. 20–21:
        `C = H·W^{KV}`, `Z = H·W^Z`) for entry `name` with what's already in
        the buffer, peel off the longest window-aligned prefix (the chunk
        ready to compress), keep the leftover in the buffer for next call,
        and return `(chunk_kv, chunk_gate, first_window_position)`. The
        returned chunk is softmax-aggregated by the compressor with
        `position_bias` to emit one compressed entry per window of
        `compress_rate` tokens.
        """
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer for next time
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        r"""
        Append freshly emitted compressed entries to `compressed_kv[name]`
        (`C^{Comp}`, paper §2.3.2 eq. 23), bump `entry_count[name]`, and
        return the running `compressed_kv[name]`.
        """
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    r"""Cache layer for CSA blocks (paper §2.3.1). Extends :class:`DeepseekV4HCACache`
    by adding an `"indexer"` entry to the inherited `buffer_kv` / `buffer_gate` /
    `compressed_kv` / `entry_count` dicts, plus per-name *overlap* state for the
    two-series window scheme.

    What "overlap" means here: the CSA `kv_proj` / `gate_proj` produce `2 * head_dim`
    features per source token — two independent compressed series Ca and Cb stored
    in one tensor. Ca occupies `[..., :head_dim]`, Cb occupies `[..., head_dim:]`.
    Pooled entry `w` is the softmax-gated convex combination of window `w-1`'s Ca
    slice with window `w`'s Cb slice — effective width `2 * compress_rate_csa`,
    stride `compress_rate_csa` (paper §2.3.1).

    Because adjacent windows share state only through *the previous window's Ca
    slice*, the only thing we need to carry across a forward boundary is
    `chunk[:, -1, :, :head_dim]` (Ca) of the last full window — Cb is never read
    again. That's what `overlap_kv[name]` / `overlap_gate[name]` persist.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r"""
        Read the `name` entry's prior window's Ca slice (saved on the previous
        forward call) and persist the *current* call's last-window Ca slice for
        the next call. Only the `:head_dim` slice (Ca) is ever consumed
        downstream — Cb has already been folded into the previous window's
        emitted compressed entry — so we store half what `chunk[:, -1]` holds.
        Returns `(prior_kv, prior_gate)` — both `None` on the very first call.
        """
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


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
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class DeepseekV4HCACompressor(nn.Module):
    """
    Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
    every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
    entry.

    Each closed window of m' tokens produces one compressed entry:
    `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
    `rope_head_dim` slice is applied at the deterministic absolute position
    `i * compress_rate_hca + first_window_position` so cross-call concatenation
    stays causality-correct. Returns the running list of *all* compressed
    entries emitted so far (shape `[B, 1, T, head_dim]` with
    `T = entry_count["compressor"]`), so the attention can attend over the
    full long-range history.

    When `past_key_values is None` runs in stateless single-shot mode: compress
    every complete window from `hidden_states` and discard the remainder
    (instead of caching it).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:  # there were at least self.compress_rate tokens
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        return compressed.unsqueeze(1), None


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-`k` compressed KV blocks per query, with
    `k = config.index_topk`. Each query then attends only to those `k` of the
    `seq_len / compress_rate_csa` compressed entries — reduction factor
    `(seq_len / compress_rate_csa) / index_topk` over full attention against
    the entire compressed sequence.

    The indexer runs its own scaled-down compressor at `index_head_dim` over
    the same windows as the outer CSA compressor, then scores queries against
    the compressed keys with `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and
    keeps the top `index_topk` indices.

    The indexer has its own rotary because it applies RoPE to two sets of
    tensors:

      * *compressed keys* at deterministic positions
        `i * compress_rate + first_window_position`,
      * *queries* at the model's current `position_ids` (variable per forward).

    Both must use the same theta as the outer compressor
    (`compress_rope_theta`) so query/key inner products are
    translation-invariant — if they used different thetas, `q · k` would carry
    a residual position-dependent skew. We can't precompute cos/sin once at
    init because the query positions vary per call, so the indexer owns its
    own rotary and calls it twice per forward (once for compressed keys, once
    for queries) with `layer_type=self.rope_layer_type` (always `"compress"`).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
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
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Same Ca / Cb overlap layout as the outer CSA compressor, at index_head_dim.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)
        )

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        # ReLU(q·kᵀ) * weights, then top-k
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        T = compressed_kv.shape[1]
        topk = min(self.index_topk, T)

        # Causal pre-mask: compressed block s covers source positions
        # `[s*compress_rate, (s+1)*compress_rate)`. A query at absolute position
        # `p` may only attend to blocks whose tokens are strictly past, i.e.
        # `(s+1)*compress_rate <= p`, equivalently `s < (p+1) // compress_rate`.
        # Without this, the top-k can land on blocks containing future tokens
        # during prefill / training-style forwards. After top-k, any picks that
        # are still future-pointing (e.g. when every block is masked for an
        # early query) are replaced with a `-1` sentinel for the gather caller.
        if T > 0:
            pos = position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0).expand(batch, -1)
            causal_threshold = (pos + 1) // self.compress_rate  # [B, S]
            block_idx = torch.arange(T, device=index_scores.device)
            future_mask = block_idx.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
            topk_idxs = index_scores.topk(topk, dim=-1).indices  # [B, S, k]
            invalid = topk_idxs >= causal_threshold.unsqueeze(-1)
            return torch.where(invalid, torch.full_like(topk_idxs, -1), topk_idxs)

        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses
    every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on
    top of the compressed KV that scores queries with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
    entries per query before they reach core attention.

    `kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each
    token contributes two independent compressed series Ca and Cb stored in
    one tensor. Ca = `[..., :head_dim]` (its contribution to the *next*
    window's compressed entry), Cb = `[..., head_dim:]` (its contribution to
    the *current* window's compressed entry). Compressed entry `w` is the
    softmax-gated convex combination of window `w-1`'s Ca slice with window
    `w`'s Cb slice over `2 * compress_rate_csa` slots — width
    `2 * compress_rate_csa`, stride `compress_rate_csa`. For `w = 0` we need
    the previous window's Ca slice from the *previous forward call*; the
    cache holds it in `overlap_kv` and hands it back here. On the very first
    call (or when there is no cache) that slot stays zero-kv / `-inf`-gate,
    which gives it softmax weight 0.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
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

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb
            # (`[..., head_dim:]`) goes in the second half (current window),
            # Ca of the previous window (`[..., :head_dim]`) goes in the
            # first half. Window 0's first half stays zero-kv / -inf-gate
            # (softmax weight 0) on the very first forward call; on later
            # calls the cache fills it with the saved Ca slice.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        # Lightning Indexer: gather top-`index_topk` compressed entries per query.
        # Indexer may return `-1` sentinels for queries with fewer ready blocks than
        # `index_topk` (early prefill positions): clamp them to a safe gather index
        # and remember the validity to build a per-query block mask below.
        topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
        k = topk.shape[-1]
        valid = topk >= 0  # [B, S, k]
        safe_topk = topk.clamp(min=0)
        expanded = compressed_kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        idx = safe_topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        gathered = torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)  # [B, 1, S*k, D]

        # Per-query block bias over the flat `S*k` compressed segment: query `t` may
        # only see slots `[t*k : (t+1)*k]` (its own gathered entries) and only the
        # ones marked valid by the indexer. Everything else is `-inf`. Without this
        # the downstream right-pad with 0.0 would let every query attend to entries
        # selected for other queries, which is not equivalent to per-query CSA.
        block_bias = gathered.new_full((batch, 1, seq_len, seq_len, k), float("-inf"))
        allowed = torch.where(valid, gathered.new_zeros(()), gathered.new_full((), float("-inf")))  # [B, S, k]
        arange_s = torch.arange(seq_len, device=gathered.device)
        block_bias[:, 0, arange_s, arange_s, :] = allowed  # diagonal: q_idx == block_idx
        block_bias = block_bias.view(batch, 1, seq_len, seq_len * k)
        return gathered, block_bias


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
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )

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

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        compressed_bias = None
        if self.compressor is not None:  # Compressed KV (CSA or HCA)
            compressed_kv, compressed_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressor path concatenates extra entries onto the KV axis after the
        # standard sliding-window cache update, so a tensor `attention_mask` (built
        # for the pre-concat KV length) needs an additive bias covering them.
        # CSA returns a per-query block bias so query `t` only sees the `k`
        # compressed entries the indexer picked for `t` (anything else is `-inf`);
        # HCA returns `None` and falls back to zero padding (all compressed entries
        # visible to every query). Flex-attention gets a `BlockMask` whose KV-length
        # axis comes from its own `mask_mod`, so we skip both code paths there.
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            extra = kv.shape[2] - attention_mask.shape[-1]
            if compressed_bias is None:
                attention_mask = F.pad(attention_mask, (0, extra), value=0.0)
            else:
                attention_mask = torch.cat([attention_mask, compressed_bias.to(attention_mask.dtype)], dim=-1)

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

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (`-sin`) at the query position to undo it on the rope slice of the
        # output before the grouped output projection mixes heads. The transpose pair is
        # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
        # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
        # `[B, H, S, D]`.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
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
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
        Knopp: starting from the sigmoid-positive matrix, alternate row and
        column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
        the `hc_mult` parallel streams into a single sequence (input projection
        into the sublayer); `post` and `comb` are returned for the caller to
        apply on the sublayer output.
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
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
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

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class (like gpt-oss's _apply_gate) so the grouped_mm / batched_mm
        # backends swapped in by `@use_experts_implementation` apply the same clamp +
        # SiLU on top of their packed gate_up output instead of bypassing it.
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

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
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
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
        # `post` / `comb` come out of the HC modules in fp32 (Sinkhorn projection runs
        # in float); the .to(dtype) puts everything back to the input dtype before mixing
        # so both sites stay consistent with `hidden_states`'s entry dtype.
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    # V4 ships eager-only. The non-eager backends are off for the following reasons:
    #
    #   * FlashAttention 2 / 3 cap the head dim at 256; V4's `head_dim=512`
    #     (V4-Flash and V4-Pro both) is structurally incompatible — `flash_attention_2`
    #     and the `kernels-community/vllm-flash-attn3` kernel both fail with
    #     `RuntimeError: FlashAttention forward only supports head dimension at most
    #     256`. FA4 has the same 256 cap, so it's off too.
    #   * SDPA: torch's SDPA kernel doesn't carry the per-head learnable sink term V4
    #     inherits from gpt-oss-style attention.
    #   * FlexAttention: V4 attention concatenates compressor entries onto the KV
    #     axis *inside* the attention block, after the model-level mask was built,
    #     so the resulting KV length doesn't match the BlockMask's `kv_len`.
    #     BlockMask has no runtime resize, and rebuilding it per-block would require
    #     teaching the compressor's variable output count to a `mask_mod` — not
    #     worth it for a path the compressor already owns its own causality
    #     bookkeeping for.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressor's rolling-window buffer / compressed-entries / overlap state
    # lives on the per-layer cache (:class:`DeepseekV4HCACache` /
    # :class:`DeepseekV4CSACache`) and isn't compatible with :class:`StaticCache`
    # — that path would hand the compressor a :class:`StaticSlidingWindowLayer`
    # with no `store_compression_weights` method. Disabling fullgraph compile
    # keeps generation tests on the dynamic cache build that does dispatch to
    # V4's own cache layers.
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["attn_hc", "ffn_hc", "e_score_correction_bias"]
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
