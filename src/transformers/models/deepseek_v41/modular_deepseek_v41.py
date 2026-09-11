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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sympy import isprime
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...generation import GenerationMixin
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoTokenizer
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from .configuration_deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    """Eager shared-KV attention with the per-head learnable sink of V4.1.

    The sink joins the softmax as one extra logit column and is then dropped — i.e. it
    only grows the denominator, matching the reference kernel (where
    `sum_exp += exp(attn_sink - max)` and the output is normalized by it). Rows whose
    every slot is masked still get a finite result thanks to the sink."""
    # The shared K=V head ([B, 1, T, D]) broadcasts against the query heads in the
    # matmuls — no Hx materialization of the KV tensor.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.attn_sink.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks.float()], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # the sink only appears in the denominator
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights  # [B, S, H, D]


_FP8_MAX = 448.0  # float8_e4m3fn finite max
_FP4_MAX = 6.0  # float4_e2m1fn max
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_LUT_CACHE = {}  # device-keyed cache of the e2m1 value table


def _fp4_lut(device: torch.device) -> torch.Tensor:
    lut = _FP4_LUT_CACHE.get(device)
    if lut is None:
        lut = _FP4_TABLE.to(device)
        _FP4_LUT_CACHE[device] = lut
    return lut


def _pow2_ceil_scale(t: torch.Tensor) -> torch.Tensor:
    """`2^ceil(log2(t))` for fp32 `t > 0`, via the same IEEE-754 bit manipulation
    the reference kernel uses (exponent field minus 127, plus any nonzero
    mantissa) — the ue8m0 power-of-two scale rounding."""
    bits = t.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    k = exponent - 127 + (mantissa != 0).to(torch.int32)
    return torch.exp2(k.to(torch.float32))


def _e2m1_codes(q: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even cast of pre-clamped fp32 values onto the e2m1 grid;
    returns uint8 codes with the sign in bit 3. Ties at even-code boundaries
    (0.25, 1.25, 2.5, 5.0) round down, at odd-code boundaries round up."""
    magnitude = q.abs()
    negative = torch.signbit(q)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=q.device)
    ties_up = torch.tensor([False, True, False, True, False, True, False], device=q.device)
    thresholds = torch.where(
        ties_up, boundaries, torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
    )
    codes = (magnitude.unsqueeze(-1) >= thresholds).sum(-1).to(torch.uint8)
    return codes | (negative.to(torch.uint8) << 3)


def _fake_quant_fp8_block(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block-wise FP8 fake-quantization with ue8m0 (power-of-two) scales — the
    reference's window-KV quantizer, applied to the whole post-RoPE vector. Returns
    the quantized-then-dequantized tensor (out of place: training needs the original
    in the autograd graph; the values are identical to the reference's in-place
    call)."""
    n = x.shape[-1]
    if n % block_size:
        return x  # shapes the reference cannot block (tiny test configs) run unquantized
    blocks = x.float().view(*x.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    scale = _pow2_ceil_scale(amax * (1.0 / _FP8_MAX))
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    dequantized = quantized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return dequantized.view(*x.shape[:-1], n).to(x.dtype)


def _fake_quant_fp4_block(x: torch.Tensor, block_size: int, e4m3_scales: bool = False) -> torch.Tensor:
    """Block-wise FP4 fake-quantization — the reference's indexer (block 32, ue8m0
    scales) and compressed-KV (block 16, e4m3 scales) quantizers. Returns the tensor
    quantized onto the e2m1 grid and dequantized (out of place; values identical to
    the reference's in-place call)."""
    n = x.shape[-1]
    if n % block_size:
        return x  # shapes the reference cannot block (tiny test configs) run unquantized
    blocks = x.float().view(*x.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(-1)
    if e4m3_scales:
        scale = (amax.clamp_min(6.0 * 2.0**-9) / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        scale = _pow2_ceil_scale(amax.clamp_min(6.0 * 2.0**-126) * (1.0 / _FP4_MAX))
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX)
    codes = _e2m1_codes(quantized)
    values = _fp4_lut(x.device)[codes.long()]
    return (values * scale.unsqueeze(-1)).view(*x.shape[:-1], n).to(x.dtype)


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False):
    """Interleaved-pair RoPE on the trailing rope slice of `x`.

    DeepSeek-V4.1 pairs *adjacent* channels of the rope slice (even, odd) and rotates
    each pair as a complex number — matching the reference implementation's
    ``torch.view_as_complex(x.unflatten(-1, (-1, 2)))`` convention. `cos` / `sin`
    carry one entry per pair (``rope_dim // 2``). The leading nope channels pass
    through. ``inverse=True`` conjugates the rotation: the attention output carries
    the query's RoPE, and the inverse rotation removes it so the shared rotated cache
    stays in one form. Accepts ``[B, S, D]`` and ``[B, S, H, D]``.
    """
    if inverse:
        sin = -sin
    rope_dim = cos.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)  # [B, S, 1, rd/2]
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)


class DeepseekV41RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV41UnweightedRMSNorm(nn.Module):
    """RMS normalization without a learned weight — used on the flattened
    hyper-connection stream before the mix projection."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV41RotaryEmbedding(DeepseekV4RotaryEmbedding):
    """Same two-rope scheme as V4, reused verbatim: `main` = plain `rope_theta` for
    pure sliding-window layers, `compress` = `compress_rope_theta` + optional YaRN for
    layers with a compressed branch (one latent stands for `compress_ratio` tokens, so
    its positions are further apart — hence the larger base). The config folds the
    checkpoint's flat `rope_scaling` into `rope_parameters` the same way V4 does."""

    pass


class DeepseekV41GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear of the attention output projection.

    The stacked attention output is `num_heads * head_dim`-dim (32768 for the released
    model) — a direct projection to `hidden_size` would dominate the per-token cost.
    Instead the heads are split into `o_groups` groups, each projected independently
    to `o_lora_rank`, then mixed to `hidden_size` by `wo_b`. This module owns the
    per-group block (`wo_a`)."""

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


class DeepseekV41CSACache(DynamicSlidingWindowLayer):
    r"""Cache layer for a V4.1 **KV-source** layer (CSA2). On top of the shared-KV
    sliding-window ring every layer keeps, it holds the group state of the shared
    compressed branch:

      * `buffer_kv` / `buffer_gate` — source tokens arrived since the last complete
        compress group; once `compress_ratio` tokens accumulate the compressor closes a
        group and drains the buffer. This is what makes chunked prefill seamless:
        partial groups simply carry across forward calls.
      * `compressed_kv["compressor"]` — the running compressed KV entries (one per
        complete group, at `head_dim`), published to the whole group via the per-forward
        shared state; consumer layers hold plain sliding layers and read this one.
      * `compressed_kv["indexer"]` — the running *indexer keys* (one per complete
        group, at `index_head_dim`), derived from the same pooled latents by the
        source's indexer (`wk` + `k_norm`): the whole group scores against one key set.
      * `entry_count["compressor"]` — groups emitted so far, so
        `entry_count * compress_ratio` is the absolute position of the next group's
        first source token.

    The compress ratio is passed per call by the compressor (it is a per-layer config
    value, not a per-layer-type one, so it cannot be resolved at cache-construction
    time).
    """

    _layer_type = "shared_compressed_attention"

    def __init__(self, config: "DeepseekV41TextConfig", **kwargs):
        super().__init__(sliding_window=config.sliding_window)
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        # Only the compressor counter is read (group positions); indexer keys are
        # appended without needing a position anchor.
        self.entry_count: dict[str, int] = {"compressor": 0}

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        # The base class permutes only the sliding-window keys; the group state
        # (partial-group buffers, shared compressed KV, indexer keys) is per-batch-row
        # and must follow the beams too, or beams silently attend each other's groups.
        super().reorder_cache(beam_idx)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor.index_select(0, beam_idx.to(tensor.device))
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor.index_select(0, beam_idx.to(tensor.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor.repeat_interleave(repeats, dim=0)
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        for name, tensor in self.compressed_kv.items():
            if tensor is not None:
                self.compressed_kv[name] = tensor[indices, ...]
        for attr in ("buffer_kv", "buffer_gate"):
            buffer = getattr(self, attr)
            for name, tensor in buffer.items():
                if tensor is not None:
                    buffer[name] = tensor[indices, ...]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """Sliding-window K=V update: return everything seen so far (the attention
        mask selects the window), keep the last `sliding_window - 1` entries cached."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor | None, compress_ratio: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        r"""Concatenate the newly projected `(kv, gate)` with the buffer, peel off the
        longest group-aligned prefix, keep the remainder buffered, and return
        `(chunk_kv, chunk_gate, first_group_position)` — the absolute position of the
        first group's first source token. `gate` is `None` at ratio 1 (no pooling:
        every token is its own group)."""
        first_group_position = self.entry_count[name] * compress_ratio
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            if gate is not None:
                gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // compress_ratio) * compress_ratio
        self.buffer_kv[name] = kv[:, usable:]
        self.buffer_gate[name] = None if gate is None else gate[:, usable:]
        return kv[:, :usable], None if gate is None else gate[:, :usable], first_group_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        r"""Append freshly emitted entries to `compressed_kv[name]`, bump the group
        count, and return the running tensor."""
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[2] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=2)
        if name == "compressor":
            self.entry_count[name] += compressed.shape[2]
        return self.compressed_kv[name]


class DeepseekV41Compressor(nn.Module):
    r"""Pools `compress_ratio` consecutive tokens into one KV latent with a learned
    softmax gate (pooling in fp32). Returns the latent **before RoPE** — the indexer
    derives its keys from the unrotated form. At ratio 1 there is no pooling and no
    gate: a plain per-token projection (the CED "decoder" branch — full-resolution KV
    projected once by the source layer instead of per layer)."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.head_dim = config.head_dim
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False) if self.compress_ratio > 1 else None
        self.norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, cache_layer: DeepseekV41CSACache | None
    ) -> tuple[torch.Tensor | None, int]:
        """Returns the pre-RoPE latents of the groups completing in this call (`None`
        while a group is still filling — decode steps between group boundaries), plus
        the absolute position of the first returned group."""
        if self.wgate is None:  # ratio 1: every token is a group, no pooling
            latent = self.norm(self.wkv(hidden_states))
            first_group_position = 0 if cache_layer is None else cache_layer.entry_count["compressor"]
            return latent, first_group_position

        # Pooling math runs in fp32 (the reference stores `wkv` fp32 above ratio 1);
        # upcast the weight explicitly so a half-precision model does not crash.
        kv = nn.functional.linear(hidden_states.float(), self.wkv.weight.float())
        # The gate is computed in fp32 regardless of the storage dtype (the released
        # checkpoint stores `wgate` in BF16).
        gate = nn.functional.linear(hidden_states.float(), self.wgate.weight.float())
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            chunk_kv, chunk_gate, first_group_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_group_position = cache_layer.store_compression_weights(
                "compressor", kv, gate, self.compress_ratio
            )
        if chunk_kv.shape[1] == 0:
            return None, first_group_position
        n_groups = chunk_kv.shape[1] // self.compress_ratio
        kv = chunk_kv.view(chunk_kv.shape[0], n_groups, self.compress_ratio, -1)
        gate = chunk_gate.view(chunk_gate.shape[0], n_groups, self.compress_ratio, -1)
        latent = (kv * gate.softmax(dim=2, dtype=torch.float32)).sum(dim=2)
        return self.norm(latent.to(hidden_states.dtype)), first_group_position


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k (hierarchical sparse indexer): keep the
    `topk_blocks` highest-scoring blocks of `block_size` compressed positions per query.

    `logits` is `[..., n_positions]` with unreachable positions already at -inf, which
    makes a block score of -inf mean "not reachable yet". The block holding the query's
    newest position is only partly filled, so it is pinned in — it holds the most recent
    tokens but could otherwise be outscored by an older, full block. Returns a bool mask
    shaped like `logits`."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # `compress_lens` carries a trailing axis: [.., 1] against the [.., blocks] scores.
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device).view(-1) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # Fewer reachable blocks than topk_blocks: leftover picks come back -inf — drop them.
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class DeepseekV41Indexer(nn.Module):
    r"""Sparse indexer of a CSA2 **index-source** layer. Scores each query against the
    shared indexer keys (one per compressed group, at `index_head_dim`) and keeps the
    top `index_topk` groups per query; the resulting per-query block bias is published
    to the group's other layers ("Reuse" mode).

    Key sharing: the keys are `k_norm(wk(latent))` of the *compressor latent* — only a
    layer that also owns its compressor (`kv_source_layer_ids`) can produce them
    (`owns_k`); every later index source ("Reindex" mode) rescores with its own weights
    against the keys published by its group's source. Queries come from the attention's
    low-rank residual (`q_norm(wq_a(x))`) through `wq_b`, rotated with the compress
    rope. The candidate source layer additionally publishes the two-level-top-k
    candidate mask that constrains all later index sources."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.owns_k = layer_idx in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.heads_scaling = self.num_heads**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        latent: torch.Tensor | None,
        first_group_position: int,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV41CSACache | None,
        shared: dict,
    ) -> None:
        """Publishes `shared["topk_bias"]` (the per-query block bias over the shared
        compressed KV), and `shared["candidates"]` at the candidate source layer."""
        batch, seq_len, _ = hidden_states.shape
        ratio = self.compress_ratio

        # 1. Publish the index keys of the groups that completed in this call. The keys
        #    are derived from the PRE-rope latent, before the compressor rotates the
        #    same values into the main cache.
        if self.owns_k:
            if latent is not None:
                k = self.k_norm(self.wk(latent))
                positions = first_group_position + ratio * torch.arange(latent.shape[1], device=k.device)
                cos, sin = self.rotary_emb(
                    k, position_ids=positions.unsqueeze(0).expand(batch, -1), layer_type="compress"
                )
                k = apply_rotary_pos_emb(k, cos, sin)
                # QAT semantics: indexer keys are FP4-quantized (ue8m0 scale per 32
                # channels) before they land in the shared key cache.
                k = _fake_quant_fp4_block(k, block_size=32)
                k = k.unsqueeze(1)  # [B, 1, G, idh]
                if cache_layer is not None:
                    cache_layer.update_compressor_states("indexer", k)
                else:
                    shared["index_k"] = (
                        k if shared.get("index_k") is None else torch.cat([shared["index_k"], k], dim=2)
                    )
            # Publish the RUNNING key cache — decode steps between group boundaries emit
            # nothing new but the group still scores against everything emitted so far.
            if cache_layer is not None:
                shared["index_k"] = cache_layer.compressed_kv["indexer"]
        index_k = shared.get("index_k")
        compressed_len = 0 if index_k is None else index_k.shape[2]
        if compressed_len == 0:
            return

        # 2. Score the queries against the shared keys.
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        # QAT semantics: the indexer query is FP4-quantized too, so the top-k
        # selection matches the trained quantized scoring.
        q = _fake_quant_fp4_block(q, block_size=32)
        scores = torch.einsum("bshd,btd->bsht", q.float(), index_k[:, 0].float())
        scores = scores.relu_() * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.heads_scaling
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

        # 3. Visibility: a group becomes visible once the query passed its last token —
        # in ABSOLUTE positions, so a chunk that starts mid-sequence sees exactly the
        # groups it could see in a one-shot prefill. `compress_lens` keeps a trailing
        # axis so every broadcast below is explicit: `masked_fill` expands its result
        # to the broadcast shape, and a sloppy `[T] >= [B, S]` mask would silently
        # grow the score tensor.
        entry_indices = torch.arange(compressed_len, device=index_scores.device).view(1, 1, -1)
        compress_lens = (position_ids.long().unsqueeze(-1) + 1) // ratio  # [B, S, 1]
        index_scores = index_scores.masked_fill(entry_indices >= compress_lens, float("-inf"))

        # 4. Two-level top-k: the candidate source publishes its block mask; every
        #    later index source scores only inside it.
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                index_scores, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates and shared.get("candidates") is not None:
            index_scores = index_scores.masked_fill(~shared["candidates"], float("-inf"))

        # 5. Top-k per query. Early queries can have fewer visible groups than
        #    `index_topk`, so some picks come back with a -inf score; clamp those into
        #    the dummy slot past the end (dropped by the slice) — scattering them at
        #    their raw index would leak future groups into the attention.
        top_k = min(self.index_topk, compressed_len)
        block_bias = index_scores.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        if top_k > 0:
            topk = index_scores.topk(top_k, dim=-1, sorted=False)
            valid = topk.values > float("-inf")
            safe = torch.where(valid, topk.indices, torch.full_like(topk.indices, compressed_len))
            block_bias.scatter_(-1, safe.unsqueeze(1), 0.0)
        shared["topk_bias"] = block_bias[..., :compressed_len]


class DeepseekV41Attention(nn.Module):
    r"""Latent shared-KV attention over two KV sources: a sliding window of raw KV plus,
    when the layer has a compressed branch, the *shared* compressed KV of its group.

    - Q and the output projection are low-rank; the output projection is grouped
      (block-diagonal `wo_a` over `o_groups`, then the mixing `wo_b`).
    - K=V is a single latent (`wkv` + `kv_norm`); the attention output's rope slice is
      inverse-rotated so the shared rotated cache works.
    - Per-head learnable attention sink (`attn_sink`), like gpt-oss.
    - **KV sharing (CSA2)**: only `kv_source_layer_ids` layers own a
      :class:`DeepseekV41Compressor` — the layers in between read the source's
      compressed cache through the per-forward `shared` dict ("Reuse" mode). Index
      sources run the indexer and publish the top-k bias; the layers in between reuse
      it. `shared` is keyed per forward, but the persistent state (group buffers,
      compressed KV, indexer keys) lives on the source's cache layer, so this stays
      correct across prefill / chunked prefill / decode."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        # Shared-KV latent attention: a single KV head broadcast to all heads.
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_layer_type = "compress" if self.compress_ratio else "main"
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV41GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads))

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        # trf-ignore: TRF050 — the compress-rope cos/sin here are evaluated at the
        # LATENT positions (first_group_position + ratio*k), which are only known
        # after the compressor runs; the model-level rotary cannot precompute them in
        # `position_embeddings`. Only kv-source layers own one (shared by their group).
        self.compress_rotary = DeepseekV41RotaryEmbedding(config) if self.is_kv_source else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        shared: dict,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # position_ids flows through **kwargs (TRF043): the decoder layer passes the
        # model-level kwargs straight in, and it must stay available to the attention
        # interface below (padding-free paths read it from kwargs).
        position_ids = kwargs["position_ids"]
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)  # [B, H, S, D]

        kv = self.kv_norm(self.wkv(hidden_states))
        kv = apply_rotary_pos_emb(kv, cos, sin).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        # QAT semantics: the window KV cache stores FP8-quantized values (one ue8m0
        # scale per 32 channels, RoPE tail included) — part of the model, applied
        # even in otherwise-unquantized runs.
        kv = _fake_quant_fp8_block(kv, block_size=32)
        if past_key_values is not None:  # K == V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compress_ratio:
            cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
            latent, first_group_position = (None, 0)
            if self.is_kv_source:
                latent, first_group_position = self.compressor(hidden_states, cache_layer)

            # The indexer consumes the PRE-rope latent; it must run before the latent is
            # rotated into the main compressed cache.
            if self.is_index_source:
                self.indexer(
                    hidden_states, q_residual, latent, first_group_position, position_ids, cache_layer, shared
                )
            block_bias = shared.get("topk_bias")

            if latent is not None:
                positions = first_group_position + self.compress_ratio * torch.arange(
                    latent.shape[1], device=latent.device
                )
                cos_c, sin_c = self.compress_rotary(
                    latent, position_ids=positions.unsqueeze(0).expand(batch, -1), layer_type="compress"
                )
                rotated = apply_rotary_pos_emb(latent, cos_c, sin_c)
                # QAT semantics: the compressed KV cache stores FP4-quantized latents
                # (e2m1 grid, one e4m3 scale per 16 channels).
                rotated = _fake_quant_fp4_block(rotated, block_size=16, e4m3_scales=True)
                rotated = rotated.unsqueeze(1)  # [B, 1, G, hd]
                if cache_layer is not None:
                    cache_layer.update_compressor_states("compressor", rotated)
                else:
                    shared["compress_kv"] = (
                        rotated
                        if shared.get("compress_kv") is None
                        else torch.cat([shared["compress_kv"], rotated], dim=2)
                    )
            if self.is_kv_source and cache_layer is not None:
                # Publish the RUNNING compressed cache — a decode step between group
                # boundaries emits nothing new, but the group still attends over
                # everything emitted so far.
                shared["compress_kv"] = cache_layer.compressed_kv["compressor"]
            compressed_kv = shared.get("compress_kv")
            if compressed_kv is not None:
                kv = torch.cat([kv, compressed_kv], dim=2)

        # The compressed branch concatenated extra entries onto the KV axis after the
        # model-level mask was built: extend the mask with the indexer's per-query
        # block bias instead of zero-padding (which would attend everywhere).
        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                # A compressed branch with no index source in this forward (legal but
                # unusual schedule): mask every compressed entry off instead of
                # attending all of them.
                attention_mask = F.pad(
                    attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=float("-inf")
                )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            **kwargs,
        )

        # K == V carried RoPE on its rope slice; remove the query's rotation from the
        # output before the grouped projection mixes the heads.
        attn_output = apply_rotary_pos_emb(attn_output, cos, sin, inverse=True)
        grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)
        output = self.wo_b(self.wo_a(grouped).flatten(2))
        return output, attn_weights


class DeepseekV41TopKRouter(nn.Module):
    """MoE gate. The correction bias (`bias`) steers expert *selection* only; the
    routing weights come from the unbiased scores. Image-span tokens switch to a
    separate `bias_vl` (training `noaux_tc_for_vl`)."""

    def __init__(self, config: DeepseekV41TextConfig, n_experts: int, n_activated: int):
        super().__init__()
        self.top_k = n_activated
        self.score_fn = ACT2FN[config.scoring_func]
        self.gate_temp = config.gate_temp
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(n_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(n_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.empty(n_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
        scores = F.linear(flat, self.weight.float()) / self.gate_temp
        scores = self.score_fn(scores)
        bias = self.bias
        if image_mask is not None and image_mask.any():
            bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, self.bias)
        indices = (scores + bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            # `+1e-20` on the sum — NOT `rms_norm_eps`; matches training.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor, indices


class DeepseekV41Expert(nn.Module):
    """One SwiGLU expert (`w1` gate / `w3` up / `w2` down). The clamps come from
    training: they keep fp8/fp4 activations in range — up on both sides, gate above."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        inter = config.moe_intermediate_size
        self.w1 = nn.Linear(config.hidden_size, inter, bias=False)
        self.w2 = nn.Linear(inter, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, inter, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = self.act_fn(gate) * up
        if weight is not None:
            y = weight * y
        return self.w2(y.to(dtype))


class DeepseekV41SparseMoeBlock(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through. Eager
    dispatch loops over the experts that received tokens — spec-grade, not a serving
    path."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        # DSpark draft layers (M2) have their own expert counts; this block is only
        # ever built for backbone layers.
        self.n_experts = config.n_routed_experts
        n_activated = config.num_experts_per_tok
        self.gate = DeepseekV41TopKRouter(config, self.n_experts, n_activated)
        self.experts = nn.ModuleList([DeepseekV41Expert(config) for _ in range(self.n_experts)])
        self.shared_experts = DeepseekV41Expert(config)

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        weights, indices = self.gate(hidden_states, image_mask)
        y = torch.zeros_like(flat, dtype=torch.float32)
        # eager dispatch loop: per-expert host reads are inherent to it (spec-grade,
        # not a serving path) — the counts stay a tensor (TRF056).
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts)
        for i in range(self.n_experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](flat[idx], weights[idx, top, None]).float()
        y += self.shared_experts(flat).float()
        return y.to(hidden_states.dtype).view(shape)


class DeepseekV41EngramEmbedding(nn.Module):
    """The n-gram hash table: fp8 rows with per-row / per-32-channel E8M0 scales in the
    checkpoint, dequantized on lookup. When the model is loaded in a float dtype the
    fp8 values are represented exactly (e4m3 ⊂ bf16/fp32), so the table is a plain
    embedding and the scales are unused. ~98 GB per table in the released checkpoint —
    memory-map friendly (pure row gather)."""

    def __init__(self, num_embeddings: int, head_dim: int, block_size: int = 32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.head_dim = head_dim
        self.block_size = block_size
        self.weight = nn.Parameter(torch.empty(num_embeddings, head_dim))
        self.scale = nn.Parameter(torch.empty(num_embeddings, head_dim // block_size))

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        values = F.embedding(hash_ids, self.weight)
        if self.weight.dtype == torch.float8_e4m3fn:
            scales = F.embedding(hash_ids, self.scale).float()
            values = values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            values = values.flatten(-2)
        return values


class DeepseekV41Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc stream
    plus a shared value. The gate is a sigmoid of the signed sqrt of a normalized dot
    product between the stream and the key (weights `q_weight * k_weight`, used only as
    a product). Hash ids are computed once per forward by
    :class:`DeepseekV41NgramHashState` and passed in per engram layer."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_hash_index = config.engram_layer_ids.index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.embed = DeepseekV41EngramEmbedding(
            config.engram_num_embeddings[self.layer_hash_index], config.engram_head_dim
        )
        self.wkv = nn.Linear(
            n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))

    def forward(
        self, hidden_streams: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """hidden_streams: [B, S, hc, D]; hash_ids: [B, S, n_hash_cols]; token_mask:
        [B, S], False shuts the gate so those positions pass through untouched."""
        # The dequantized fp8 rows are exact in any dtype; cast to the consumer's.
        kv = self.wkv(self.embed(hash_ids).flatten(-2).to(self.wkv.weight.dtype))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h = hidden_streams.float()
        # Normalized per (token, hc stream) over D — NOT jointly over the streams.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        # Signed sqrt before the sigmoid, matching the training kernel.
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        out = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        return out.to(hidden_streams.dtype)


def _find_next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables: a position is hashed as
    `max_ngram_size - 1` n-grams (2..N-gram), each split over `n_heads` heads; every
    (n-gram size, head) pair owns a disjoint prime-sized bucket range — the primes are
    drawn in order above `engram_vocab_size` and never reused."""

    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: DeepseekV41TextConfig):
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.engram_max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(config.engram_n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=config.engram_max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=config.engram_n_heads,
            head_dim=config.engram_head_dim,
        )


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike
    collapse together (" The", "the", "THE" hash the same). The compressed vocab size
    must match `engram_compressed_vocab_size` — every hash multiplier derives from it,
    so a mismatch means the whole table rehashes to garbage."""
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives Strip() instead
    # of collapsing to the empty string and merging with unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with.
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, key it by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized or text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: tuple, max_ngram_size: int, compressed_vocab_size: int) -> torch.Tensor:
    """One multiplier per (layer, look-back), from a per-layer RNG so layers hash
    differently. Kept odd and bounded so `id * multiplier` cannot overflow int64."""
    multiplier_bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class DeepseekV41NgramHashState:
    """Maps each position to the hash ids of the n-grams ending there — once per
    forward, for all engram layers.

    Ids go through the compressed table, then each position is hashed with the
    `max_ngram_size - 1` tokens before it; look-back stops at the start of the sequence
    and at any DEAD token (an image-span token), so an n-gram never spans one. The
    absolute-position history buffer carries all of this across the prefill / chunked
    prefill / decode split (a request resumed mid-sequence reconstructs its look-back
    from the positions it writes, exactly like the reference's position-indexed
    cache)."""

    DEAD = -1

    def __init__(self, config: DeepseekV41TextConfig, tokenizer):
        self.layout = EngramLayout.from_config(config)
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"The tokenizer-derived compressed vocabulary size ({vocab_size}) does not match "
                f"`engram_compressed_vocab_size` ({config.engram_compressed_vocab_size}); the hash "
                "multipliers would silently rehash the whole engram table."
            )
        self.pad_id = token_map[config.engram_pad_id]
        self.max_ngram_size = self.layout.max_ngram_size
        # Primes as [L, n-gram-1, heads] (the per-step modulus is over all heads of one
        # n-gram size). Bucket offsets are PER LAYER over the flat (n-gram, head) order:
        # each layer's table is addressed from its own start, and the ranges inside it
        # are disjoint because the primes are drawn in order and never reused.
        self.primes = torch.tensor(self.layout.primes)  # [L, n-gram-1, heads]
        offsets = []
        for layer in self.layout.primes:
            flat = [p for per in layer for p in per]
            row, total = [], 0
            for p in flat:
                row.append(total)
                total += p
            offsets.append(row)
        self.offsets = torch.tensor(offsets)  # [L, n_cols]
        self.multipliers = compute_hash_multipliers(self.layout.layer_ids, self.max_ngram_size, vocab_size)
        self.token_map = torch.tensor(token_map)
        self.history: torch.Tensor | None = None

    def _to(self, device: torch.device):
        for name in ("primes", "offsets", "multipliers", "token_map", "history"):
            tensor = getattr(self, name)
            if tensor is not None and tensor.device != device:
                setattr(self, name, tensor.to(device))

    def __call__(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Returns `[B, S, n_engram_layers, n_hash_cols]` hash ids."""
        batch, seq_len = input_ids.shape
        device = input_ids.device
        self._to(device)

        max_pos = int(position_ids.max().item()) + 1
        if self.history is None or self.history.shape[0] < batch or self.history.shape[1] < max_pos:
            # Geometric growth: a decode step advances max_pos by one, so a fixed-size
            # increment would reallocate and copy O(T) on every step.
            shape = (
                max(batch, 2 * self.history.shape[0] if self.history is not None else 0),
                max(max_pos, 2 * self.history.shape[1] if self.history is not None else 0),
            )
            grown = torch.full(shape, self.DEAD, dtype=torch.long, device=device)
            if self.history is not None:
                grown[: min(self.history.shape[0], shape[0]), : self.history.shape[1]] = self.history[
                    : shape[0], : shape[1]
                ]
            self.history = grown

        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, self.DEAD))
        self.history.scatter_(1, position_ids.long(), compressed)

        positions = position_ids.long()
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = self.history.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim=-1)  # [B, S, max_ngram_size]

        # XOR the multiplied ids one look-back at a time: after step i the running value
        # is the hash of the (i+1)-gram, landing in its own prime bucket range.
        products = tokens.unsqueeze(2) * self.multipliers.to(device)  # [B, S, L, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(0)


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    r"""A V4.1 block: the residual stream is `hc_mult` parallel copies (hyper-
    connections), with the engram lookup injected at its layers before the block.

    The single-pass mHC mapping computes all three coefficient sets (pre / post / comb)
    from ONE projection of the flattened stream — but the `pre` a site computes is
    consumed by the *next* site: attention collapses with the previous site's mix and
    the FFN with the attention's. The mHC parameters are raw layer attributes
    (`hc_attn_fn` / `hc_attn_base` / `hc_attn_scale`, fp32), matching the checkpoint."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.attn = DeepseekV41Attention(config, layer_idx)
        self.ffn = DeepseekV41SparseMoeBlock(config, layer_idx)
        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # CODEPATH: DeepSeek-V4.1-Flash ships n-gram hash tables on layers 1 and 14
        # (engram_layer_ids=[1, 14]); every other checkpoint — and the tiny test
        # configs — sets engram_layer_ids=[] and takes the None side (no engram path).
        self.engram = DeepseekV41Engram(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        self.hc_input_norm = DeepseekV41UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_mixes(self, hidden_streams: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        """One projection of the normalized flattened stream → (pre, post, comb), with
        `comb` Sinkhorn-projected onto the doubly-stochastic manifold. Normalization is
        over the whole flattened hc·D stream (one statistic per token)."""
        hc = self.hc_mult
        # fp32 from the start — the reference upcasts before normalizing, so a
        # bf16/fp16 model must not round the stream before the mix projection.
        flat = self.hc_input_norm(hidden_streams.flatten(start_dim=2).float())
        mixes = F.linear(flat, fn.float())
        pre = torch.sigmoid(mixes[..., :hc] * scale[0].float() + base[:hc].float()) + self.hc_eps
        post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1].float() + base[hc : 2 * hc].float())
        comb_logits = (mixes[..., 2 * hc :] * scale[2].float() + base[2 * hc :].float()).view(
            *mixes.shape[:-1], hc, hc
        )
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    @staticmethod
    def hc_collapse(hidden_streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies into one sublayer input, weighted by `pre`."""
        return (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=2).to(hidden_streams.dtype)

    @staticmethod
    def hc_expand(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Place the sublayer output into the streams and mix the residual through
        `comb`: out_k = post_k * x + Σ_j comb[j, k] * residual_j."""
        mixed = torch.einsum("bsjk,bsjd->bskd", comb.float(), residual.float())
        out = post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed
        return out.to(residual.dtype)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        pre_mix: torch.Tensor,
        hash_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        shared: dict,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_streams: [B, S, hc, hidden]
        if self.engram is not None and hash_ids is not None:
            hidden_streams = self.engram(hidden_streams, hash_ids[:, :, self.engram.layer_hash_index, :], token_mask)

        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            hidden_streams, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        collapsed = self.hc_collapse(hidden_streams, pre_mix)
        attn_output, _ = self.attn(self.attn_norm(collapsed), shared=shared, **kwargs)
        hidden_streams = self.hc_expand(attn_output, residual, attn_post, attn_comb)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            hidden_streams, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        collapsed = self.hc_collapse(hidden_streams, attn_pre)
        ffn_output = self.ffn(self.ffn_norm(collapsed))
        hidden_streams = self.hc_expand(ffn_output, residual, ffn_post, ffn_comb)
        return hidden_streams, ffn_pre


@auto_docstring
class DeepseekV41PreTrainedModel(PreTrainedModel):
    # trf-ignore: TRF001 — deliberate: this base serves BOTH model types. The text
    # backbone chain (DeepseekV41TextModel, registered as `deepseek_v41_text`) needs
    # the flat DeepseekV41TextConfig; DeepseekV41ForCausalLM overrides with the
    # composite DeepseekV41Config because the released checkpoint's config.json is
    # composite and its top-level quantization_config must reach the quantizer.
    config_class = DeepseekV41TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV41DecoderLayer"]
    # Eager-only, same reasons as V4: FA caps head_dim at 256 (V4.1 uses 512); SDPA has
    # no per-head sink term; the compressed branch concatenates entries onto the KV axis
    # inside the block, after the model-level mask was built.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    # The compressor's group-buffer state isn't rewindable across drafts.
    _is_stateful = True
    # DSpark draft layers, the vision tower, the aligner and the image delimiter
    # embeddings ship in the checkpoint but their modules land in follow-up PRs.
    _keys_to_ignore_on_load_unexpected = [
        r"(^|\.)mtp\..*",
        r"^vision\..*",
        r"^aligner\..*",
        r"^image_(start|end|newline)$",
    ]
    # fp32-critical parameters: exactly the tensors the released checkpoint stores
    # in F32 — the raw mHC parameters, the attention sinks and the gate biases
    # (`bias` / `bias_vl`; every other linear is bias-free). The norms and the
    # ratio-2 compressor gate ship BF16 and stay in the model dtype (the reference
    # upcasts them at load; the forward computes in fp32 either way).
    _keep_in_fp32_modules_strict = [
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "attn_sink",
        "bias",
        "bias_vl",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.bias)
            init.zeros_(module.bias_vl)
        elif isinstance(module, DeepseekV41Attention):
            init.zeros_(module.attn_sink)
        elif isinstance(module, DeepseekV41DecoderLayer):
            init.normal_(module.hc_attn_fn, mean=0.0, std=std)
            init.zeros_(module.hc_attn_base)
            init.ones_(module.hc_attn_scale)
            init.normal_(module.hc_ffn_fn, mean=0.0, std=std)
            init.zeros_(module.hc_ffn_base)
            init.ones_(module.hc_ffn_scale)
        elif isinstance(module, DeepseekV41Engram):
            init.normal_(module.embed.weight, mean=0.0, std=std)
            init.ones_(module.embed.scale)
            init.ones_(module.q_weight)
            init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41RotaryEmbedding):
            # `from_pretrained` builds on the meta device, so the inv_freq buffers
            # computed in __init__ never materialize — rebuild them here.
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV41TextModel(DeepseekV41PreTrainedModel):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepseekV41DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        self.engram_layout = EngramLayout.from_config(config)
        self.engram_hash_state: DeepseekV41NgramHashState | None = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def set_input_embeddings(self, value: nn.Module):
        self.embed = value

    def bind_tokenizer(self, tokenizer):
        """Build the engram hash state (tokenizer-derived compressed token map, prime
        bucket layout, per-layer hash multipliers). The hashes must be replicated
        exactly or the pretrained tables are meaningless. Called automatically on the
        first forward when the model was loaded from a hub checkpoint; call it
        explicitly otherwise."""
        self.engram_hash_state = DeepseekV41NgramHashState(self.config, tokenizer)
        return self

    def _ensure_hash_state(self, device: torch.device):
        if self.engram_layout is None or not self.engram_layout.layer_ids:
            return
        if self.engram_hash_state is None:
            if not getattr(self.config, "_name_or_path", ""):
                raise ValueError(
                    "The engram layers need the tokenizer to build their n-gram hash state "
                    "(compressed token map + hash multipliers). Call "
                    "`model.model.bind_tokenizer(tokenizer)` on a `DeepseekV41ForCausalLM` "
                    "(or `model.bind_tokenizer(tokenizer)` on a `DeepseekV41TextModel`), "
                    "or load the model from a hub checkpoint (it binds automatically)."
                )
            self.bind_tokenizer(AutoTokenizer.from_pretrained(self.config._name_or_path))

    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV41TopKRouter),
        # The residual stream is `hc_mult` parallel copies; the recorded
        # `hidden_states` are the collapsed per-block inputs (each layer's
        # `attn_norm` in/out, plus the initial embedding collapse).
        "hidden_states": OutputRecorder(DeepseekV41RMSNorm, layer_name="attn_norm"),
        "attentions": DeepseekV41Attention,
    }

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
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        elif self.engram_layout:
            # The engram hashes token ids; there is no way to recover them from embeddings.
            raise ValueError("engram layers require `input_ids` (the hash state cannot consume `inputs_embeds`)")
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        if isinstance(attention_mask, dict):
            # `generate()` may pass a per-layer-type mask dict; both V4.1 layer types
            # attend over the same sliding window, so any of them works.
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
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        self._ensure_hash_state(inputs_embeds.device)
        # Pads (attention_mask == 0) are hashed as DEAD so n-grams never span them.
        # Only a 2D mask carries per-token liveness; generate()'s per-layer-type mask
        # dict has no 2D form to read it from.
        live_mask = (
            attention_mask.bool() if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 else None
        )
        hash_ids = (
            self.engram_hash_state(input_ids, position_ids, live_mask) if self.engram_hash_state is not None else None
        )

        shared: dict = {}
        # One-hot initial mix: the first site collapses stream 0 only.
        pre_mix = hidden_states.new_zeros(*hidden_states.shape[:-1], dtype=torch.float32)
        pre_mix[..., 0] = 1.0
        for layer in self.layers:
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                hash_ids,
                None,
                shared,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
            )
        # Final collapse with the last site's pre mix, then the shared norm.
        hidden_states = DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    # The released checkpoint's config.json is composite (top-level `quantization_config`,
    # `text_config`, `vision_config`) with `architectures: [DeepseekV41ForCausalLM]`, so this
    # class must accept the composite config: `get_hf_quantizer` only sees `quantization_config`
    # on the config produced from `config_class` — pointing it at the bare text config silently
    # dropped the FP8 quantization config and fp8 tensors then failed to load. The text config
    # is unwrapped in `__init__` (same pattern as `MllamaForCausalLM`); a text config passed
    # directly is returned unchanged by `get_text_config()`.
    config_class = DeepseekV41Config

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.model = DeepseekV41TextModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
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
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        shift_labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `-100` (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is computed over tokens with labels in `[0, ..., config.vocab_size]`.
        shift_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Already-shifted next-token targets, aligned with `logits` (used for sequence and context
            parallel training, where the shift must happen before sharding). When given, they take
            precedence over `labels` for the loss.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        loss = None
        if labels is not None or shift_labels is not None:
            # `shift_labels` carries already-aligned targets (sequence / context
            # parallel training); `self.loss_function` shifts plain `labels` itself.
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, shift_labels=shift_labels
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _reorder_cache(self, past_key_values: "Cache", beam_idx: torch.LongTensor) -> "Cache":
        """Beam-search support: the cache layers' `reorder_cache` permutes the
        sliding-window and group state, and the engram n-gram history (which lives
        on the model, not the cache) must follow the beams too — otherwise a beam
        hashes its next tokens against another beam's predecessor history."""
        past_key_values.reorder_cache(beam_idx)
        hash_state = self.model.engram_hash_state
        if hash_state is not None and hash_state.history is not None:
            hash_state.history = hash_state.history.index_select(0, beam_idx.to(hash_state.history.device))
        return past_key_values


__all__ = [
    "DeepseekV41PreTrainedModel",
    "DeepseekV41TextModel",
    "DeepseekV41ForCausalLM",
    "DeepseekV41CSACache",
    "DeepseekV41EngramEmbedding",
    "EngramLayout",
    "DeepseekV41NgramHashState",
]
