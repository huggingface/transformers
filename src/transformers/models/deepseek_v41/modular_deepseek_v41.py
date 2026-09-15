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

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...generation import GenerationMixin
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, infer_channel_dimension_format, is_pil_image
from ...integrations.moe import use_experts_implementation
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.hub import cached_file
from ...utils.import_utils import is_torch_distributed_available
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4RotaryEmbedding
from ..mixtral.modeling_mixtral import MixtralExperts, MixtralTopKRouter, load_balancing_loss_func
from ..qwen2_vl.modeling_qwen2_vl import VisionAttention
from .configuration_deepseek_v41 import DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig


if is_torch_distributed_available():
    from torch.distributed.tensor import DTensor, Shard
else:  # the engram table's TP path is only reachable with torch.distributed
    DTensor = Shard = None


def eager_attention_forward_dms(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    selected_kv: torch.Tensor | None = None,
    selected_valid: torch.Tensor | None = None,
    **kwargs,
):
    """Eager shared-KV attention with the text backbone's per-head denominator sink."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    window = attn_weights.shape[-1]
    if selected_kv is not None:
        selected_kv = selected_kv.to(query.dtype)
        picked = torch.einsum("bhsd,bskd->bhsk", query, selected_kv) * scaling
        picked = picked.masked_fill(~selected_valid.unsqueeze(1), float("-inf"))
        attn_weights = torch.cat([attn_weights, picked], dim=-1)
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks.float()], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value.dtype)
    attn_output = torch.matmul(attn_weights[..., :window], value)
    if selected_kv is not None:
        attn_output = attn_output + torch.einsum("bhsk,bskd->bhsd", attn_weights[..., window:], selected_kv)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


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


class DeepseekV41HyperConnection(DeepseekV4HyperConnection):
    r"""One mHC site of V4.1's **single-pass** hyper-connections. Same `(fn, base,
    scale)` parametrization and the same `pre` / `post` / `comb` mapping as V4 (one
    projection of the normalized flattened stream; `comb` Sinkhorn-projected onto the
    doubly-stochastic manifold), with one difference in who consumes `pre`: V4 collapses
    the streams with the `pre` of the site that computed it, V4.1 feeds it to the NEXT
    site (attention collapses with the previous site's `pre`, the FFN with the
    attention's, the final norm with the last FFN's). The module therefore returns
    `(pre, post, comb)` and leaves the collapse / expand to the decoder layer."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(pre, post, comb)` of this site — `pre` is for the NEXT site's collapse (see
        the class docstring). Normalization is over the whole flattened hc·D stream
        (one statistic per token); `comb` is Sinkhorn-projected onto the
        doubly-stochastic manifold for `hc_sinkhorn_iters` steps."""
        hc = self.hc_mult
        # fp32 from the start — the reference upcasts before normalizing, so a
        # bf16/fp16 model must not round the stream before the mix projection.
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


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
    to `o_lora_rank`, then mixed to `hidden_size` by `o_b_proj`. This module owns the
    per-group block (`o_a_proj`)."""

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


# Sentinel in the engram token history: an n-gram never reaches across it (the
# sequence start, a pad, an image span). Any negative value is safe — compressed ids
# are non-negative and the sentinel is replaced by `engram_pad_id` before hashing.
ENGRAM_DEAD = -1


class DeepseekV41EngramHistoryLayer:
    r"""Cache-layer mixin carrying the engram's n-gram look-back: the last
    `max_ngram_size - 1` compressed token ids of every batch row, `ENGRAM_DEAD` where an
    n-gram must not reach. Same trick as Qwen4-Exp's token context in `conv_states`,
    minus the linear-attention layer it rides on there.

    The history is a property of the *sequence* (hashed once per forward for every
    engram layer), so it is parked on ONE cache layer: the first
    :class:`DeepseekV41CSACache`, the only V4.1-owned layer class and therefore the
    only one whose batch hooks we control. Mixed in ahead of the base layer so the
    layer's `reorder_cache` / `batch_repeat_interleave` / `batch_select_indices` /
    `reset` chains reach it: beam search and `num_return_sequences` permute the
    look-back together with the KV, with no model-side hook.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engram_context: torch.Tensor | None = None

    def update_engram_context(self, compressed: torch.Tensor, context_len: int) -> torch.Tensor:
        """Return `[B, context_len + S]`: the stored look-back (`ENGRAM_DEAD` on the
        first call) followed by `compressed`; keep the last `context_len` ids for the
        next call. CSA cannot rewind, even when sliding-window past recording is on."""
        if self.engram_context is None:
            self.engram_context = compressed.new_full((compressed.shape[0], context_len), ENGRAM_DEAD)
        full = torch.cat([self.engram_context.to(compressed.device), compressed], dim=1)
        self.engram_context = full[:, -context_len:]
        return full[:, -(context_len + compressed.shape[1]) :]

    def reset(self) -> None:
        super().reset()
        self.engram_context = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.engram_context is not None:
            self.engram_context = self.engram_context.index_select(0, beam_idx.to(self.engram_context.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.engram_context is not None:
            self.engram_context = self.engram_context.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.engram_context is not None:
            self.engram_context = self.engram_context[indices, ...]


class DeepseekV41CSACache(DeepseekV41EngramHistoryLayer, DynamicSlidingWindowLayer):
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
        source's indexer (`k_proj` + `k_norm`): the whole group scores against one key set.
      * `entry_count["compressor"]` / `token_count["compressor"]` — per-row
        completed groups and live source tokens. Padding never advances either
        sequence; partial groups retain their source positions across calls.

    The compress ratio is passed per call by the compressor (it is a per-layer config
    value, not a per-layer-type one, so it cannot be resolved at cache-construction
    time).
    """

    _layer_type = "shared_compressed_attention"
    is_croppable = False

    def __init__(self, config: "DeepseekV41TextConfig", **kwargs):
        super().__init__(sliding_window=config.sliding_window)
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_positions: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_lengths: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.entry_count: dict[str, torch.Tensor | None] = {"compressor": None}
        self.token_count: dict[str, torch.Tensor | None] = {"compressor": None}

    def crop(self, tokens_to_remove: int) -> None:
        # KV and engram updates already retain their needed look-back, so a
        # trim-only crop is also a no-op before lazy initialization.
        if tokens_to_remove != 0:
            raise RuntimeError("DeepseekV41CSACache cannot roll back compressed states.")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for attr in (
            "compressed_kv",
            "buffer_kv",
            "buffer_gate",
            "buffer_positions",
            "buffer_lengths",
            "entry_count",
            "token_count",
        ):
            state = getattr(self, attr)
            for name, tensor in state.items():
                if tensor is not None:
                    state[name] = tensor.index_select(0, beam_idx.to(tensor.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for attr in (
            "compressed_kv",
            "buffer_kv",
            "buffer_gate",
            "buffer_positions",
            "buffer_lengths",
            "entry_count",
            "token_count",
        ):
            state = getattr(self, attr)
            for name, tensor in state.items():
                if tensor is not None:
                    state[name] = tensor.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        for attr in (
            "compressed_kv",
            "buffer_kv",
            "buffer_gate",
            "buffer_positions",
            "buffer_lengths",
            "entry_count",
            "token_count",
        ):
            state = getattr(self, attr)
            for name, tensor in state.items():
                if tensor is not None:
                    state[name] = tensor.index_select(0, indices.to(tensor.device))

    def reset(self) -> None:
        self.keys = self.values = None
        self.is_initialized = False
        super().reset()
        for attr in (
            "compressed_kv",
            "buffer_kv",
            "buffer_gate",
            "buffer_positions",
            "buffer_lengths",
            "entry_count",
            "token_count",
        ):
            state = getattr(self, attr)
            for name in state:
                state[name] = None

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

    def update_compressor_states(
        self, name: str, compressed: torch.Tensor, group_counts: torch.Tensor, compress_ratio: int
    ) -> torch.Tensor:
        """Write each row's new groups immediately after its own previous groups.

        Capacity follows physical sequence length, avoiding a device-to-host
        synchronization to find the longest live row. Unused trailing slots are
        excluded by the per-query completed-group counts.
        """
        capacity = self.cumulative_length // compress_ratio
        previous = self.compressed_kv[name]
        if previous is None:
            updated = compressed.new_zeros(compressed.shape[0], 1, capacity, compressed.shape[-1])
        elif capacity > previous.shape[2]:
            updated = F.pad(previous, (0, 0, 0, capacity - previous.shape[2]))
        else:
            # Inference can fill unused slots without copying the running cache;
            # keep previous forward graphs intact when gradients are enabled.
            updated = previous.clone() if torch.is_grad_enabled() else previous
        slots = torch.arange(compressed.shape[2], device=compressed.device).unsqueeze(0)
        valid = slots < group_counts.unsqueeze(-1)
        destinations = torch.where(valid, self.entry_count["compressor"].unsqueeze(-1) + slots, 0)
        updates = compressed.masked_fill(~valid[:, None, :, None], 0)
        # New groups occupy previously unused zero slots. Invalid rows add zero
        # at slot zero rather than overwriting a live entry or requiring a dummy slot.
        updated.scatter_add_(2, destinations[:, None, :, None].expand_as(updates), updates)
        self.compressed_kv[name] = updated
        if name == "compressor":
            self.entry_count[name] = self.entry_count[name] + group_counts
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
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False) if self.compress_ratio > 1 else None
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_layer: DeepseekV41CSACache | None,
        position_ids: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool complete groups of live tokens; retain partial groups per row.

        Returns latents, their first-token RoPE positions, newly completed group
        counts, and the live-token count at each current query.
        """
        batch, seq_len, _ = hidden_states.shape
        ratio = self.compress_ratio
        positions = position_ids.expand(batch, -1)
        live = torch.ones_like(positions, dtype=torch.bool) if token_mask is None else token_mask.bool()
        previous_count = positions.new_zeros(batch)
        if cache_layer is not None:
            if cache_layer.token_count["compressor"] is not None:
                previous_count = cache_layer.token_count["compressor"]
            if cache_layer.entry_count["compressor"] is None:
                cache_layer.entry_count["compressor"] = positions.new_zeros(batch)
        query_lengths = previous_count.unsqueeze(-1) + live.long().cumsum(-1)

        if self.gate_proj is None:
            kv = self.kv_proj(hidden_states)
            gate = None
        else:
            # Pooling and its gate use fp32, including when weights are stored in bf16.
            kv = F.linear(hidden_states.float(), self.kv_proj.weight.float())
            gate = F.linear(hidden_states.float(), self.gate_proj.weight.float())

        if cache_layer is not None and cache_layer.buffer_kv["compressor"] is not None:
            buffered = cache_layer.buffer_kv["compressor"]
            buffer_live = torch.arange(buffered.shape[1], device=kv.device).unsqueeze(0) < cache_layer.buffer_lengths[
                "compressor"
            ].unsqueeze(-1)
            kv = torch.cat([buffered, kv], dim=1)
            if gate is not None:
                gate = torch.cat([cache_layer.buffer_gate["compressor"], gate], dim=1)
            positions = torch.cat([cache_layer.buffer_positions["compressor"], positions], dim=1)
            live = torch.cat([buffer_live, live], dim=1)

        length = kv.shape[1]
        counts = live.long().sum(-1)
        group_counts = counts // ratio
        # Stable compaction without sorting projected KV or synchronizing on a
        # data-dependent output size. Invalid input indices go to a discarded slot.
        destinations = torch.where(live, live.long().cumsum(-1) - 1, length)
        order = positions.new_zeros(batch, length + 1)
        order.scatter_(1, destinations, torch.arange(length, device=kv.device).expand(batch, -1))
        n_groups = length // ratio
        if cache_layer is not None:
            n_groups = min(n_groups, cache_layer.cumulative_length // ratio)
            remainder = counts % ratio
            tail_slots = (group_counts.unsqueeze(-1) * ratio + torch.arange(ratio - 1, device=kv.device)).clamp_max(
                length
            )
            tail_indices = order.gather(1, tail_slots)
            cache_layer.buffer_kv["compressor"] = kv.gather(1, tail_indices.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))
            cache_layer.buffer_gate["compressor"] = (
                None if gate is None else gate.gather(1, tail_indices.unsqueeze(-1).expand(-1, -1, gate.shape[-1]))
            )
            cache_layer.buffer_positions["compressor"] = positions.gather(1, tail_indices)
            cache_layer.buffer_lengths["compressor"] = remainder
            cache_layer.token_count["compressor"] = query_lengths[:, -1]

        group_indices = order[:, : n_groups * ratio]
        group_positions = positions.gather(1, group_indices[:, ::ratio])
        if n_groups == 0:
            return None, group_positions, group_counts, query_lengths
        grouped_kv = kv.gather(1, group_indices.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))
        valid = torch.arange(n_groups, device=kv.device).unsqueeze(0) < group_counts.unsqueeze(-1)
        if gate is None:
            latent = grouped_kv
        else:
            grouped_gate = gate.gather(1, group_indices.unsqueeze(-1).expand(-1, -1, gate.shape[-1]))
            grouped_kv = grouped_kv.reshape(batch, n_groups, ratio, -1)
            grouped_gate = grouped_gate.reshape(batch, n_groups, ratio, -1)
            latent = (grouped_kv * grouped_gate.softmax(dim=2, dtype=torch.float32)).sum(dim=2)
        latent = latent.masked_fill(~valid.unsqueeze(-1), 0)
        return self.kv_norm(latent.to(hidden_states.dtype)), group_positions, group_counts, query_lengths


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

    Key sharing: the keys are `k_norm(k_proj(latent))` of the *compressor latent* — only a
    layer that also owns its compressor (`kv_source_layer_ids`) can produce them
    (`owns_k`); every later index source ("Reindex" mode) rescores with its own weights
    against the keys published by its group's source. Queries come from the attention's
    low-rank residual (`q_a_norm(q_a_proj(x))`) through `q_b_proj`, rotated with the compress
    rope. The candidate source layer additionally publishes the two-level-top-k
    candidate mask that constrains all later index sources."""

    # Query chunking of the [chunk, heads, T] score tensor: at most this many fp32
    # elements per chunk (2 GiB). T is the number of compressed groups seen so far.
    _SCORE_BUDGET = 2**29

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
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        if self.owns_k:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        latent: torch.Tensor | None,
        group_positions: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV41CSACache | None,
        shared: dict,
    ) -> None:
        """Publishes `shared["topk_idx"]` — the `index_topk` compressed entries each
        query attends to (`[B, S, top_k]`, `-1` = no entry) — and `shared["candidates"]`
        at the candidate source layer."""
        batch, seq_len, _ = hidden_states.shape
        ratio = self.compress_ratio

        # 1. Publish the index keys of the groups that completed in this call. The keys
        #    are derived from the PRE-rope latent, before the compressor rotates the
        #    same values into the main cache.
        if self.owns_k:
            if cache_layer is None:
                # Without a cache, the shared slot belongs to this source's current
                # forward only, not to a preceding source layer.
                shared["index_k"] = None
            if latent is not None:
                k = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary_emb(k, position_ids=group_positions, layer_type="compress")
                k = apply_rotary_pos_emb(k, cos, sin)
                # QAT semantics: indexer keys are FP4-quantized (ue8m0 scale per 32
                # channels) before they land in the shared key cache.
                k = _fake_quant_fp4_block(k, block_size=32)
                k = k.unsqueeze(1)  # [B, 1, G, idh]
                if cache_layer is not None:
                    cache_layer.update_compressor_states("indexer", k, shared["group_counts"], ratio)
                else:
                    shared["index_k"] = k
            # Publish the RUNNING key cache — decode steps between group boundaries emit
            # nothing new but the group still scores against everything emitted so far.
            if cache_layer is not None:
                shared["index_k"] = cache_layer.compressed_kv["indexer"]
        index_k = shared.get("index_k")
        if index_k is not None:
            index_k = index_k.to(hidden_states.device)  # source group may sit on another device
        compressed_len = 0 if index_k is None else index_k.shape[2]
        if compressed_len == 0:
            shared["topk_idx"] = None
            if self.is_candidate_source:
                shared["candidates"] = None
            return

        # 2. Score the queries against the shared keys, in query chunks: the score
        #    tensor is [chunk, heads, T], and T grows with the context (one entry per
        #    `ratio` tokens), so a whole-prefill [S, heads, T] would not fit at long
        #    context. Each chunk publishes its own top-k; only [S, top_k] indices leave.
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        # QAT semantics: the indexer query is FP4-quantized too, so the top-k
        # selection matches the trained quantized scoring.
        q = _fake_quant_fp4_block(q, block_size=32).float()
        keys = index_k[:, 0].float()  # [B, T, idh]
        weights = self.weights_proj(hidden_states).float() * self.heads_scaling  # [B, S, heads]
        # Counts follow live tokens, independently of padding offsets or custom
        # RoPE coordinates. A pooled entry becomes visible only after its last token.
        compress_lens = shared["compress_lens"].to(keys.device).unsqueeze(-1)
        entry_indices = torch.arange(compressed_len, device=keys.device).view(1, 1, -1)
        top_k = min(self.index_topk, compressed_len)
        candidates_prev = shared.get("candidates") if (self.uses_candidates and not self.is_candidate_source) else None
        if candidates_prev is not None:
            candidates_prev = candidates_prev.to(keys.device)

        chunk = max(1, min(seq_len, self._SCORE_BUDGET // max(self.num_heads * compressed_len, 1)))
        topk_idx, candidates_out = [], []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            scores = torch.einsum("bshd,btd->bsht", q[:, start:end], keys)
            scores = scores.relu_() * self.softmax_scale
            index_scores = (scores * weights[:, start:end].unsqueeze(-1)).sum(dim=2)  # [B, c, T]
            index_scores = index_scores.masked_fill(entry_indices >= compress_lens[:, start:end], float("-inf"))
            # Two-level top-k: the candidate source publishes its block mask; every
            # later index source scores only inside it.
            if self.is_candidate_source:
                cand = select_candidate_blocks(
                    index_scores, compress_lens[:, start:end], self.candidate_topk_blocks, self.candidate_block_size
                )
                candidates_out.append(cand)
            elif candidates_prev is not None:
                index_scores = index_scores.masked_fill(~candidates_prev[:, start:end], float("-inf"))
            # Early queries can have fewer visible groups than `index_topk`: those picks
            # come back with a -inf score and are marked -1 (never attended).
            if top_k > 0:
                picked = index_scores.topk(top_k, dim=-1, sorted=False)
                idx = torch.where(picked.values > float("-inf"), picked.indices, torch.full_like(picked.indices, -1))
            else:
                idx = index_scores.new_empty((batch, end - start, 0), dtype=torch.long)
            topk_idx.append(idx)
        if self.is_candidate_source:
            shared["candidates"] = torch.cat(candidates_out, dim=1)
        shared["topk_idx"] = torch.cat(topk_idx, dim=1)  # [B, S, top_k], -1 = no entry


class DeepseekV41Attention(nn.Module):
    r"""Latent shared-KV attention over two KV sources: a sliding window of raw KV plus,
    when the layer has a compressed branch, the *shared* compressed KV of its group.

    - Q and the output projection are low-rank; the output projection is grouped
      (block-diagonal `o_a_proj` over `o_groups`, then the mixing `o_b_proj`).
    - K=V is a single latent (`kv_proj` + `kv_norm`); the attention output's rope slice is
      inverse-rotated so the shared rotated cache works.
    - Per-head learnable attention sink (`sinks`), like gpt-oss.
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

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV41GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.compressor = DeepseekV41Compressor(config, layer_idx) if self.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, layer_idx) if self.is_index_source else None
        # Latent RoPE positions come from each complete group's first live token,
        # including partial groups carried across cached calls.
        # Only kv-source layers own one (shared by their group).
        self.compress_rotary = DeepseekV41RotaryEmbedding(config) if self.is_kv_source else None  # trf-ignore: TRF050

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
        padding_mask = kwargs.pop("padding_mask", None)
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)  # [B, H, S, D]

        kv = self.kv_norm(self.kv_proj(hidden_states))
        kv = apply_rotary_pos_emb(kv, cos, sin).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        # QAT semantics: the window KV cache stores FP8-quantized values (one ue8m0
        # scale per 32 channels, RoPE tail included) — part of the model, applied
        # even in otherwise-unquantized runs.
        kv = _fake_quant_fp8_block(kv, block_size=32)
        if past_key_values is not None:  # K == V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        selected_kv = selected_valid = None
        if self.compress_ratio:
            cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
            latent = group_positions = None
            if self.is_kv_source:
                if cache_layer is None:
                    shared["compress_kv"] = None
                latent, group_positions, group_counts, query_lengths = self.compressor(
                    hidden_states, cache_layer, position_ids, padding_mask
                )
                shared["group_counts"] = group_counts
                shared["compress_lens"] = query_lengths // self.compress_ratio
                if padding_mask is not None:
                    shared["compress_lens"] = shared["compress_lens"].masked_fill(~padding_mask, 0)

            # The indexer consumes the PRE-rope latent; it must run before the latent is
            # rotated into the main compressed cache.
            if self.is_index_source:
                self.indexer(hidden_states, q_residual, latent, group_positions, position_ids, cache_layer, shared)
            topk_idx = shared.get("topk_idx")

            if latent is not None:
                cos_c, sin_c = self.compress_rotary(latent, position_ids=group_positions, layer_type="compress")
                rotated = apply_rotary_pos_emb(latent, cos_c, sin_c)
                # QAT semantics: the compressed KV cache stores FP4-quantized latents
                # (e2m1 grid, one e4m3 scale per 16 channels).
                rotated = _fake_quant_fp4_block(rotated, block_size=16, e4m3_scales=True)
                rotated = rotated.unsqueeze(1)  # [B, 1, G, hd]
                if cache_layer is not None:
                    cache_layer.update_compressor_states(
                        "compressor", rotated, shared["group_counts"], self.compress_ratio
                    )
                else:
                    shared["compress_kv"] = rotated
            if self.is_kv_source and cache_layer is not None:
                # Publish the RUNNING compressed cache — a decode step between group
                # boundaries emits nothing new, but the group still attends over
                # everything emitted so far.
                shared["compress_kv"] = cache_layer.compressed_kv["compressor"]
            compressed_kv = shared.get("compress_kv")
            # Gather ONLY the entries the indexer picked for each query ([B, S, top_k, D]):
            # attention over the whole compressed cache with a -inf bias is the same
            # softmax, but its [heads, S, T] scores would not fit at long context. A
            # compressed branch with no index source in this forward (legal but unusual
            # schedule) attends no compressed entry.
            if compressed_kv is not None and topk_idx is not None and topk_idx.shape[-1] > 0:
                entries = compressed_kv[:, 0].to(kv.device)  # [B, T, D]; source may sit elsewhere
                topk_idx = topk_idx.to(kv.device)
                selected_valid = topk_idx >= 0
                rows = torch.arange(batch, device=kv.device).view(batch, 1, 1)
                selected_kv = entries[rows, topk_idx.clamp_min(0)]  # [B, S, top_k, D]

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward_dms
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
            selected_kv=selected_kv,
            selected_valid=selected_valid,
            **kwargs,
        )

        # K == V carried RoPE on its rope slice; remove the query's rotation from the
        # output before the grouped projection mixes the heads.
        attn_output = apply_rotary_pos_emb(attn_output, cos, sin, inverse=True)
        grouped = attn_output.reshape(batch, seq_len, self.config.o_groups, -1)
        output = self.o_b_proj(self.o_a_proj(grouped).flatten(2))
        return output, attn_weights


class DeepseekV41TopKRouter(MixtralTopKRouter):
    """MoE gate. The correction bias (`e_score_correction_bias`) steers expert
    *selection* only; the routing weights come from the unbiased scores. Image-span
    tokens switch to a separate `e_score_correction_bias_vl` (training
    `noaux_tc_for_vl`). Both are persistent fp32 buffers, like V4's."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.gate_temp = config.gate_temp
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))
        self.e_score_correction_bias_vl = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(
        self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim).float()
        logits = F.linear(flat, self.weight.float()) / self.gate_temp
        scores = self.score_fn(logits)
        bias = self.e_score_correction_bias
        if image_mask is not None and image_mask.any():
            bias = torch.where(image_mask.reshape(-1, 1), self.e_score_correction_bias_vl, bias)
        indices = (scores + bias).topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.top_k > 1:
            # `+1e-20` on the sum — NOT `rms_norm_eps`; matches training.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        # `logits` first so `OutputRecorder(..., index=0)` records router logits.
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV41MLP(nn.Module):
    """The shared expert: a SwiGLU MLP with the training-time clamps that keep fp8/fp4
    activations in range (`up` on both sides, `gate` from above), computed in fp32."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x).float(), self.up_proj(x).float()
        if self.limit > 0:
            up = up.clamp(min=-self.limit, max=self.limit)
            gate = gate.clamp(max=self.limit)
        return self.down_proj((self.act_fn(gate) * up).to(x.dtype))


@use_experts_implementation
class DeepseekV41Experts(MixtralExperts):
    """Routed experts as 3D tensors: `gate_up_proj[e]` = `[w1; w3]` (gate rows first),
    `down_proj[e]` = `w2`. Same clamped SwiGLU as the shared expert; the routing
    weight multiplies the fp32 activation BEFORE the down projection (the reference's
    order — `w2(weight * act)`, not `weight * w2(act)`)."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.limit = config.swiglu_limit

    def _clamped_swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate, up = gate.float(), up.float()
        if self.limit > 0:
            up = up.clamp(min=-self.limit, max=self.limit)
            gate = gate.clamp(max=self.limit)
        return self.act_fn(gate) * up

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class (like V4 / gpt-oss) so the batched_mm / grouped_mm backends
        # swapped in by `@use_experts_implementation` apply the same clamp + SiLU on their
        # packed gate_up output.
        gate, up = gate_up.chunk(2, dim=-1)
        return self._clamped_swiglu(gate, up).to(gate_up.dtype)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """Eager dispatch: returns the fp32 accumulator (the block adds the shared
        expert in fp32 before casting back, as the reference does)."""
        inter = self.intermediate_dim
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            # Token-major order (`mask[e].T` is [tokens, top_k]), and gate / up as two
            # GEMMs over the halves of the fused weight: both keep the accumulation
            # order of the reference's per-expert `w1` / `w3` loop (one fused GEMM or a
            # top_k-major row order rounds differently at some shapes).
            token_idx, top_k_pos = torch.where(mask[expert_idx].T)
            current_state = hidden_states[token_idx]
            weight = self.gate_up_proj[expert_idx]
            gate, up = F.linear(current_state, weight[:inter]), F.linear(current_state, weight[inter:])
            current = self._clamped_swiglu(gate, up) * top_k_weights[token_idx, top_k_pos, None]
            current = F.linear(current.to(current_state.dtype), self.down_proj[expert_idx])
            final.index_add_(0, token_idx, current.float())
        return final


class DeepseekV41SparseMoeBlock(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        # DSpark draft layers (M2) have their own expert counts; this block is only
        # ever built for backbone layers.
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV41Experts(config)
        self.shared_experts = DeepseekV41MLP(config)

    def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, weights, indices = self.gate(hidden_states, image_mask)
        # routed + shared summed in fp32 (eager experts return fp32; the batched_mm /
        # grouped_mm backends return the model dtype), then one cast back.
        y = self.experts(flat, indices, weights).float() + self.shared_experts(flat).float()
        return y.to(hidden_states.dtype).view(shape)


class DeepseekV41EngramEmbedding(nn.Embedding):
    """The n-gram hash table: fp8 rows with per-row / per-32-channel E8M0 scales in the
    checkpoint, dequantized on lookup. The scales are `weight_scale_inv`, the name the
    FP8 quantizer gives every checkpoint `.scale` (and that `FP8Linear` uses), so a
    `dequantize=False` load lands them here. When the model is loaded in a float dtype
    (`dequantize=True`, or a bf16 checkpoint) the quantizer has already folded the
    scales into the rows, the table is a plain embedding and `weight_scale_inv` is
    unused. ~98 GB per table in the released checkpoint — memory-map friendly (pure
    row gather).

    An `nn.Embedding` so tensor parallelism shards it along the embedding dim
    (`colwise_gather_output`, like Qwen4-Exp's n-gram table): each rank holds
    `head_dim / tp_size` channels of every row — whole 32-channel scale blocks, so the
    per-block dequantization stays rank-local before the output gather."""

    def __init__(self, num_embeddings: int, head_dim: int, block_size: int = 32):
        super().__init__(num_embeddings, head_dim)
        self.block_size = block_size
        self.weight_scale_inv = nn.Parameter(torch.empty(num_embeddings, head_dim // block_size))

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        weight, scale = self.weight, self.weight_scale_inv
        mesh = None
        if DTensor is not None and isinstance(weight, DTensor):
            # TP: gather on the local shards, hand back a Shard(-1) DTensor for the output gather
            mesh, weight, scale = weight.device_mesh, weight.to_local(), scale.to_local()
        if DTensor is not None and isinstance(hash_ids, DTensor):
            hash_ids = hash_ids.to_local()
        # Under `device_map` the table is in `_no_placement_params` and stays wherever it
        # fits (typically host RAM) while the ids arrive on the layer's device: run the
        # gather where the table lives and move only the rows (Qwen4-Exp pattern).
        table_device = weight.device if weight.device.type != "meta" else hash_ids.device
        ids = hash_ids.to(table_device)
        values = F.embedding(ids, weight)
        if weight.dtype == torch.float8_e4m3fn:
            scales = F.embedding(ids, scale).float()
            values = values.float().unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)
            values = values.flatten(-2)
        values = values.to(hash_ids.device)
        if mesh is not None:
            values = DTensor.from_local(values, mesh, [Shard(-1)], run_check=False)
        return values


class DeepseekV41Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches.

    `wkv` turns the gathered rows (`n_hash_cols` of them per token) into one key per hc
    stream plus a shared value. The gate is a sigmoid of the signed sqrt of a
    normalized dot product between the stream and the key (weights
    `q_weight * k_weight`, used only as a product). The hash ids are computed and the
    rows gathered once per forward by the model (:class:`DeepseekV41NgramHashState` +
    the model-level :class:`DeepseekV41EngramEmbedding` tables) and passed in per
    engram layer — the ~98 GB tables must live outside the decoder layer so the layer
    stays a no-split unit under `device_map` while the table is excluded from
    placement."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_hash_index = config.engram_layer_ids.index(layer_idx)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.wkv = nn.Linear(
            n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.empty(config.hc_mult, config.hidden_size))

    def forward(
        self, hidden_streams: torch.Tensor, rows: torch.Tensor, token_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """hidden_streams: [B, S, hc, D]; rows: [B, S, n_hash_cols, head_dim] gathered
        table rows; token_mask: [B, S], False shuts the gate so those positions pass
        through untouched."""
        # The dequantized fp8 rows are exact in any dtype; cast to the stream's dtype
        # (NOT the weight's: under `dequantize=False` `wkv` is an FP8Linear whose
        # weight is float8, and its input must stay bf16).
        kv = self.wkv(rows.flatten(-2).to(hidden_streams.dtype))
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


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _find_next_prime(start: int, seen: set) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables: a position is hashed as
    `max_ngram_size - 1` n-grams (2..N-gram), each split over `n_heads` heads; every
    (n-gram size, head) pair owns a disjoint prime-sized bucket range — the primes are
    drawn in order above `engram_vocab_size` and never reused. `primes` is
    `[L][n-gram-1][heads]` (the per-step modulus is over all heads of one n-gram size);
    `offsets` is `[L][n_cols]`, PER LAYER over the flat (n-gram, head) order: each
    layer's table is addressed from its own start, and the ranges inside it are
    disjoint because the primes are never reused."""

    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple
    offsets: tuple
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
        offsets = []
        for layer in primes:
            row, total = [], 0
            for prime in (p for per in layer for p in per):
                row.append(total)
                total += prime
            offsets.append(tuple(row))
        return cls(
            max_ngram_size=config.engram_max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.engram_num_embeddings),
            primes=tuple(primes),
            offsets=tuple(offsets),
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


class DeepseekV41NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there — once per
    forward, for all engram layers.

    Ids go through the compressed table, then each position is hashed with the
    `max_ngram_size - 1` tokens before it; look-back stops at the start of the sequence
    and at any DEAD token (a pad, an image-span token), so an n-gram never spans one.
    Across the prefill / chunked prefill / decode split the look-back is read from and
    written to the cache (:class:`DeepseekV41EngramHistoryLayer`), so it follows the
    KV through beam reorders and batch expansion; without a cache every call starts
    from an empty look-back.

    The prime buckets and per-layer multipliers depend only on the config and are
    non-persistent buffers (rebuilt by `_init_weights` after the meta-device load, like
    `inv_freq`); the compressed token map is the tokenizer's and is filled by
    `bind_tokenizer` — until then the module refuses to hash."""

    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__()
        self.layout = EngramLayout.from_config(config)
        self.max_ngram_size = self.layout.max_ngram_size
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.engram_pad_id = config.engram_pad_id
        self.pad_id: int | None = None  # compressed id of `engram_pad_id`, known once bound
        tables = self.hash_tables()
        self.primes = nn.Buffer(tables["primes"], persistent=False)
        self.offsets = nn.Buffer(tables["offsets"], persistent=False)
        self.multipliers = nn.Buffer(tables["multipliers"], persistent=False)
        # Filled by `bind_tokenizer` (tokenizer-derived); empty until then. A buffer so
        # it follows `.to()` / device_map like the config tables.
        self.token_map = nn.Buffer(torch.empty(0, dtype=torch.long), persistent=False)

    def hash_tables(self) -> dict[str, torch.Tensor]:
        """The config-derived tables as fresh tensors: `primes` [L, n-gram-1, heads],
        `offsets` [L, n_cols], `multipliers` [L, max_ngram_size]."""
        return {
            "primes": torch.tensor(self.layout.primes),
            "offsets": torch.tensor(self.layout.offsets),
            "multipliers": compute_hash_multipliers(
                self.layout.layer_ids, self.max_ngram_size, self.compressed_vocab_size
            ),
        }

    def bind_tokenizer(self, tokenizer):
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != self.compressed_vocab_size:
            raise ValueError(
                f"The tokenizer-derived compressed vocabulary size ({vocab_size}) does not match "
                f"`engram_compressed_vocab_size` ({self.compressed_vocab_size}); the hash "
                "multipliers would silently rehash the whole engram table."
            )
        self.pad_id = token_map[self.engram_pad_id]
        self.token_map = torch.tensor(token_map, device=self.primes.device)

    def forward(
        self, input_ids: torch.Tensor, token_mask: torch.Tensor | None, past_key_values: Cache | None
    ) -> torch.Tensor:
        """Returns `[B, S, n_engram_layers, n_hash_cols]` hash ids. `token_mask` is
        `[B, S]`, False marking DEAD tokens."""
        if self.token_map.numel() == 0:
            raise ValueError(
                "The engram layers need the tokenizer to build their n-gram hash state "
                "(compressed token map). Call `bind_tokenizer(tokenizer)` on the text "
                "backbone — `model.model.bind_tokenizer(...)` on a "
                "`DeepseekV41ForCausalLM` / `DeepseekV41ForConditionalGeneration`, "
                "`model.bind_tokenizer(...)` on a bare `DeepseekV41TextModel` — or load "
                "the model with `from_pretrained` from a checkpoint that ships its "
                "tokenizer (it binds automatically)."
            )
        context_len = self.max_ngram_size - 1
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, ENGRAM_DEAD)
        if past_key_values is None:
            empty = compressed.new_full((compressed.shape[0], context_len), ENGRAM_DEAD)
            history = torch.cat([empty, compressed], dim=1)
        else:
            layer = next((l for l in past_key_values.layers if isinstance(l, DeepseekV41EngramHistoryLayer)), None)
            if layer is None:
                raise ValueError(
                    "The engram n-gram look-back lives on a `shared_compressed_attention` cache layer "
                    "(a KV-source layer); this cache has none."
                )
            history = layer.update_engram_context(compressed, context_len)

        # Look-back windows over [context | current]: once a shift hits DEAD, that
        # position's longer n-grams are blocked too and hash the pad id instead.
        seq_len = compressed.shape[1]
        tokens, blocked = [], torch.zeros_like(compressed, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = history[:, context_len - shift : context_len - shift + seq_len]
            blocked = blocked | (source == ENGRAM_DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim=-1)  # [B, S, max_ngram_size]

        # XOR the multiplied ids one look-back at a time: after step i the running value
        # is the hash of the (i+1)-gram, landing in its own prime bucket range.
        products = tokens.unsqueeze(2) * self.multipliers  # [B, S, L, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets.unsqueeze(0)


class DeepseekV41DecoderLayer(GradientCheckpointingLayer):
    r"""A V4.1 block: the residual stream is `hc_mult` parallel copies (hyper-
    connections), with the engram lookup injected at its layers before the block.

    Single-pass mHC: each site's :class:`DeepseekV41HyperConnection` returns
    `(pre, post, comb)`, but the `pre` a site computes is consumed by the *next* site —
    attention collapses with `pre_mix` (the previous site's), the FFN with the
    attention site's, and the layer hands its FFN `pre` on."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV41Attention(config, layer_idx)
        self.mlp = DeepseekV41SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # CODEPATH: DeepSeek-V4.1-Flash ships n-gram hash tables on layers 1 and 14
        # (engram_layer_ids=[1, 14]); every other checkpoint — and the tiny test
        # configs — sets engram_layer_ids=[] and takes the None side (no engram path).
        self.engram = DeepseekV41Engram(config, layer_idx) if layer_idx in config.engram_layer_ids else None
        self.attn_hc = DeepseekV41HyperConnection(config)
        self.ffn_hc = DeepseekV41HyperConnection(config)

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
        engram_rows: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        shared: dict,
        image_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_streams: [B, S, hc, hidden]; engram_rows: this layer's gathered table
        # rows [B, S, n_hash_cols, head_dim] (None on non-engram layers); token_mask: the
        # engram gate mask ([B, S], False = DEAD — image spans and pads); image_mask:
        # [B, S], True inside image spans — switches the router to its VL correction bias.
        if self.engram is not None and engram_rows is not None:
            hidden_streams = self.engram(hidden_streams, engram_rows, token_mask)

        residual = hidden_streams
        attn_pre, attn_post, attn_comb = self.attn_hc(hidden_streams)
        collapsed = self.hc_collapse(hidden_streams, pre_mix)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), shared=shared, **kwargs)
        hidden_streams = self.hc_expand(attn_output, residual, attn_post, attn_comb)

        residual = hidden_streams
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(hidden_streams)
        collapsed = self.hc_collapse(hidden_streams, attn_pre)
        ffn_output = self.mlp(self.post_attention_layernorm(collapsed), image_mask=image_mask)
        hidden_streams = self.hc_expand(ffn_output, residual, ffn_post, ffn_comb)
        return hidden_streams, ffn_pre


# Checkpoint keys of the vision tower, the aligner and the image delimiter
# embeddings: modules of the VL classes (DeepseekV41Model /
# DeepseekV41ForConditionalGeneration), unexpected for the text-only ones.
_VISION_ONLY_LOAD_PATTERNS = [r"^vision\..*", r"^aligner\..*", r"^image_(start|end|newline)$"]


# Deliberate: this base serves BOTH model types. The text backbone chain
# (DeepseekV41TextModel, registered as `deepseek_v41_text`) needs the flat
# DeepseekV41TextConfig; DeepseekV41ForCausalLM overrides with the composite
# DeepseekV41Config because the released checkpoint's config.json is composite and
# its top-level quantization_config must reach the quantizer.
@auto_docstring
class DeepseekV41PreTrainedModel(PreTrainedModel):  # trf-ignore: TRF001
    config_class = DeepseekV41TextConfig

    base_model_prefix = "model"
    # `DeepseekV41EngramEmbedding` is no-split so `_no_placement_params` can exclude
    # its whole table (see below) instead of splitting it into offloaded pieces.
    _no_split_modules = ["DeepseekV41DecoderLayer", "DeepseekV41EngramEmbedding"]
    # `past_key_values` as everywhere; `shared` is the per-forward CSA2 group-state dict
    # that every layer must see as ONE object (accelerate's hooks would otherwise hand
    # each layer a device-moved copy, losing the source layers' writes).
    _skip_keys_device_placement = ["past_key_values", "shared"]
    # Eager-only, same reasons as V4: FA caps head_dim at 256 (V4.1 uses 512); SDPA has
    # no per-head sink term; the compressed branch concatenates entries onto the KV axis
    # inside the block, after the model-level mask was built.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    # The compressor's group-buffer state isn't rewindable across drafts.
    _is_stateful = True
    # The released checkpoint ships the DSpark draft layers, the vision tower, the
    # aligner and the image delimiter embeddings. The text-only classes
    # (DeepseekV41TextModel / DeepseekV41ForCausalLM) own no matching modules, so they
    # ignore those keys when loading the full checkpoint; the VL classes own them and
    # drop the vision patterns again in their `post_init` (children re-merge them).
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*", *_VISION_ONLY_LOAD_PATTERNS]
    # The engram tables' `weight_scale_inv` only exists as a parameter under
    # `dequantize=False`; a dequantized (or bf16) load has folded it into the rows.
    _keys_to_ignore_on_load_missing = [r"engram_tables\.\d+\.weight_scale_inv$"]
    # fp32-critical parameters: exactly the tensors the released checkpoint stores in
    # F32 — the mHC sites, the attention sinks and the router's correction biases
    # (fp32 buffers from construction; listed so `from_pretrained(dtype=...)` keeps
    # them fp32, as V4 does). The norms and the ratio-2 compressor gate ship BF16 and
    # stay in the model dtype (the reference upcasts them at load; the forward
    # computes in fp32 either way).
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "sinks",
        "e_score_correction_bias",
        "e_score_correction_bias_vl",
    ]
    # The released checkpoint ships these projections in BF16 with no companion
    # `.scale` (every other linear is fp8 / packed fp4). Listed here (non-strict) so
    # the FP8 quantizer's `get_modules_to_not_convert` auto-skips them, like V4;
    # non-strict has no dtype effect at BF16, so they stay BF16.
    _keep_in_fp32_modules = [
        "self_attn.compressor.kv_proj",
        "self_attn.compressor.gate_proj",
        "self_attn.indexer.k_proj",
        "self_attn.indexer.weights_proj",
    ]
    # The two engram tables are ~98 GB each in the released checkpoint. Like
    # Qwen4-Exp's n-gram table they are excluded from `device_map` placement (names are
    # relative to the no-split `DeepseekV41EngramEmbedding` module): when a table does
    # not fit an accelerator it is skipped by the device-map inference and loads into
    # host RAM with no offload hook — the model-level gather runs there and moves only
    # the rows. On an accelerator large enough to hold it, it is placed normally.
    _no_placement_params = ["weight", "weight_scale_inv"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        # `self` may be a VL wrapper whose (composite / vision) config carries no
        # `initializer_range` of its own — fall back to the text config, then 0.02.
        std = getattr(self.config, "initializer_range", None) or getattr(
            self.config.get_text_config(), "initializer_range", 0.02
        )
        if isinstance(module, DeepseekV41Model):
            # Raw `nn.Parameter`s (no module type the generic init recognizes).
            init.normal_(module.image_start, mean=0.0, std=std)
            init.normal_(module.image_end, mean=0.0, std=std)
            init.normal_(module.image_newline, mean=0.0, std=std)
        elif isinstance(module, DeepseekV41TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
            init.zeros_(module.e_score_correction_bias_vl)
        elif isinstance(module, DeepseekV41Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV41Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV41HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV41EngramEmbedding):
            init.normal_(module.weight, mean=0.0, std=std)
            init.ones_(module.weight_scale_inv)
        elif isinstance(module, DeepseekV41Engram):
            init.ones_(module.q_weight)
            init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41NgramHashState):
            # Config-derived hash tables are non-persistent buffers: like `inv_freq`, the
            # meta-device load leaves them empty. The token map is the tokenizer's
            # (`bind_tokenizer` fills it) and is never touched here.
            for name, table in module.hash_tables().items():
                init.copy_(getattr(module, name), table)
        elif isinstance(module, DeepseekV41VisionRotaryEmbedding):
            # `from_pretrained` builds on the meta device, so the inv_freq buffer
            # computed in `__init__` never materializes — rebuild it (the config-derived
            # buffer is non-persistent, like the text rotary's).
            init.copy_(module.inv_freq, module.compute_inv_freq(module.config))
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Loads as usual, then binds the checkpoint's own tokenizer to the engram hash
        state (the compressed token map is tokenizer-derived, and the forward must stay
        free of hub / disk access). Checkpoints without a tokenizer — e.g. weights saved
        on their own — load unbound; `bind_tokenizer` is then the caller's job."""
        loaded = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded  # `output_loading_info=True`
        text_model = model.base_model  # the text backbone owns the hash state
        if getattr(text_model, "engram_hash_state", None) is not None and pretrained_model_name_or_path is not None:
            hub_kwargs = {
                key: kwargs[key]
                for key in ("cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder")
                if key in kwargs
            }
            has_tokenizer = cached_file(
                pretrained_model_name_or_path,
                "tokenizer_config.json",
                **hub_kwargs,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            if has_tokenizer is not None:
                from ..auto import AutoTokenizer

                text_model.bind_tokenizer(AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **hub_kwargs))
        return loaded


@auto_docstring
class DeepseekV41TextModel(DeepseekV41PreTrainedModel):
    def __init__(self, config: DeepseekV41TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # CODEPATH: DeepSeek-V4.1-Flash ships engram layers ([1, 14]) and takes the
        # hash-state path; tiny test configs set engram_layer_ids=[] and take None.
        # The ~98 GB tables are model-level (keyed by layer index, `engram_tables["1"]`)
        # rather than inside their decoder layer: a no-split layer holding a
        # `_no_placement_params` tensor gets split down to parameter-level device-map
        # entries, which carry no accelerate hooks. Registered BEFORE `layers` so the
        # device-map inference always has a later accelerator to test against.
        self.engram_hash_state = DeepseekV41NgramHashState(config) if config.engram_layer_ids else None
        self.engram_tables = nn.ModuleDict(
            {
                str(layer_idx): DeepseekV41EngramEmbedding(config.engram_num_embeddings[k], config.engram_head_dim)
                for k, layer_idx in enumerate(config.engram_layer_ids)
            }
        )
        self.layers = nn.ModuleList([DeepseekV41DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)
        self.engram_layout = EngramLayout.from_config(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value

    def bind_tokenizer(self, tokenizer):
        """Give the engram hash state its tokenizer (the compressed token map). The
        hashes must be replicated exactly or the pretrained tables are meaningless.
        `from_pretrained` binds the checkpoint's own tokenizer; call this explicitly
        when the model was built any other way."""
        if self.engram_hash_state is not None:
            self.engram_hash_state.bind_tokenizer(tokenizer)
        return self

    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV41TopKRouter, index=0),
        # The residual stream is `hc_mult` parallel copies; the recorded
        # `hidden_states` are the collapsed per-block inputs (each layer's
        # `input_layernorm` in/out, plus the initial embedding collapse).
        "hidden_states": OutputRecorder(DeepseekV41RMSNorm, layer_name="input_layernorm"),
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
        image_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        image_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask, `True` inside image spans (delimiter positions included). The
            VL wrapper ([`DeepseekV41Model`]) hands it over with the pre-merged
            `inputs_embeds`: image-span tokens switch the MoE router to its VL
            correction bias and are hashed as DEAD by the engram. `None` — the
            text-only path — leaves both at their pre-VL behavior.
        """
        # `image_mask` (`[B, S]` bool, True inside image spans) is handed over by the VL
        # wrapper: it switches the MoE router to its VL correction bias and, inverted,
        # marks the image-span tokens DEAD for the engram. Text-only callers leave it
        # `None` and run the exact pre-VL path.
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif input_ids is None and self.engram_layout:
            # The engram hashes token ids; there is no way to recover them from embeddings.
            raise ValueError("engram layers require `input_ids` (the hash state cannot consume `inputs_embeds`)")
        seq_len = inputs_embeds.shape[1]
        padding_mask = None
        source_mask = next(iter(attention_mask.values())) if isinstance(attention_mask, dict) else attention_mask
        if isinstance(source_mask, torch.Tensor):
            if source_mask.ndim == 2:
                padding_mask = source_mask[:, -seq_len:].bool()
            elif source_mask.ndim == 4:
                # A causal mask's current-token diagonal retains key liveness even
                # after padding has been folded into an additive mask.
                visible = source_mask if source_mask.dtype == torch.bool else source_mask == 0
                padding_mask = visible.diagonal(offset=source_mask.shape[-1] - seq_len, dim1=-2, dim2=-1).any(1)
            if padding_mask is not None:
                padding_mask = padding_mask.to(inputs_embeds.device).expand(inputs_embeds.shape[0], -1)
        if image_mask is not None:
            if image_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError(
                    "`image_mask` must describe the current input chunk, with shape (batch_size, sequence_length)."
                )
            image_mask = image_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        if position_ids is None:
            if isinstance(source_mask, torch.Tensor) and source_mask.ndim == 2:
                position_ids = (source_mask.long().cumsum(-1) - 1)[:, -seq_len:].to(inputs_embeds.device)
                position_ids = position_ids.masked_fill(~padding_mask, 0)
            else:
                past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
                # CODEPATH: DeepSeek-V4.1-Flash has KV sources [2, 8, 14, 20].
                # Custom sliding-only configs have none and use the physical cache length.
                if past_key_values is not None and self.config.kv_source_layer_ids:
                    source_cache = past_key_values.layers[self.config.kv_source_layer_ids[0]]
                    count = source_cache.token_count["compressor"]
                    if count is not None:
                        past_seen = count.to(inputs_embeds.device).unsqueeze(-1)
                if padding_mask is None:
                    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0) + past_seen
                else:
                    position_ids = (padding_mask.long().cumsum(-1) - 1 + past_seen).masked_fill(~padding_mask, 0)
                position_ids = position_ids.expand(inputs_embeds.shape[0], -1)
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
        if causal_mask is not None and causal_mask.dtype == torch.bool:
            causal_mask = torch.where(
                causal_mask,
                inputs_embeds.new_zeros(()),
                inputs_embeds.new_full((), torch.finfo(inputs_embeds.dtype).min),
            )

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        engram_rows: dict[int, torch.Tensor] = {}
        gate_mask = None
        if self.engram_hash_state is not None:
            # Padding and image spans are DEAD for n-gram hashing, but image spans
            # remain live tokens for attention and compression.
            gate_mask = padding_mask
            if image_mask is not None:
                gate_mask = ~image_mask if gate_mask is None else gate_mask & ~image_mask
            hash_ids = self.engram_hash_state(input_ids, gate_mask, past_key_values)
            # Gather each engram layer's rows here, where the tables live (host RAM under
            # `device_map`); the layers receive dense rows on their own device.
            for k, layer_idx in enumerate(self.config.engram_layer_ids):
                engram_rows[layer_idx] = self.engram_tables[str(layer_idx)](hash_ids[:, :, k, :])

        # `shared` carries the CSA2 group state between layers within ONE forward. It is
        # passed as a keyword and listed in `_skip_keys_device_placement` so accelerate's
        # hooks hand every layer the same dict instead of a per-layer copy (writes by a
        # source layer must be visible to its consumers); readers move what they use.
        shared: dict = {}
        # One-hot initial mix: the first site collapses stream 0 only.
        pre_mix = hidden_states.new_zeros(*hidden_states.shape[:-1], dtype=torch.float32)
        pre_mix[..., 0] = 1.0
        for layer in self.layers:
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                engram_rows.get(layer.layer_idx),
                gate_mask,
                shared=shared,
                image_mask=image_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                padding_mask=padding_mask,
                past_key_values=past_key_values,
            )
        # Final collapse with the last site's pre mix, then the shared norm.
        hidden_states = DeepseekV41DecoderLayer.hc_collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}
    # The released checkpoint's config.json is composite (top-level `quantization_config`,
    # `text_config`, `vision_config`) with `architectures: [DeepseekV41ForCausalLM]`, so this
    # class must accept the composite config: `get_hf_quantizer` only sees `quantization_config`
    # on the config produced from `config_class` — pointing it at the bare text config silently
    # dropped the FP8 quantization config and fp8 tensors then failed to load. The text config
    # is unwrapped in `__init__` (same pattern as `MllamaForCausalLM`; explicit bases rather
    # than `MixtralForCausalLM`, whose inlined `__init__` would drop the unwrap); a text config
    # passed directly is returned unchanged by `get_text_config()`.
    config_class = DeepseekV41Config

    def __init__(self, config):
        super().__init__(config.get_text_config())
        self.model = DeepseekV41TextModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.router_aux_loss_coef = self.config.router_aux_loss_coef
        self.num_experts = self.config.n_routed_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok
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
        output_router_logits: bool | None = None,
        shift_labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
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
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        # Only compute the logits that are needed, and do not upcast them unless the loss needs it
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None or shift_labels is not None:
            # `shift_labels` carries already-aligned targets (sequence / context
            # parallel training); `self.loss_function` shifts plain `labels` itself.
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, shift_labels=shift_labels
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


# =====================================================================================
# Vision tower, aligner and the image-text-to-text wrapper. The ViT is a close cousin
# of Qwen2-VL's: the same fused qkv attention (reused verbatim from
# `VisionAttention`), the same axial 2D RoPE — but per-patch row-major positions
# (Qwen2-VL orders patches by merge window), RMSNorm pre-norms instead of LayerNorm,
# and a SwiGLU MLP. The aligner plays Qwen2-VL's `PatchMerger`: 3×3 patch windows
# (zero-padded up to a multiple of 3) unfolded channel-major into one GELU MLP.
# =====================================================================================


class DeepseekV41VisionRMSNorm(nn.Module):
    """Vision RMSNorm with fp32 statistics and multiplication before casting back.

    The weight follows the model's load dtype; the calculation matches the reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(dtype)


class DeepseekV41VisionPatchEmbed(nn.Module):
    """Patch embedding of the vision tower: one `nn.Linear(3·patch_size², hidden)` on
    pre-flattened patches (channel-major `(c, ph, pw)` flattening — the layout the
    image processor emits, identical to a Conv2d weight viewed as a matrix)."""

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(1)  # [n_patches, 3, p, p] -> [n_patches, 3*p*p]
        return self.proj(hidden_states)


class DeepseekV41VisionRotaryEmbedding(nn.Module):
    """Axial 2D RoPE of the vision tower: `head_dim // 4` inverse frequencies, each
    patch position `(h, w)` contributing `h·inv_freq ⊕ w·inv_freq`. The frequencies of
    the two axes are concatenated then duplicated over the halves, so the standard
    `rotate_half` application (`apply_rotary_pos_emb_vision`) rotates the whole head —
    the same layout Qwen2-VL's vision rotary produces, and the same math as the
    reference's chunk-pair rotation."""

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.config = config
        self.inv_freq = nn.Buffer(self.compute_inv_freq(config), persistent=False)

    @staticmethod
    def compute_inv_freq(config: DeepseekV41VisionConfig) -> torch.Tensor:
        head_dim = config.hidden_size // config.num_attention_heads
        spatial_dim = head_dim // 2
        return 1.0 / (config.rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (n_patches, 2) — row 0 = h coords, row 1 = w coords, row-major per image
        freqs = position_ids[..., None].float() * self.inv_freq.float()  # [n, 2, head_dim // 4]
        cos, sin = freqs.cos(), freqs.sin()
        cos = torch.cat([cos[:, 0], cos[:, 1]], dim=-1)  # h-freqs | w-freqs -> [n, head_dim // 2]
        sin = torch.cat([sin[:, 0], sin[:, 1]], dim=-1)
        cos = torch.cat([cos, cos], dim=-1)  # duplicated halves, for rotate_half
        sin = torch.cat([sin, sin], dim=-1)
        return cos.to(x.dtype), sin.to(x.dtype)


class DeepseekV41VisionAttention(VisionAttention):
    """The ViT attention: Qwen2-VL's `VisionAttention` verbatim (fused `qkv` projection
    with bias, chunk-3 split, full bidirectional attention per image through the
    `cu_seqlens` splits, `proj` output projection with bias) — the reference's `wqkv` /
    `wo` under our module names."""

    def __init__(self, config: DeepseekV41VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads


class DeepseekV41VisionMLP(nn.Module):
    """SwiGLU MLP of the vision tower: `w1` is the fused `[gate; up]` projection (no
    bias, gate rows first), `w2` the down projection (no bias)."""

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(GradientCheckpointingLayer):
    """A ViT block: RMSNorm pre-norms (fp32 calculation, eps 1e-6), residual attention
    and MLP, exactly Qwen2-VL's block structure."""

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = DeepseekV41VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = DeepseekV41VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = DeepseekV41VisionAttention(config=config)
        self.mlp = DeepseekV41VisionMLP(config=config)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths — one segment per image, so attention never
            crosses an image boundary.
        position_embeddings (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
            `(cos, sin)` of the 2D RoPE, one entry per patch.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@auto_docstring
class DeepseekV41VisionModel(DeepseekV41PreTrainedModel):
    r"""
    The DeepSeek-V4.1 vision tower (DeepSeek-ViT): a plain ViT over the patch grid of
    ONE image at a time — the packed forward processes all images' patches at once, but
    attention is split per image (`cu_seqlens`) and each patch carries its own row-major
    2D-RoPE position. No merger: the aligner (`DeepseekV41Aligner`) lives one level up,
    next to the language model, because its output width is the *text* hidden size.
    """

    config: DeepseekV41VisionConfig
    config_class = DeepseekV41VisionConfig
    input_modalities = ("image",)
    # The tower is not the text backbone: SDPA works for its plain bidirectional
    # per-image attention (the eager-only restrictions of the base come from the text
    # attention's per-head sinks and shared-KV tricks, none of which apply here).
    _supports_sdpa = True
    _no_split_modules = ["DeepseekV41VisionBlock"]
    _can_record_outputs = {
        "hidden_states": DeepseekV41VisionBlock,
        "attentions": DeepseekV41VisionAttention,
    }

    def __init__(self, config: DeepseekV41VisionConfig):
        super().__init__(config)
        self.patch_embed = DeepseekV41VisionPatchEmbed(config)
        self.rotary_pos_emb = DeepseekV41VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = DeepseekV41VisionRMSNorm(config.hidden_size, eps=1e-6)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(total_patches, 3 * patch_size ** 2)`):
                The flattened patches of all images, concatenated in image order.
            grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The temporal, height and width dimensions of the patch grid of each
                image — `[1, n_vit_h, n_vit_w]` (temporal must be 1: images only).
        """
        # Row-major (h, w) position per patch — `spatial_merge_size=1` keeps the raster
        # order (Qwen2-VL instead orders patches by merge window).
        position_ids = get_vision_position_ids(grid_thw, 1, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=None)


class DeepseekV41Aligner(nn.Module):
    """The vision→text bridge (Qwen2-VL's `PatchMerger` role): one ViT image's features
    `[n_h·n_w, C]` are read back as a grid, zero-padded up to multiples of the 3×3
    downsample window, unfolded into `[ceil(n_h/3)·ceil(n_w/3), 9·C]` rows in
    `F.unfold`'s channel-major order, and mapped to `hidden_size` by a two-layer GELU
    MLP. One row = one LLM image token."""

    def __init__(self, config: DeepseekV41Config):
        super().__init__()
        # window side from the vision config, output width from the text config: the
        # aligner is the vision→text bridge, so it reads both off the composite config
        self.downsample_ratio = config.vision_config.downsample_ratio
        in_dim = config.vision_config.hidden_size * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.text_config.hidden_size, bias=True)
        self.w2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r"""`x`: one image's ViT features `[n_h·n_w, C]`; returns
        `[ceil(n_h/r)·ceil(n_w/r), hidden]` in reading order."""
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)  # [C, n_h, n_w]
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))  # zero-pad up to whole windows
        # F.unfold's channel-major (c, kh, kw) order decides w1's input layout
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


@auto_docstring
class DeepseekV41Model(DeepseekV41PreTrainedModel):
    r"""
    The image-text-to-text backbone of DeepSeek-V4.1, Qwen2-VL-shaped:
    [`DeepseekV41VisionModel`] (`visual`), the [`DeepseekV41Aligner`] and the three
    learned delimiter embeddings (`image_start` / `image_end` / `image_newline`) sit
    next to the text backbone (`language_model`), mirroring the reference's module
    layout. `get_image_features` runs the tower and the aligner per image;
    `merge_image_embeddings` scatters the rows into the image-span tokens.
    """

    config: DeepseekV41Config
    config_class = DeepseekV41Config
    # Native checkpoints store these BF16 modules without FP8 block scales.
    _keep_in_fp32_modules = [*DeepseekV41PreTrainedModel._keep_in_fp32_modules, "visual", "aligner"]

    def __init__(self, config: DeepseekV41Config):
        super().__init__(config)
        self.visual = DeepseekV41VisionModel._from_config(config.vision_config)
        self.aligner = DeepseekV41Aligner(config)
        # Learned embeddings of the image-span delimiters; the span layout in the text
        # stream is fixed, so the model scatters them structurally (no token ids).
        self.image_start = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.language_model = DeepseekV41TextModel._from_config(config.text_config)
        self.post_init()

    def post_init(self):
        super().post_init()
        # The text backbone's ignore list covers the vision / aligner / delimiter keys
        # (a text-only load of the full checkpoint must skip them); `post_init` merged
        # the children's lists back in — here those keys are EXPECTED, so drop them.
        self._keys_to_ignore_on_load_unexpected -= set(_VISION_ONLY_LOAD_PATTERNS)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * patch_size ** 2)`):
                The flattened patches of all images, concatenated in image order.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The `[1, n_vit_h, n_vit_w]` patch grid of each image.

        Returns:
            `BaseModelOutputWithPooling`: `pooler_output` is a tuple with one
            `[n_llm_h·n_llm_w, text_hidden]` aligner row-set per image, in image order;
            `last_hidden_state` the ViT features of all patches.
        """
        if image_grid_thw is None:
            raise ValueError("`image_grid_thw` is required with `pixel_values`.")
        if (image_grid_thw[:, 0] != 1).any():
            raise ValueError("DeepSeek-V4.1 accepts images only (`image_grid_thw` rows must be `[1, h, w]`).")
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        vit_features = vision_outputs.last_hidden_state
        features, start = [], 0
        for _, n_h, n_w in image_grid_thw.tolist():
            features.append(self.aligner(vit_features[start : start + n_h * n_w], n_h, n_w))
            start += n_h * n_w
        vision_outputs.pooler_output = tuple(features)
        return vision_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor | None, inputs_embeds: torch.FloatTensor | None = None
    ) -> torch.Tensor:
        """Find image placeholders from token IDs, or exact image-token embedding matches."""
        if input_ids is not None and (inputs_embeds is None or input_ids.shape == inputs_embeds.shape[:2]):
            return input_ids == self.config.image_token_id
        image_embedding = self.get_input_embeddings()(
            torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
        )
        return (inputs_embeds == image_embedding).all(-1)

    def _get_image_spans(
        self, image_mask: torch.Tensor, image_grid_thw: torch.LongTensor
    ) -> list[tuple[int, int, int, int]]:
        """Parse complete image spans in batch order, including adjacent images in one run."""
        # Transfer only run boundaries, not one scalar per token, from the accelerator.
        boundaries = torch.diff(F.pad(image_mask.to(torch.int8), (1, 1)), dim=-1).nonzero().tolist()
        grids = image_grid_thw.tolist()
        spans = []
        ratio = self.config.vision_config.downsample_ratio
        for (row, start), (_, end) in zip(boundaries[::2], boundaries[1::2]):
            while start < end:
                if len(spans) == len(grids):
                    raise ValueError("Image features and image tokens do not match: more spans than images.")
                temporal, height, width = grids[len(spans)]
                if temporal != 1 or height <= 0 or width <= 0:
                    raise ValueError(
                        "`image_grid_thw` must contain image grids `[1, positive_height, positive_width]`."
                    )
                height, width = -(-height // ratio), -(-width // ratio)
                length = height * (width + 1) + 2
                if start + length > end:
                    raise ValueError(
                        "Image features and image tokens do not match: an image span does not fit its placeholder run."
                    )
                spans.append((row, start, height, width))
                start += length
        if len(spans) != len(grids):
            raise ValueError("Image features and image tokens do not match: fewer spans than images.")
        return spans

    def merge_image_embeddings(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: tuple[torch.FloatTensor, ...],
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.Tensor]:
        """Overwrite every image-span position of `inputs_embeds`: the IMAGE slots take
        the aligner rows in reading order, the delimiters (`IMAGE_START`,
        one `IMAGE_NEWLINE` per row, `IMAGE_END`) their learned embeddings. Every span
        position carries `image_token_id` in `input_ids`, so the spans are the
        contiguous runs of that id; `image_grid_thw` lists the images in the same
        batch-major order. Returns the scattered `inputs_embeds` and the `[B, S]`
        `image_mask` (True on every span position, delimiters included)."""
        special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds).to(inputs_embeds.device)
        spans = self._get_image_spans(special_image_mask, image_grid_thw)
        if len(image_features) != len(spans):
            raise ValueError("Image features and image tokens do not match: one feature tensor is required per image.")
        if not spans:
            return inputs_embeds, special_image_mask
        image_slot_mask = torch.zeros_like(special_image_mask)
        delimiter_rows = []
        for (row, start, height, width), features in zip(spans, image_features):
            if features.shape != (height * width, inputs_embeds.shape[-1]):
                raise ValueError(
                    "Image features and image tokens do not match: aligned feature shape does not match its grid."
                )
            slots = [start + 1 + i * (width + 1) + j for i in range(height) for j in range(width)]
            image_slot_mask[row, slots] = True
            delimiter_rows.extend([self.image_start, *([self.image_newline] * height), self.image_end])
        features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_slot_mask.unsqueeze(-1), features)
        delimiter_mask = special_image_mask & ~image_slot_mask
        delimiter_embeds = torch.stack(delimiter_rows).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(delimiter_mask.unsqueeze(-1), delimiter_embeds)
        return inputs_embeds, special_image_mask

    @property
    def engram_hash_state(self):
        # `from_pretrained`'s auto-bind looks for the hash state through `base_model`;
        # the wrapper is that base model, so delegate to the text backbone.
        return self.language_model.engram_hash_state

    def bind_tokenizer(self, tokenizer):
        """Bind the tokenizer to the text backbone's engram hash state (the compressed
        token map — see [`DeepseekV41TextModel.bind_tokenizer`])."""
        self.language_model.bind_tokenizer(tokenizer)
        return self

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
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        image_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask, `True` inside image spans — normally derived from `pixel_values`
            (see [`DeepseekV41Model.merge_image_embeddings`]); passing it explicitly runs
            the same router / engram paths without the vision tower. It must describe
            the current input chunk, not the cached prefix.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Token embeddings before image replacement. Without `input_ids`, image
            placeholders are located by matching the image-token embedding. Models
            with Engram layers still require the real `input_ids` alongside these embeddings.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if input_ids is None and self.language_model.engram_layout:
            raise ValueError("engram layers require `input_ids` (the hash state cannot consume `inputs_embeds`)")
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is not None:
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                raise ValueError(
                    "Image inputs must be prefilled in one chunk — pass the image spans with the first forward "
                    "call (the reference asserts the same: `start_pos == 0`)."
                )
            image_features = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            ).pooler_output
            inputs_embeds, image_mask = self.merge_image_embeddings(
                input_ids, inputs_embeds, image_features, image_grid_thw
            )
        # input_ids still flows to the text model: the engram hash needs the token ids
        # even though the token stream comes pre-embedded.
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_mask=image_mask,
            **kwargs,
        )
        return outputs


@auto_docstring
class DeepseekV41ForConditionalGeneration(DeepseekV41PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}
    config_class = DeepseekV41Config
    _keep_in_fp32_modules = [*DeepseekV41PreTrainedModel._keep_in_fp32_modules, "model.visual", "model.aligner"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV41Model(config)
        text_config = config.get_text_config()
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.router_aux_loss_coef = text_config.router_aux_loss_coef
        self.num_experts = text_config.n_routed_experts
        self.num_experts_per_tok = text_config.num_experts_per_tok
        self.post_init()

    def post_init(self):
        super().post_init()
        # see DeepseekV41Model.post_init: the vision / aligner / delimiter keys are ours
        self._keys_to_ignore_on_load_unexpected -= set(_VISION_ONLY_LOAD_PATTERNS)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        image_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if inputs_embeds is not None and is_first_iteration and self.model.language_model.engram_layout:
            # Generation normally drops IDs when embeddings are supplied. Engram needs
            # the real IDs too, never an inferred/hash substitute for the embeddings.
            if input_ids.shape[1] != inputs_embeds.shape[1]:
                raise ValueError("engram layers require `input_ids` for every supplied input embedding.")
            model_inputs["input_ids"] = (
                input_ids[:, -next_sequence_length:] if next_sequence_length is not None else input_ids
            )
        if image_mask is not None:
            current = model_inputs.get("inputs_embeds")
            if current is None:
                current = model_inputs["input_ids"]
            model_inputs["image_mask"] = image_mask[:, -current.shape[1] :]
        if not is_first_iteration and kwargs.get("use_cache", True):
            model_inputs.pop("image_grid_thw", None)
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )
        if model_kwargs.get("image_mask") is not None:
            # Generated tokens are text, even when the prompt ends inside an image
            # span. Also retain prompt masks when generation recomputes without cache.
            model_kwargs["image_mask"] = F.pad(model_kwargs["image_mask"], (0, num_new_tokens), value=False)
        return model_kwargs

    def _expand_inputs_for_generation(
        self,
        expand_size: int | list[int] = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        if isinstance(expand_size, int) and expand_size == 1:
            return input_ids, model_kwargs
        inputs_embeds = model_kwargs.get("inputs_embeds")
        batch_input = inputs_embeds if inputs_embeds is not None else input_ids
        repeats = [expand_size] * batch_input.shape[0] if isinstance(expand_size, int) else list(expand_size)
        if len(repeats) != batch_input.shape[0] or any(repeat < 0 for repeat in repeats):
            raise ValueError("`expand_size` must provide a nonnegative repeat count for every batch row.")
        visual_inputs = {}
        grid = model_kwargs.pop("image_grid_thw", None)
        pixels = model_kwargs.pop("pixel_values", None)
        if pixels is not None and grid is None:
            raise ValueError("`image_grid_thw` is required with `pixel_values`.")
        if grid is not None:
            mask = self.model.get_placeholder_mask(input_ids, inputs_embeds)
            spans = self.model._get_image_spans(mask, grid)
            image_counts = [0] * batch_input.shape[0]
            patch_counts = [0] * batch_input.shape[0]
            for (row, _, _, _), (_, height, width) in zip(spans, grid.tolist()):
                image_counts[row] += 1
                patch_counts[row] += height * width
            for key, value, lengths in (
                ("image_grid_thw", grid, image_counts),
                ("pixel_values", pixels, patch_counts),
            ):
                if value is not None:
                    # Repeat each row's entire ordered image block, not individual
                    # patches or images. Text-only rows have zero-length blocks.
                    chunks = value.split(lengths)
                    visual_inputs[key] = torch.cat(
                        [chunk.repeat(repeat, *([1] * (value.ndim - 1))) for chunk, repeat in zip(chunks, repeats)]
                    )
        if isinstance(expand_size, int):
            input_ids, model_kwargs = super()._expand_inputs_for_generation(
                expand_size=expand_size,
                is_encoder_decoder=is_encoder_decoder,
                input_ids=input_ids,
                **model_kwargs,
            )
        else:
            row_indices = torch.arange(batch_input.shape[0], device=batch_input.device).repeat_interleave(
                torch.tensor(repeats, device=batch_input.device)
            )
            if input_ids is not None:
                input_ids = input_ids.index_select(0, row_indices.to(input_ids.device))
            for key, value in model_kwargs.items():
                if isinstance(value, torch.Tensor):
                    model_kwargs[key] = value.index_select(0, row_indices.to(value.device))
            if is_encoder_decoder:
                raise ValueError("DeepSeek-V4.1 generation uses a decoder-only text backbone.")
        model_kwargs.update(visual_inputs)
        return input_ids, model_kwargs

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
        output_router_logits: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_mask: torch.Tensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        image_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask, `True` inside image spans — normally derived from `pixel_values`
            (see [`DeepseekV41Model.merge_image_embeddings`]); passing it explicitly runs
            the same router / engram paths without the vision tower.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `-100` (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored, masked out of the loss computation, and the loss is computed over tokens with labels in
            `[0, ..., config.vocab_size]`.
        shift_labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Already-shifted next-token targets, aligned with `logits` (used for sequence and context
            parallel training, where the shift must happen before sharding). When given, they take
            precedence over `labels` for the loss.
        """
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.get_text_config().output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_mask=image_mask,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        # Only compute the logits that are needed, and do not upcast them unless the loss needs it
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None or shift_labels is not None:
            # `shift_labels` carries already-aligned targets (sequence / context
            # parallel training); `self.loss_function` shifts plain `labels` itself.
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
                shift_labels=shift_labels,
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


# =====================================================================================
# Image processing: the resize plan of the reference `image_processor.py` — upscale to
# `min_pixels`, round both sides up to whole patches, shrink until the image costs at
# most `max_image_tokens` LLM tokens, then aspect-fit into the planned box and gray-pad
# the remainder (very wide images are squeezed instead). Shared by both backends.
# =====================================================================================


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Tokens one image costs in the text stream: the aligner grid in reading order,
    one IMAGE_NEW_LINE per row, wrapped in IMAGE_START / IMAGE_END."""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    """The token grid the aligner produces from a patch grid of this pixel size."""
    return (
        math.ceil((best_height // patch_size) / downsample_ratio),
        math.ceil((best_width // patch_size) / downsample_ratio),
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """The largest aspect-preserving pixel size whose token grid still fits in
    `max_n_token` (the reference's closed-form solution)."""
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most `max_n_token` LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError(f"The resize plan cannot fit the image in {max_n_token} image tokens.")
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width, height, patch_size, downsample_ratio, min_pixels, max_n_token, max_wh_ratio=None):
    """Resize plan for an image of the given original size — a pure function of its
    arguments. Returns `(n_llm_h, n_llm_w, best_height, best_width)`."""
    if width <= 0 or height <= 0 or patch_size <= 0 or downsample_ratio <= 0:
        raise ValueError("Image dimensions, patch_size and downsample_ratio must be positive.")
    if min_pixels <= 0 or max_n_token < 4:
        raise ValueError("min_pixels must be positive and max_image_tokens must be at least 4.")
    if max_wh_ratio is not None and max_wh_ratio <= 0:
        raise ValueError("max_wh_ratio must be positive or None.")
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    return safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token)


def contain_size(height, width, box_height, box_width):
    """The aspect-preserving fit inside the box, with PIL `ImageOps.contain`'s rounding
    (one side exact, the other rounded) — the fit both backends apply before padding."""
    image_ratio, box_ratio = width / height, box_width / box_height
    if image_ratio > box_ratio:
        return round(height / width * box_width), box_width
    if image_ratio < box_ratio:
        return box_height, round(width / height * box_height)
    return box_height, box_width


class DeepseekV41ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `14`):
        Spatial patch size of the vision encoder.
    downsample_ratio (`int`, *optional*, defaults to `3`):
        The aligner's window side: `downsample_ratio²` ViT patches merge into one LLM token.
    min_pixels (`int`, *optional*, defaults to `295936`):
        Minimum number of pixels per image (544²); smaller images are upscaled. Carried
        in `size["shortest_edge"]`.
    max_image_tokens (`int`, *optional*, defaults to `1024`):
        Maximum number of image tokens per image; larger images are shrunk until they
        fit. Carried in `size["longest_edge"]`.
    max_wh_ratio (`int`, *optional*):
        Maximum width / height aspect ratio kept by the resize plan; `None` (the
        default) disables the cap. Images wider than the cap are squeezed into the
        planned box, aspect ratio not preserved.
    """

    patch_size: int
    downsample_ratio: int
    min_pixels: int
    max_image_tokens: int
    max_wh_ratio: int | None


@auto_docstring
class DeepseekV41ImageProcessor(TorchvisionBackend):
    r"""
    Builds DeepSeek-V4.1 ViT patches. `size` carries the two plan knobs —
    `size["shortest_edge"]` is the minimum pixel budget (`min_pixels`, 544²) and
    `size["longest_edge"]` the maximum number of image tokens (`max_image_tokens`). The
    plan upscales to the minimum budget, rounds both sides up to whole patches, shrinks
    under the token budget, then aspect-fits into the planned box and gray-pads
    (`127`) the remainder, like the reference's `ImageOps.pad`. The output is
    `pixel_values` `[total_patches, 3·patch_size²]` (row-major patch order, flattened
    channel-major) plus `image_grid_thw` `[n_images, 3]` = `[1, n_vit_h, n_vit_w]`.
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 295936, "longest_edge": 1024}
    do_rescale = True
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 14
    downsample_ratio = 3
    max_wh_ratio = None
    pad_value = 127  # the reference's gray ImageOps.pad fill
    valid_kwargs = DeepseekV41ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[DeepseekV41ImageProcessorKwargs]):
        # `min_pixels` / `max_image_tokens` override the corresponding `size` entries
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else dict(size)
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
        if (max_image_tokens := kwargs.pop("max_image_tokens", None)) is not None:
            size["longest_edge"] = max_image_tokens
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_image_tokens: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None or max_image_tokens is not None:
            size = dict(self.size) if size is None else dict(size)
            if min_pixels is not None:
                size["shortest_edge"] = min_pixels
            if max_image_tokens is not None:
                size["longest_edge"] = max_image_tokens
        return super()._standardize_kwargs(size=size, **kwargs)

    def resize(
        self,
        images: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        patch_size: int | None = None,
        downsample_ratio: int | None = None,
        max_wh_ratio: int | None = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize to the DeepSeek-V4.1 plan (see the class docstring) and pad the
        aspect-fit remainder with `pad_value`."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(
                f"`size` must carry 'shortest_edge' (min_pixels) and 'longest_edge' (max_image_tokens) but got {size}."
            )
        patch_size = patch_size if patch_size is not None else self.patch_size
        downsample_ratio = downsample_ratio if downsample_ratio is not None else self.downsample_ratio
        height, width = images.shape[-2:]
        _, _, best_height, best_width = plan_image_grid(
            width, height, patch_size, downsample_ratio, size.shortest_edge, size.longest_edge, max_wh_ratio
        )
        if max_wh_ratio is not None and width >= max_wh_ratio * height:
            # very wide: squeeze into the planned box, aspect ratio not preserved
            return super().resize(images, SizeDict(height=best_height, width=best_width), resample)
        fitted_height, fitted_width = contain_size(height, width, best_height, best_width)
        images = super().resize(images, SizeDict(height=fitted_height, width=fitted_width), resample)
        top = round((best_height - fitted_height) * 0.5)
        left = round((best_width - fitted_width) * 0.5)
        return tvF.pad(
            images,
            [left, top, best_width - fitted_width - left, best_height - fitted_height - top],
            fill=self.pad_value,
        )

    def _prepare_images_structure(self, images, expected_ndims=3):
        if isinstance(images, (list, tuple)) and images:
            return [
                image
                for item in images
                for image in self._prepare_images_structure(item, expected_ndims=expected_ndims)
            ]
        # The backend already adds a channel axis to individual grayscale arrays;
        # preserve them until then instead of rejecting them as malformed batches.
        ndim = 2 if getattr(images, "ndim", None) == 2 else expected_ndims
        return super()._prepare_images_structure(images, expected_ndims=ndim)

    def process_image(self, image, do_convert_rgb=None, input_data_format=None, **kwargs):
        if is_pil_image(image):
            input_data_format = ChannelDimension.FIRST
        elif image.ndim == 2:
            input_data_format = ChannelDimension.FIRST
        elif input_data_format is None:
            input_data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))
        image = super().process_image(
            image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, **kwargs
        )
        if do_convert_rgb:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] == 4:
                image = image[:3]
        return image

    def patchify(self, images: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        """`[B, C, H, W]` -> `[B, n_h·n_w, C·patch_size²]`: row-major patch order, each
        patch flattened channel-major `(c, ph, pw)` — the `patch_embed.proj` layout."""
        batch, channel, height, width = images.shape
        if height % patch_size or width % patch_size:
            raise ValueError("Image height and width must be divisible by patch_size when do_resize=False.")
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = images.reshape(batch, channel, grid_h, patch_size, grid_w, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # [B, gh, gw, C, ph, pw]
        return patches.reshape(batch, grid_h * grid_w, channel * patch_size * patch_size)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        downsample_ratio: int,
        max_wh_ratio: int | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    images=stacked_images,
                    size=size,
                    resample=resample,
                    patch_size=patch_size,
                    downsample_ratio=downsample_ratio,
                    max_wh_ratio=max_wh_ratio,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = self.patchify(stacked_images, patch_size=patch_size)
            processed_images_grouped[shape] = patches
            processed_grids[shape] = [
                [1, stacked_images.shape[-2] // patch_size, stacked_images.shape[-1] // patch_size]
            ] * stacked_images.shape[0]

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
        pixel_values = processed_images[0] if len(processed_images) == 1 else torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns the number of ViT patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of ViT patches per image.
        """
        grid_h, grid_w = self.get_image_grid(height, width, images_kwargs)
        return grid_h * grid_w

    def get_image_grid(self, height: int, width: int, images_kwargs: dict | None = None) -> tuple[int, int]:
        """Return the ViT grid for an input size, using the same overrides as preprocessing."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        if not images_kwargs.get("do_resize", self.do_resize):
            if height % patch_size or width % patch_size:
                raise ValueError("Image height and width must be divisible by patch_size when do_resize=False.")
            return height // patch_size, width // patch_size
        size = images_kwargs.get("size", self.size)
        min_pixels = images_kwargs.get("min_pixels")
        max_image_tokens = images_kwargs.get("max_image_tokens")
        _, _, best_height, best_width = plan_image_grid(
            width,
            height,
            patch_size,
            images_kwargs.get("downsample_ratio", self.downsample_ratio),
            size["shortest_edge"] if min_pixels is None else min_pixels,
            size["longest_edge"] if max_image_tokens is None else max_image_tokens,
            images_kwargs.get("max_wh_ratio", self.max_wh_ratio),
        )
        return best_height // patch_size, best_width // patch_size


@auto_docstring
class DeepseekV41ImageProcessorPil(PilBackend):
    r"""
    PIL/NumPy backend of [`DeepseekV41ImageProcessor`] — identical preprocessing
    (plan, gray pad, patch layout), portable to environments without torchvision. The
    constructor signature and defaults match the torchvision class exactly.
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 295936, "longest_edge": 1024}
    do_rescale = True
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 14
    downsample_ratio = 3
    max_wh_ratio = None
    pad_value = 127
    valid_kwargs = DeepseekV41ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[DeepseekV41ImageProcessorKwargs]):
        # `min_pixels` / `max_image_tokens` override the corresponding `size` entries
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else dict(size)
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
        if (max_image_tokens := kwargs.pop("max_image_tokens", None)) is not None:
            size["longest_edge"] = max_image_tokens
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_image_tokens: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None or max_image_tokens is not None:
            size = dict(self.size) if size is None else dict(size)
            if min_pixels is not None:
                size["shortest_edge"] = min_pixels
            if max_image_tokens is not None:
                size["longest_edge"] = max_image_tokens
        return super()._standardize_kwargs(size=size, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        patch_size: int | None = None,
        downsample_ratio: int | None = None,
        max_wh_ratio: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize to the DeepSeek-V4.1 plan (see the class docstring) and pad the
        aspect-fit remainder with `pad_value`."""
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(
                f"`size` must carry 'shortest_edge' (min_pixels) and 'longest_edge' (max_image_tokens) but got {size}."
            )
        patch_size = patch_size if patch_size is not None else self.patch_size
        downsample_ratio = downsample_ratio if downsample_ratio is not None else self.downsample_ratio
        height, width = image.shape[-2:]
        _, _, best_height, best_width = plan_image_grid(
            width, height, patch_size, downsample_ratio, size.shortest_edge, size.longest_edge, max_wh_ratio
        )
        if max_wh_ratio is not None and width >= max_wh_ratio * height:
            return super().resize(image, SizeDict(height=best_height, width=best_width), resample)
        fitted_height, fitted_width = contain_size(height, width, best_height, best_width)
        image = super().resize(image, SizeDict(height=fitted_height, width=fitted_width), resample)
        top = round((best_height - fitted_height) * 0.5)
        left = round((best_width - fitted_width) * 0.5)
        return np.pad(
            image,
            ((0, 0), (top, best_height - fitted_height - top), (left, best_width - fitted_width - left)),
            mode="constant",
            constant_values=self.pad_value,
        )

    def _prepare_images_structure(self, images, expected_ndims=3):
        if isinstance(images, (list, tuple)) and images:
            return [
                image
                for item in images
                for image in self._prepare_images_structure(item, expected_ndims=expected_ndims)
            ]
        ndim = 2 if getattr(images, "ndim", None) == 2 else expected_ndims
        return super()._prepare_images_structure(images, expected_ndims=ndim)

    def process_image(self, image, do_convert_rgb=None, input_data_format=None, **kwargs):
        if is_pil_image(image):
            input_data_format = None
        elif image.ndim == 2:
            input_data_format = ChannelDimension.FIRST
        elif input_data_format is None:
            input_data_format = infer_channel_dimension_format(image, num_channels=(1, 3, 4))
        image = super().process_image(
            image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, **kwargs
        )
        if do_convert_rgb:
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] == 4:
                image = image[:3]
        return image

    def patchify(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """`[C, H, W]` -> `[n_h·n_w, C·patch_size²]`: row-major patch order, each patch
        flattened channel-major `(c, ph, pw)` — the `patch_embed.proj` layout."""
        channel, height, width = image.shape
        if height % patch_size or width % patch_size:
            raise ValueError("Image height and width must be divisible by patch_size when do_resize=False.")
        grid_h, grid_w = height // patch_size, width // patch_size
        patches = image.reshape(channel, grid_h, patch_size, grid_w, patch_size)
        patches = np.transpose(patches, (1, 3, 0, 2, 4))  # [gh, gw, C, ph, pw]
        return patches.reshape(grid_h * grid_w, channel * patch_size * patch_size)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        downsample_ratio: int,
        max_wh_ratio: int | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        all_patches, all_grids = [], []
        for image in images:
            if do_resize:
                image = self.resize(
                    image,
                    size=size,
                    resample=resample,
                    patch_size=patch_size,
                    downsample_ratio=downsample_ratio,
                    max_wh_ratio=max_wh_ratio,
                )
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            all_patches.append(self.patchify(image, patch_size=patch_size))
            all_grids.append([1, image.shape[-2] // patch_size, image.shape[-1] // patch_size])

        pixel_values = np.concatenate(all_patches, axis=0)
        image_grid_thw = np.array(all_grids, dtype=np.int64)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns the number of ViT patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of ViT patches per image.
        """
        grid_h, grid_w = self.get_image_grid(height, width, images_kwargs)
        return grid_h * grid_w

    def get_image_grid(self, height: int, width: int, images_kwargs: dict | None = None) -> tuple[int, int]:
        """Return the ViT grid for an input size, using the same overrides as preprocessing."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        if not images_kwargs.get("do_resize", self.do_resize):
            if height % patch_size or width % patch_size:
                raise ValueError("Image height and width must be divisible by patch_size when do_resize=False.")
            return height // patch_size, width // patch_size
        size = images_kwargs.get("size", self.size)
        min_pixels = images_kwargs.get("min_pixels")
        max_image_tokens = images_kwargs.get("max_image_tokens")
        _, _, best_height, best_width = plan_image_grid(
            width,
            height,
            patch_size,
            images_kwargs.get("downsample_ratio", self.downsample_ratio),
            size["shortest_edge"] if min_pixels is None else min_pixels,
            size["longest_edge"] if max_image_tokens is None else max_image_tokens,
            images_kwargs.get("max_wh_ratio", self.max_wh_ratio),
        )
        return best_height // patch_size, best_width // patch_size


class DeepseekV41ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: DeepseekV41ImageProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


@auto_docstring
class DeepseekV41Processor(ProcessorMixin):
    r"""
    Processor of DeepSeek-V4.1 (image-text-to-text). The tokenizer needs ONE image
    placeholder token whose id matches the model config's `image_token_id` (129264);
    each placeholder in the text expands to the full image span
    `[IMAGE_START] + ([IMAGE]·n_llm_w + [IMAGE_NEW_LINE])·n_llm_h + [IMAGE_END]`,
    every position carrying that same token id — the model tells the positions apart
    from `image_grid_thw`, so no per-position token ids are required.
    """

    valid_processor_kwargs = DeepseekV41ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token=None,
        image_token_id=None,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*):
            Existing tokenizer token used for image placeholders. Defaults to the tokenizer's
            `image_token` when defined, otherwise the released spelling `<｜deepseek_image｜>`.
        image_token_id (`int`, *optional*):
            Expected ID of `image_token`. When supplied, it must match the tokenizer.
        """
        self.image_token = (
            getattr(tokenizer, "image_token", None) or "<｜deepseek_image｜>" if image_token is None else image_token
        )
        resolved_id = tokenizer.convert_tokens_to_ids(self.image_token)
        if (
            resolved_id is None
            or resolved_id == tokenizer.unk_token_id
            or tokenizer.convert_ids_to_tokens(resolved_id) != self.image_token
            or tokenizer.encode(self.image_token, add_special_tokens=False) != [resolved_id]
            or tokenizer.encode(self.image_token * 2, add_special_tokens=False) != [resolved_id, resolved_id]
        ):
            raise ValueError(f"The tokenizer must encode {self.image_token!r} as one existing image token.")
        if image_token_id is not None and image_token_id != resolved_id:
            raise ValueError(f"image_token_id={image_token_id} does not match {self.image_token!r} ({resolved_id}).")
        self.image_token_id = resolved_id
        # Do not register the image token as special: the released tokenizer deliberately
        # keeps it as a regular added token, including when decoding with skip_special_tokens.
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        """The expansion of one image placeholder: the whole image span in the same
        image token (the model rebuilds the layout from `image_grid_thw`)."""
        _, n_vit_h, n_vit_w = image_inputs["image_grid_thw"][image_idx]
        n_vit_h, n_vit_w = int(n_vit_h), int(n_vit_w)
        ratio = kwargs.get("downsample_ratio", self.image_processor.downsample_ratio)
        n_llm_h, n_llm_w = -(-n_vit_h // ratio), -(-n_vit_w // ratio)
        return self.image_token * num_image_tokens(n_llm_h, n_llm_w)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Count image-span tokens and ViT patches for input sizes given as `(height, width)`."""
        if image_sizes is None:
            return MultiModalData()
        images_kwargs = {**kwargs, **kwargs.get("images_kwargs", {})}
        ratio = images_kwargs.get("downsample_ratio", self.image_processor.downsample_ratio)
        grids = [self.image_processor.get_image_grid(height, width, images_kwargs) for height, width in image_sizes]
        return MultiModalData(
            num_image_tokens=[num_image_tokens(-(-height // ratio), -(-width // ratio)) for height, width in grids],
            num_image_patches=[height * width for height, width in grids],
        )

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape
                `(batch_size, sequence_length)` or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's
                `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces in the output. Argument passed to the
                tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = [
    "DeepseekV41PreTrainedModel",
    "DeepseekV41TextModel",
    "DeepseekV41ForCausalLM",
    "DeepseekV41VisionModel",
    "DeepseekV41Model",
    "DeepseekV41ForConditionalGeneration",
    "DeepseekV41ImageProcessor",
    "DeepseekV41ImageProcessorPil",
    "DeepseekV41Processor",
    "DeepseekV41CSACache",
    "DeepseekV41EngramEmbedding",
    "EngramLayout",
    "DeepseekV41NgramHashState",
]
