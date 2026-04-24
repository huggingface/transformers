# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""DeepSeek V4 model — modular source.

Architecture overview (vs DeepSeek V3/V3.2):
    * Attention: MLA is gone. Replaced by sliding-window attention with an optional
      per-layer KV Compressor (learned gated pooling) and an Indexer that selects
      top-k compressed positions for long-range attention.
    * Residuals: replaced by Hyper-Connections — ``hc_mult`` parallel state copies
      mixed via a Sinkhorn-normalized pre/post transform. Always on.
    * MoE routing: Mixtral-style top-k with no expert groups. First ``num_hash_layers``
      layers use static hash routing keyed by input token id.
    * No MTP in this file (added elsewhere).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralPreTrainedModel,
    load_balancing_loss_func,
)


logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(PreTrainedConfig):
    r"""
    compress_ratios (`list[int]`):
        Per-layer compression schedule. ``0`` means pure local SWA (no Compressor).
        ``4`` enables overlapping-window compression plus an Indexer selecting top-k
        compressed positions. ``128`` enables disjoint-window compression without
        an Indexer (all compressed positions attended). Length must equal
        ``num_hidden_layers + num_nextn_predict_layers``; the trailing MTP entries
        are kept to match the upstream checkpoint layout but are not instantiated.
    compress_rope_theta (`float`):
        RoPE base used on layers that carry a Compressor (paired with YaRN).
        Layers with ``compress_ratios[i] == 0`` use ``rope_theta`` without YaRN.
    hc_mult (`int`, defaults to 4):
        Number of parallel Hyper-Connection streams. Always active.
    num_hash_layers (`int`, defaults to 3):
        First N layers route experts via a frozen ``tid2eid`` lookup keyed by input
        token id; the learned gate weights are still used to weight activations.
    scoring_func (`str`, defaults to "sqrtsoftplus"):
        One of ``"sqrtsoftplus"``, ``"softmax"``, ``"sigmoid"`` — applied to gate
        logits before routing.
    swiglu_limit (`float`):
        Optional clipping of SwiGLU gate/up pre-activations for routed experts.
    o_groups (`int`):
        Number of groups in the grouped low-rank output projection.
    o_lora_rank (`int`):
        Rank of each group in the grouped low-rank output projection.
    index_n_heads (`int`, defaults to 64):
        Number of heads used by the sparse-attention Indexer.
    index_head_dim (`int`, defaults to 128):
        Head dimension of the Indexer's query/key projections.
    index_topk (`int`):
        Number of compressed positions the Indexer selects per query token.
    hc_sinkhorn_iters (`int`, defaults to 20):
        Iteration count for the Sinkhorn normalisation inside Hyper-Connections.
    hc_eps (`float`, defaults to 1e-6):
        Numerical floor used both in the HC norm and the Sinkhorn normalisation.
    num_nextn_predict_layers (`int`, defaults to 1):
        Number of Multi-Token-Prediction layers present in the checkpoint. The
        weights are ignored on load — MTP is implemented elsewhere.
    rope_theta (`float`, defaults to 10000.0):
        Base period for the main-attention rotary embedding.
    rope_scaling (`dict`, *optional*):
        Optional YaRN/long-context scaling dict (same schema as other HF models).
        Applied to the main-attention and Compressor rotary embeddings.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    num_hash_layers: int = 3
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    compress_ratios: list[int] | None = None
    compress_rope_theta: float = 160000.0
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    num_nextn_predict_layers: int = 1
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_parameters: RopeParameters | dict | None = None
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        total_layers = self.num_hidden_layers + (self.num_nextn_predict_layers or 0)
        if self.compress_ratios is None:
            # Reasonable default: alternate 4 / 128 with a leading SWA layer and a trailing SWA layer.
            self.compress_ratios = [0] + [4 if (i % 2) else 128 for i in range(total_layers - 2)] + [0]
        if len(self.compress_ratios) != total_layers:
            raise ValueError(
                f"`compress_ratios` must have length num_hidden_layers + num_nextn_predict_layers "
                f"({total_layers}); got {len(self.compress_ratios)}."
            )
        if self.layer_types is None:
            # Every layer carries a 128-token sliding window; Compressor/Indexer presence is
            # driven by `compress_ratios` and resolved at module __init__ time.
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        for ratio in self.compress_ratios:
            if ratio not in (0, 4, 128):
                raise ValueError(f"Unsupported compress_ratio={ratio}. Expected 0, 4, or 128.")
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        super().__post_init__(**kwargs)


# -----------------------------------------------------------------------------
# Norms / RoPE
# -----------------------------------------------------------------------------


class DeepseekV4RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV4RotaryEmbedding(nn.Module):
    """Rotary embedding over `qk_rope_head_dim` (the rope-carved sub-range of each head).

    Supports an optional ``rope_scaling={"type": "yarn", ...}`` config — same knobs as the
    rest of transformers — and an override ``rope_theta`` kwarg so a Compressor can build
    a dedicated long-range embedding.
    """

    def __init__(self, config: "DeepseekV4Config", rope_theta: float | None = None, rope_scaling: dict | None = None):
        super().__init__()
        self.rope_head_dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = rope_theta if rope_theta is not None else config.rope_theta
        self.rope_scaling = rope_scaling if rope_scaling is not None else config.rope_scaling
        self.attention_scaling = 1.0
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.rope_head_dim, 2, dtype=torch.float32) / self.rope_head_dim)
        )
        if self.rope_scaling is not None and self.rope_scaling.get("type") == "yarn":
            inv_freq, self.attention_scaling = _yarn_inv_freq(
                inv_freq,
                self.rope_head_dim,
                self.rope_theta,
                factor=self.rope_scaling["factor"],
                original_max_pos=self.rope_scaling.get(
                    "original_max_position_embeddings", self.max_position_embeddings
                ),
                beta_fast=self.rope_scaling.get("beta_fast", 32),
                beta_slow=self.rope_scaling.get("beta_slow", 1),
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        freqs = (inv_freq @ positions).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def _yarn_inv_freq(base_inv_freq, dim, theta, factor, original_max_pos, beta_fast, beta_slow):
    import math

    def _correction(rotations):
        return dim * math.log(original_max_pos / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(int(math.floor(_correction(beta_fast))), 0)
    high = min(int(math.ceil(_correction(beta_slow))), dim - 1)
    if low == high:
        high += 1
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    smooth = 1.0 - ramp
    inv_freq = base_inv_freq / factor * (1 - smooth) + base_inv_freq * smooth
    # YaRN attention scaling (mscale)
    attention_scaling = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return inv_freq, attention_scaling


# -----------------------------------------------------------------------------
# Compressor — learned gated pooling over `compress_ratio` consecutive tokens.
# -----------------------------------------------------------------------------


class DeepseekV4Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one compressed KV token.

    The pooling weights are softmax-normalised per window:

        weights = softmax(wgate(x) + ape, dim=window)
        pooled  = sum(weights * wkv(x), dim=window)

    When ``compress_ratio == 4`` we use overlapping windows so every token contributes
    to two pooled outputs (except the first/last); ``compress_ratio == 128`` uses
    disjoint windows. A per-batch state buffer accumulates tokens during autoregressive
    decode and emits a pooled vector whenever a window closes.
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int, rope_theta: float):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.hidden_size = config.hidden_size
        self.overlap = compress_ratio == 4

        self.wkv = nn.Linear(self.hidden_size, head_dim, bias=False)
        self.wgate = nn.Linear(self.hidden_size, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, head_dim))
        self.kv_norm = DeepseekV4RMSNorm(head_dim, eps=config.rms_norm_eps)

        # Dedicated RoPE for the compressed segment (may carry a different theta + YaRN scaling
        # than the main attention; the compressed segment is the long-range path).
        rope_scaling = config.rope_scaling if rope_theta == config.compress_rope_theta else None
        self.rope = DeepseekV4RotaryEmbedding(config, rope_theta=rope_theta, rope_scaling=rope_scaling)

        # Lazy decode state (initialised on first decode step to match batch/dtype/device).
        self._state_kv: torch.Tensor | None = None
        self._state_gate: torch.Tensor | None = None
        self._state_pos: int = 0

    @torch.no_grad()
    def reset_state(self):
        self._state_kv = None
        self._state_gate = None
        self._state_pos = 0

    def _compute_windows(self, kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # kv, gate: [B, W, head_dim] with W multiple of compress_ratio
        batch, length, dim = kv.shape
        r = self.compress_ratio
        kv = kv.view(batch, length // r, r, dim)
        gate = gate.view(batch, length // r, r, dim) + self.ape
        weights = gate.softmax(dim=2)
        return (kv * weights).sum(dim=2)  # [B, length/r, head_dim]

    def _apply_rope(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # kv: [B, N, head_dim]; rotate last rope_head_dim dims.
        rope_dim = self.rope_head_dim
        head_part = kv[..., -rope_dim:].unsqueeze(2)  # [B, N, 1, rope_dim]
        nope_part = kv[..., :-rope_dim]
        cos, sin = self.rope(head_part, position_ids=positions)
        # apply_rotary_pos_emb expects q/k with shape [B, H, S, D]; mimic by transposing.
        head_part = head_part.transpose(1, 2)  # [B, 1, N, rope_dim]
        k_empty = torch.zeros_like(head_part)
        rot_part, _ = apply_rotary_pos_emb(head_part, k_empty, cos, sin, unsqueeze_dim=1)
        rot_part = rot_part.transpose(1, 2).squeeze(2)  # [B, N, rope_dim]
        return torch.cat([nope_part, rot_part], dim=-1)

    def forward(self, hidden_states: torch.Tensor, start_pos: int = 0) -> torch.Tensor | None:
        """Returns the compressed KV tensor (or ``None`` if no new compressed token was emitted).

        Prefill path (``start_pos == 0``): pools every complete window in the input;
        partial trailing tokens are stashed for the next call.
        Decode path (``start_pos > 0``): accumulates one token at a time; emits when a
        window closes.
        """
        r = self.compress_ratio
        batch, seq_len, _ = hidden_states.shape
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)

        if start_pos == 0:
            usable = (seq_len // r) * r
            if usable == 0:
                # Not enough tokens to emit anything yet — just stash the remainder.
                self._state_kv = kv
                self._state_gate = gate
                self._state_pos = seq_len
                return None
            pooled = self._compute_windows(kv[:, :usable], gate[:, :usable])
            remainder = seq_len - usable
            if remainder:
                self._state_kv = kv[:, usable:]
                self._state_gate = gate[:, usable:]
            else:
                self._state_kv = kv.new_zeros(batch, 0, self.head_dim)
                self._state_gate = gate.new_zeros(batch, 0, self.head_dim)
            self._state_pos = seq_len
            positions = torch.arange(pooled.shape[1], device=pooled.device) * r
            positions = positions.unsqueeze(0).expand(batch, -1)
            pooled = self.kv_norm(pooled)
            return self._apply_rope(pooled, positions)

        # Decode: append single token to state buffer.
        self._state_kv = kv if self._state_kv is None else torch.cat([self._state_kv, kv], dim=1)
        self._state_gate = gate if self._state_gate is None else torch.cat([self._state_gate, gate], dim=1)
        self._state_pos += seq_len
        if self._state_kv.shape[1] < r:
            return None
        pooled = self._compute_windows(self._state_kv[:, :r], self._state_gate[:, :r])
        self._state_kv = self._state_kv[:, r:]
        self._state_gate = self._state_gate[:, r:]
        positions = torch.full((batch, 1), self._state_pos - r, device=pooled.device, dtype=torch.long)
        pooled = self.kv_norm(pooled)
        return self._apply_rope(pooled, positions)


# -----------------------------------------------------------------------------
# Indexer — selects top-k compressed positions per query.
# -----------------------------------------------------------------------------


class DeepseekV4Indexer(nn.Module):
    """Scores compressed KV positions with a learned per-head weighted dot product.

    Owns its own ``DeepseekV4Compressor`` (reference uses a Hadamard-rotated fp4
    variant; the transform is orthogonal so bf16/fp32 scoring is equivalent).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        # Indexer's compressor uses compress_ratio=4 and shares the compressed rope theta.
        self.compressor = DeepseekV4Compressor(
            config, compress_ratio=4, head_dim=self.head_dim, rope_theta=config.compress_rope_theta
        )

        self._cached_kv: torch.Tensor | None = None

    @torch.no_grad()
    def reset_state(self):
        self.compressor.reset_state()
        self._cached_kv = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        start_pos: int,
    ) -> torch.LongTensor | None:
        batch, seq_len, _ = hidden_states.shape
        new_kv = self.compressor(hidden_states, start_pos=start_pos)
        if new_kv is not None:
            if self._cached_kv is None or start_pos == 0:
                self._cached_kv = new_kv
            else:
                self._cached_kv = torch.cat([self._cached_kv, new_kv], dim=1)
        if self._cached_kv is None or self._cached_kv.shape[1] == 0:
            return None

        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim)
        cos, sin = position_embeddings
        # Apply rope to last rope_head_dim dims of q.
        q_rope = q[..., -self.rope_head_dim :].transpose(1, 2)
        k_empty = torch.zeros_like(q_rope)
        q_rope, _ = apply_rotary_pos_emb(q_rope, k_empty, cos, sin, unsqueeze_dim=1)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope.transpose(1, 2)], dim=-1)

        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]

        # Score: ReLU(q · k^T) * weights → [B, S, T]
        scores = torch.matmul(q.float(), self._cached_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices  # [B, S, topk]


# -----------------------------------------------------------------------------
# Attention — sliding window + optional compressor + optional indexer + sink.
# -----------------------------------------------------------------------------


def _eager_attention_with_sink(
    module: "DeepseekV4Attention",
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,  # [B, 1, T, D]
    value: torch.Tensor,  # [B, 1, T, D]
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Broadcast single-head KV across query heads.
    key = key.expand(-1, query.shape[1], -1, -1)
    value = value.expand(-1, query.shape[1], -1, -1)
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling  # [B, H, S, T]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    # Attention sink: append a per-head learnable logit column before softmax.
    sink = module.attn_sink.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
    scores = torch.cat([scores, sink.to(scores.dtype)], dim=-1)
    probs = scores.softmax(dim=-1)
    # Drop the sink column when projecting onto values.
    probs = probs[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training)
    output = torch.matmul(probs, value)
    return output.transpose(1, 2).contiguous(), probs


class DeepseekV4Attention(nn.Module):
    """Sliding-window attention with optional compressed long-range segment and
    per-head learnable attention sink. No MLA."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Grouped low-rank output projection.
        group_input = (self.num_heads * self.head_dim) // self.n_groups
        self.wo_a = nn.Linear(group_input, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, config.hidden_size, bias=False)

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads))

        # Per-layer compressor / indexer — presence is dictated by `compress_ratios`.
        self.compressor: DeepseekV4Compressor | None = None
        self.indexer: DeepseekV4Indexer | None = None
        if self.compress_ratio:
            self.compressor = DeepseekV4Compressor(
                config, self.compress_ratio, self.head_dim, config.compress_rope_theta
            )
        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(config)

    def _project_q(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.q_norm(self.wq_a(hidden_states))
        query = self.wq_b(residual).view(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
        return query, residual

    def _project_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.kv_norm(self.wkv(hidden_states)).unsqueeze(2)  # [B, S, 1, head_dim]

    def _output_projection(self, attn_output: torch.Tensor) -> torch.Tensor:
        # attn_output: [B, S, H, head_dim]. Group the (H * head_dim) flat channel axis into
        # ``n_groups`` slices of ``group_input`` each, then apply a block-diagonal low-rank
        # projection to [B, S, n_groups * o_lora_rank], finally mix across groups with wo_b.
        batch, seq_len = attn_output.shape[:2]
        grouped = attn_output.reshape(batch, seq_len, self.n_groups, -1)  # [B, S, G, group_input]
        group_input = grouped.shape[-1]
        # Reshape wo_a.weight ``(G * o_lora_rank, group_input)`` → ``(G, o_lora_rank, group_input)``,
        # then bmm along G with the permuted activations to keep per-group weights separate.
        weight = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, group_input)
        # [G, B*S, group_input] @ [G, group_input, o_lora_rank] → [G, B*S, o_lora_rank]
        lhs = grouped.permute(2, 0, 1, 3).reshape(self.n_groups, batch * seq_len, group_input)
        projected = torch.bmm(lhs, weight.transpose(-1, -2))
        projected = projected.view(self.n_groups, batch, seq_len, self.o_lora_rank)
        projected = projected.permute(1, 2, 0, 3).reshape(batch, seq_len, self.n_groups * self.o_lora_rank)
        return self.wo_b(projected)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        start_pos: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        batch, seq_len = hidden_states.shape[:2]

        # Queries.
        q, q_residual = self._project_q(hidden_states)
        cos, sin = position_embeddings
        q_rope = q[..., -self.rope_head_dim :].transpose(1, 2)  # [B, H, S, rope]
        k_empty = torch.zeros_like(q_rope)
        q_rope, _ = apply_rotary_pos_emb(q_rope, k_empty, cos, sin, unsqueeze_dim=1)
        q = torch.cat([q[..., : -self.rope_head_dim], q_rope.transpose(1, 2)], dim=-1)

        # Single-head KV (broadcast to heads).
        kv = self._project_kv(hidden_states)  # [B, S, 1, head_dim]
        k_rope = kv[..., -self.rope_head_dim :].transpose(1, 2)  # [B, 1, S, rope]
        kv_empty = torch.zeros_like(k_rope)
        k_rope, _ = apply_rotary_pos_emb(k_rope, kv_empty, cos, sin, unsqueeze_dim=1)
        kv = torch.cat([kv[..., : -self.rope_head_dim], k_rope.transpose(1, 2)], dim=-1)

        # Window K/V goes through the standard KV cache (key == value in this model).
        window_kv = kv.transpose(1, 2)  # [B, 1, S, head_dim]
        if past_key_values is not None:
            window_kv, _ = past_key_values.update(window_kv, window_kv, self.layer_idx)

        # Build the full K/V (window + compressed segment) — both K and V are the same tensor.
        full_kv = window_kv
        if self.compressor is not None:
            compressed = self.compressor(hidden_states, start_pos=start_pos)
            if compressed is not None and compressed.shape[1] > 0:
                compressed = compressed.unsqueeze(1)  # [B, 1, N_comp, head_dim]
                if self.indexer is not None:
                    topk = self.indexer(hidden_states, q_residual, position_embeddings, start_pos)
                    if topk is not None:
                        # Gather the top-k compressed tokens per query position.
                        gather_idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
                        compressed_expanded = compressed.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
                        compressed = torch.gather(compressed_expanded, 3, gather_idx).reshape(
                            batch, 1, -1, self.head_dim
                        )
                full_kv = torch.cat([full_kv, compressed], dim=2)

        # Per-query attention mask (sliding-window causal already; extend with zeros for compressed part).
        extended_mask = attention_mask
        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            pad = full_kv.shape[2] - attention_mask.shape[-1]
            extended_mask = F.pad(attention_mask, (0, pad), value=0.0)

        q_t = q.transpose(1, 2)  # [B, H, S, head_dim]

        attention_interface: Callable = _eager_attention_with_sink
        if self.config._attn_implementation != "eager":
            # Other backends don't support the sink; fall back to eager for this module.
            attention_interface = _eager_attention_with_sink
        attn_output, attn_weights = attention_interface(
            self,
            q_t,
            full_kv,
            full_kv,
            extended_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )
        # attn_output is [B, S, H, head_dim]. Inverse-rotate the last rope_head_dim dims.
        rope_part = attn_output[..., -self.rope_head_dim :].transpose(1, 2)
        rope_empty = torch.zeros_like(rope_part)
        # Conjugate RoPE = negate sin.
        rope_part, _ = apply_rotary_pos_emb(rope_part, rope_empty, cos, -sin, unsqueeze_dim=1)
        attn_output = torch.cat([attn_output[..., : -self.rope_head_dim], rope_part.transpose(1, 2)], dim=-1)

        return self._output_projection(attn_output), attn_weights, q_residual


# -----------------------------------------------------------------------------
# Hyper-Connections.
# -----------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """hc_mult parallel residual streams mixed via Sinkhorn-normalised pre/post weights.

    See https://arxiv.org/abs/2409.19606 (Hyper-Connections). The Sinkhorn normalisation
    splits a ``mix_hc = (2 + hc_mult) * hc_mult`` weight tensor into a doubly-stochastic
    ``hc_mult × hc_mult`` combination matrix plus per-stream ``pre`` and ``post`` scalars.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.mix_hc = (2 + self.hc_mult) * self.hc_mult

        self.hc_fn = nn.Parameter(torch.empty(self.mix_hc, self.hc_mult * self.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.mix_hc))
        self.hc_scale = nn.Parameter(torch.empty(3))

    def _sinkhorn(self, mixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # mixes: [..., mix_hc] → split into (pre[hc_mult], post[hc_mult], comb[hc_mult, hc_mult])
        hc = self.hc_mult
        pre_scale, post_scale, comb_scale = self.hc_scale.unbind(0)
        pre_base = self.hc_base[:hc]
        post_base = self.hc_base[hc : 2 * hc]
        comb_base = self.hc_base[2 * hc :].view(hc, hc)

        pre_logits = mixes[..., :hc] * pre_scale + pre_base
        post_logits = mixes[..., hc : 2 * hc] * post_scale + post_base
        comb_logits = mixes[..., 2 * hc :].view(*mixes.shape[:-1], hc, hc) * comb_scale + comb_base

        pre = torch.sigmoid(pre_logits) + self.hc_eps
        post = torch.sigmoid(post_logits) + self.hc_eps
        comb = torch.sigmoid(comb_logits) + self.hc_eps
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, S, hc_mult, hidden] → reduced [B, S, hidden], plus post and comb for the post-step.
        flat = x.flatten(2).float()  # [B, S, hc_mult*hidden]
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn) * rsqrt
        pre_w, post_w, comb_w = self._sinkhorn(mixes)
        reduced = (pre_w.unsqueeze(-1) * x).sum(dim=2)
        return reduced.to(x.dtype), post_w, comb_w

    def post(
        self, y: torch.Tensor, residual: torch.Tensor, post_w: torch.Tensor, comb_w: torch.Tensor
    ) -> torch.Tensor:
        # y: [B, S, hidden], residual: [B, S, hc_mult, hidden]
        expanded = post_w.unsqueeze(-1) * y.unsqueeze(-2)  # [B, S, hc_mult, hidden]
        mixed = torch.matmul(comb_w, residual)  # [B, S, hc_mult, hidden]
        return (expanded + mixed).to(y.dtype)


class DeepseekV4HyperHead(nn.Module):
    """Final HC reduction before the LM head (equivalent to `ParallelHead.hc_head` in the reference)."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * self.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, hc_mult, hidden]
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn) * rsqrt
        pre_w = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.hc_eps
        return (pre_w.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# -----------------------------------------------------------------------------
# MoE — Mixtral-style routing + shared expert + static hash routing on first N layers.
# -----------------------------------------------------------------------------


def _score_fn(logits: torch.Tensor, func: str) -> torch.Tensor:
    if func == "softmax":
        return logits.softmax(dim=-1)
    if func == "sigmoid":
        return logits.sigmoid()
    return F.softplus(logits).sqrt()


class DeepseekV4Experts(MixtralExperts):
    """Routed experts. Adds optional pre-activation clamp on gate/up (swiglu_limit)."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.num_experts = config.n_routed_experts
        self.intermediate_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def _gate_up(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.act_fn(gate) * up


class DeepseekV4TopKRouter(nn.Module):
    """Score-based router with `scoring_func` and optional noaux_tc correction bias."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = _score_fn(logits, self.scoring_func)
        biased = scores + self.bias
        indices = torch.topk(biased, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.scoring_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return logits, weights, indices


class DeepseekV4HashRouter(nn.Module):
    """Static hash routing: expert indices come from a frozen ``tid2eid`` lookup
    keyed by input token id. The learned gate ``weight`` still produces scoring
    values used to weight each expert's contribution.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "tid2eid",
            torch.zeros(self.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = _score_fn(logits, self.scoring_func)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.scoring_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return logits, weights, indices


class DeepseekV4MLP(nn.Module):
    """Shared expert — a single SwiGLU MLP with ``moe_intermediate_size`` hidden dim."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV4SparseMoeBlock(nn.Module):
    """Dispatches to ``DeepseekV4TopKRouter`` (standard) or ``DeepseekV4HashRouter`` (first
    ``num_hash_layers`` layers) and sums the routed-expert output with a single shared expert.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = layer_idx < config.num_hash_layers
        if self.is_hash:
            self.gate = DeepseekV4HashRouter(config)
        else:
            self.gate = DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError("Hash-routing layers require `input_ids` to be threaded through the model.")
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch_size, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


# -----------------------------------------------------------------------------
# Decoder layer, model, and ForCausalLM.
# -----------------------------------------------------------------------------


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.mlp_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hc_mult, hidden]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        past_key_values: Cache | None = None,
        start_pos: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        reduced, post_w, comb_w = self.attn_hc.pre(hidden_states)
        attn_input = self.input_layernorm(reduced)
        attn_output, _, _ = self.self_attn(
            attn_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            start_pos=start_pos,
            **kwargs,
        )
        hidden_states = self.attn_hc.post(attn_output, residual, post_w, comb_w)

        residual = hidden_states
        reduced, post_w, comb_w = self.mlp_hc.pre(hidden_states)
        mlp_input = self.post_attention_layernorm(reduced)
        mlp_output = self.mlp(mlp_input, input_ids)
        hidden_states = self.mlp_hc.post(mlp_output, residual, post_w, comb_w)
        return hidden_states


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False  # eager-only until we port the sink-aware kernel
    _supports_sdpa = False
    _keep_in_fp32_modules_strict = ["hc_fn", "hc_base", "hc_scale", "tid2eid"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV4TopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.bias)
        elif isinstance(module, DeepseekV4HashRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.attn_sink)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, DeepseekV4Compressor):
            init.zeros_(module.ape)


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        # Single shared mask — sliding window is the same on every layer; compressed-segment
        # tokens are always reachable so their mask contribution is zero-pad applied per layer.
        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Expand to hc_mult parallel residual streams.
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                start_pos=start_pos,
                **kwargs,
            )
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV4Config):
        PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        hidden_states = self.hc_head(outputs.last_hidden_state)
        hidden_states = self.norm(hidden_states)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                getattr(outputs, "router_logits", None),
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and aux_loss is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
]
