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
"""DeepSeek V4 — modular source.

Departures from V3 / V3.2:
    * Attention: sliding-window multi-head + optional per-layer Compressor (learned
      gated pooling over ``compress_ratio`` tokens) + optional Indexer that selects
      top-k compressed positions. No MLA.
    * Residuals: replaced by Hyper-Connections (``hc_mult`` parallel streams, Sinkhorn
      mixing). Always on.
    * MoE: Mixtral-style top-k, no expert groups. First ``num_hash_layers`` layers
      route via a frozen ``tid2eid`` lookup keyed by input token id.
    * Attention sink: per-head learnable scalar, GPT-OSS eager path.
    * MTP not included — added elsewhere; its weights are silently dropped on load.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, repeat_kv
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralPreTrainedModel,
    load_balancing_loss_func,
)


logger = logging.get_logger(__name__)


# -----------------------------------------------------------------------------
# ACT2FN extension — register `sqrtsoftplus` once so configs can name it.
# -----------------------------------------------------------------------------


class _SqrtSoftplus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x).sqrt()


def _resolve_activation(name: str) -> nn.Module:
    """Tiny wrapper over ``ACT2FN`` that also understands ``sqrtsoftplus`` (V4's
    default router activation, not in the global registry yet)."""
    if name == "sqrtsoftplus":
        return _SqrtSoftplus()
    return ACT2FN[name]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(PreTrainedConfig):
    r"""
    compress_ratios (`list[int]`):
        Per-layer compression schedule. ``0`` = pure local SWA. ``4`` = overlapping-window
        compression plus an Indexer that picks top-k compressed positions. ``128`` =
        disjoint-window compression (no Indexer; all compressed positions are attended).
        Length must equal ``num_hidden_layers + num_nextn_predict_layers``.
    compress_rope_theta (`float`):
        RoPE base for layers that carry a Compressor; paired with ``rope_scaling`` for
        long-context YaRN. Layers with ``compress_ratios[i] == 0`` use ``rope_theta``.
    hc_mult (`int`, defaults to 4):
        Number of parallel Hyper-Connection streams. Always active.
    num_hash_layers (`int`, defaults to 3):
        First N layers route experts via a frozen ``tid2eid[input_ids]`` lookup.
    scoring_func (`str`, defaults to "sqrtsoftplus"):
        Activation on gate logits before selection — must be an ``ACT2FN`` key.
    swiglu_limit (`float`, defaults to 10.0):
        Clip routed experts' gate/up pre-activations like GPT-OSS (``gate.clamp_max``,
        ``up.clamp`` both sides). Shared expert is unclipped.
    o_groups (`int`):
        Groups in the grouped low-rank output projection.
    o_lora_rank (`int`):
        Rank per group in the grouped low-rank output projection.
    index_n_heads (`int`, defaults to 64):
        Heads of the sparse-attention Indexer.
    index_head_dim (`int`, defaults to 128):
        Indexer head dim.
    index_topk (`int`):
        Compressed positions the Indexer keeps per query.
    hc_sinkhorn_iters (`int`, defaults to 20):
        Sinkhorn iterations inside the Hyper-Connection mixer.
    hc_eps (`float`, defaults to 1e-6):
        Numerical floor for the HC RMS norm and the Sinkhorn normaliser.
    num_nextn_predict_layers (`int`, defaults to 1):
        MTP layer count in the upstream checkpoint. Weights are dropped on load.
    rope_theta (`float`, defaults to 10000.0):
        Base period for the main-attention rotary embedding (SWA layers).
    rope_scaling (`dict`, *optional*):
        YaRN scaling applied to the main and Compressor rotary embeddings.
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
        total = self.num_hidden_layers + (self.num_nextn_predict_layers or 0)
        if self.compress_ratios is None:
            self.compress_ratios = [0] + [4 if (i % 2) else 128 for i in range(total - 2)] + [0]
        if len(self.compress_ratios) != total:
            raise ValueError(
                f"`compress_ratios` must be length {total} (num_hidden_layers + num_nextn_predict_layers); "
                f"got {len(self.compress_ratios)}."
            )
        for r in self.compress_ratios:
            if r not in (0, 4, 128):
                raise ValueError(f"Unsupported compress_ratio={r}. Expected 0, 4, or 128.")
        if self.layer_types is None:
            # Every layer has a local sliding window; compressor/indexer presence is
            # driven by `compress_ratios`, resolved once at module __init__.
            self.layer_types = ["sliding_attention"] * self.num_hidden_layers
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim

        # Make rotary embeddings size themselves to `qk_rope_head_dim` via the shared
        # partial-rotary path (which runs iff rope_type != "default").
        rp = dict(self.rope_parameters) if isinstance(self.rope_parameters, dict) else {}
        rp.setdefault("rope_theta", self.rope_theta)
        rp["partial_rotary_factor"] = self.qk_rope_head_dim / self.head_dim
        if self.rope_scaling is not None:
            rp["rope_type"] = self.rope_scaling.get("type", "yarn")
            for k, v in self.rope_scaling.items():
                rp.setdefault(k, v)
            rp.setdefault("factor", 1.0)
        else:
            # Force the shared (partial-rotary-aware) init path.
            rp.setdefault("rope_type", "linear")
            rp.setdefault("factor", 1.0)
        self.rope_parameters = rp
        super().__post_init__(**kwargs)


# -----------------------------------------------------------------------------
# Norms / RoPE
# -----------------------------------------------------------------------------


class DeepseekV4RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV4RotaryEmbedding(LlamaRotaryEmbedding):
    """Inherits Llama's rotary embedding. ``qk_rope_head_dim`` is honoured through
    ``rope_parameters['partial_rotary_factor']``, which is set automatically in
    :meth:`DeepseekV4Config.__post_init__`. Built twice on the model — once for the
    main rope_theta (sliding-only layers), once for ``compress_rope_theta`` (layers
    that carry a Compressor) — with a thin helper to swap the base theta.
    """


def _build_rotary(config: DeepseekV4Config, rope_theta: float, with_yarn: bool) -> DeepseekV4RotaryEmbedding:
    rp = dict(config.rope_parameters)
    rp["rope_theta"] = rope_theta
    if not with_yarn:
        rp["rope_type"] = "linear"
        rp["factor"] = 1.0
        rp.pop("original_max_position_embeddings", None)
    variant = type(config)(**{**config.to_dict(), "rope_parameters": rp})
    return DeepseekV4RotaryEmbedding(variant)


# -----------------------------------------------------------------------------
# Cache — extends DynamicCache with per-layer Compressor + Indexer state.
# -----------------------------------------------------------------------------


class DeepseekV4Cache(DynamicCache):
    """Carries per-layer state for stateless Compressor / Indexer modules.

    * ``compressor_state[layer_idx]`` — dict with ``{"buffer_kv", "buffer_gate", "pooled_kv"}``;
      holds the pre-pool accumulator and the running compressed-KV cache.
    * ``indexer_state[layer_idx]`` — dict with ``{"pooled_kv"}``; running indexer
      compressed-KV cache (separate from the attention Compressor's pool).

    Window KV states are delegated to the standard :class:`DynamicCache` machinery,
    which auto-selects ``DynamicSlidingWindowLayer`` from the config's ``sliding_window``
    field.
    """

    def __init__(self, config: DeepseekV4Config | None = None):
        super().__init__(config=config)
        n = getattr(config, "num_hidden_layers", 0) if config is not None else 0
        self.compressor_state: list[dict | None] = [None] * n
        self.indexer_state: list[dict | None] = [None] * n


# -----------------------------------------------------------------------------
# Compressor — stateless: the window / pooled state lives in DeepseekV4Cache.
# -----------------------------------------------------------------------------


class DeepseekV4Compressor(nn.Module):
    """Learned gated pooling over ``compress_ratio`` consecutive tokens.

    Overlap mode (``ratio == 4``) pools every ``ratio``-token window with a stride of
    ``ratio``; disjoint mode (``ratio == 128``) pools non-overlapping windows.

    The module is stateless — any pre-pool accumulator and running pooled cache are
    owned by :class:`DeepseekV4Cache`, identified by ``layer_idx``. The caller passes
    the rotary module to use for the rope half of the pooled output.
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, head_dim))
        self.kv_norm = DeepseekV4RMSNorm(head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _pool(kv: torch.Tensor, gate: torch.Tensor, ratio: int, ape: torch.Tensor) -> torch.Tensor:
        # kv, gate: [B, N*ratio, D]  →  [B, N, D]
        batch, length, dim = kv.shape
        kv = kv.view(batch, length // ratio, ratio, dim)
        gate = gate.view(batch, length // ratio, ratio, dim) + ape
        return (kv * gate.softmax(dim=2)).sum(dim=2)

    def _apply_rope(self, kv: torch.Tensor, rotary: nn.Module, positions: torch.Tensor) -> torch.Tensor:
        rope_dim = self.rope_head_dim
        nope, rope = kv[..., :-rope_dim], kv[..., -rope_dim:]
        # apply_rotary_pos_emb_interleave expects [B, H, S, D]; treat as single-head.
        cos, sin = rotary(kv, positions)
        rope, _ = apply_rotary_pos_emb_interleave(
            rope.unsqueeze(1), torch.zeros_like(rope).unsqueeze(1), cos, sin, unsqueeze_dim=1
        )
        return torch.cat([nope, rope.squeeze(1)], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: nn.Module,
        cache: DeepseekV4Cache | None,
        layer_idx: int,
        start_pos: int,
        state_key: str = "compressor_state",
    ) -> torch.Tensor | None:
        r = self.compress_ratio
        batch, seq_len, _ = hidden_states.shape
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        state = getattr(cache, state_key)[layer_idx] if cache is not None else None

        # Splice any previously-stashed partial-window tokens onto the front.
        if state is not None and state["buffer_kv"].shape[1]:
            kv = torch.cat([state["buffer_kv"], kv], dim=1)
            gate = torch.cat([state["buffer_gate"], gate], dim=1)
        total = kv.shape[1]
        usable = (total // r) * r
        if usable == 0:
            if cache is not None:
                getattr(cache, state_key)[layer_idx] = {
                    "buffer_kv": kv,
                    "buffer_gate": gate,
                    "pooled_kv": state["pooled_kv"] if state is not None else None,
                }
            return None

        pooled = self._pool(kv[:, :usable], gate[:, :usable], r, self.ape)
        pooled = self.kv_norm(pooled)
        # Positions: the window indexed [j*r, (j+1)*r) pools to "position (j+1)*r - r".
        base = max(0, start_pos)
        pos_start = base + (total - usable - seq_len if total > seq_len else 0)
        positions = torch.arange(pooled.shape[1], device=pooled.device) * r + pos_start
        pooled = self._apply_rope(pooled, rotary, positions.unsqueeze(0).expand(batch, -1))

        if cache is not None:
            prev = state["pooled_kv"] if state is not None else None
            new_pool = pooled if prev is None else torch.cat([prev, pooled], dim=1)
            getattr(cache, state_key)[layer_idx] = {
                "buffer_kv": kv[:, usable:],
                "buffer_gate": gate[:, usable:],
                "pooled_kv": new_pool,
            }
        return pooled


# -----------------------------------------------------------------------------
# Indexer — stateless: its own pooled-KV cache lives on DeepseekV4Cache.
# -----------------------------------------------------------------------------


class DeepseekV4Indexer(nn.Module):
    """Scores compressed positions with a per-head weighted dot product and returns
    the top-k indices into the indexer's running pooled-KV cache. Stateless."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(config, compress_ratio=4, head_dim=self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: DeepseekV4Cache | None,
        layer_idx: int,
        start_pos: int,
    ) -> torch.LongTensor | None:
        batch, seq_len, _ = hidden_states.shape
        # Ensure a compressed-KV tensor is available for this layer.
        self.compressor(
            hidden_states,
            rotary=rotary,
            cache=cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            state_key="indexer_state",
        )
        state = cache.indexer_state[layer_idx] if cache is not None else None
        if state is None or state.get("pooled_kv") is None or state["pooled_kv"].shape[1] == 0:
            return None
        pooled = state["pooled_kv"]

        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim)
        nope, rope = q[..., : -self.rope_head_dim], q[..., -self.rope_head_dim :]
        cos, sin = position_embeddings
        rope, _ = apply_rotary_pos_emb_interleave(
            rope.transpose(1, 2), torch.zeros_like(rope.transpose(1, 2)), cos, sin
        )
        q = torch.cat([nope, rope.transpose(1, 2)], dim=-1)

        # score[b,s,t] = sum_h weights[b,s,h] * ReLU(q[b,s,h,:] · k[b,t,:]) * softmax_scale
        scores = torch.matmul(q.float(), pooled.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices


# -----------------------------------------------------------------------------
# Attention — sliding window + optional Compressor + optional Indexer + sink.
# -----------------------------------------------------------------------------


def eager_attention_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager attention with a per-head learnable sink column appended to the softmax
    denominator — ported from :func:`transformers.models.gpt_oss.eager_attention_forward`.
    SDPA / flash backends are not used for V4 yet because they do not honour ``s_aux``.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values  # bf16/fp16 overflow guard
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
    return output, probs


class DeepseekV4Attention(nn.Module):
    """Sliding-window MHA with single-head (MQA-ish) KV, grouped low-rank output,
    per-head learnable attention sink (GPT-OSS eager path), and optional long-range
    compressed segment.

    Q: wq_a → RMSNorm → wq_b  (num_heads * head_dim)
    K/V (shared tensor, ``num_key_value_heads=1``): wkv → RMSNorm
    O: grouped low-rank ``wo_a`` (per-group Linear, block-diagonal) → ``wo_b``
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.layer_type = config.layer_types[layer_idx]

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))  # named `sinks` to match GPT-OSS

        self.compressor: DeepseekV4Compressor | None = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim) if self.compress_ratio else None
        )
        self.indexer: DeepseekV4Indexer | None = DeepseekV4Indexer(config) if self.compress_ratio == 4 else None
        self.n_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank

    def _grouped_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Block-diagonal low-rank: per-group bmm via a reshaped wo_a weight."""
        batch, seq_len = attn_output.shape[:2]
        grouped = attn_output.reshape(batch, seq_len, self.n_groups, -1)
        group_input = grouped.shape[-1]
        weight = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, group_input)
        lhs = grouped.permute(2, 0, 1, 3).reshape(self.n_groups, batch * seq_len, group_input)
        projected = torch.bmm(lhs, weight.transpose(-1, -2))  # [G, B*S, o_lora_rank]
        projected = projected.view(self.n_groups, batch, seq_len, self.o_lora_rank)
        return self.wo_b(projected.permute(1, 2, 0, 3).reshape(batch, seq_len, -1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor],
        rotary_main: nn.Module,
        rotary_compress: nn.Module,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        start_pos: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        # ---- Q projection (with residual stash for the Indexer) ----
        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q[..., : -self.rope_head_dim], q[..., -self.rope_head_dim :]

        # ---- K/V (shared single-head tensor) ----
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        kv_nope, kv_rope = kv[..., : -self.rope_head_dim], kv[..., -self.rope_head_dim :]

        # ---- RoPE on the rope slice only ----
        q_rope, kv_rope = apply_rotary_pos_emb_interleave(q_rope, kv_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        # ---- Sliding-window KV cache via the standard DynamicCache path ----
        if past_key_values is not None:
            kv, _ = past_key_values.update(kv, kv, self.layer_idx)

        full_kv = kv

        # ---- Optional compressed long-range segment ----
        if self.compressor is not None:
            compressed = self.compressor(
                hidden_states,
                rotary=rotary_compress,
                cache=past_key_values,
                layer_idx=self.layer_idx,
                start_pos=start_pos,
            )
            state = past_key_values.compressor_state[self.layer_idx] if past_key_values is not None else None
            pooled = state["pooled_kv"] if state is not None else compressed
            if pooled is not None and pooled.shape[1] > 0:
                pooled = pooled.unsqueeze(1)  # [B, 1, N_comp, head_dim]
                if self.indexer is not None:
                    topk = self.indexer(
                        hidden_states,
                        q_residual,
                        rotary_compress,
                        position_embeddings_compress,
                        cache=past_key_values,
                        layer_idx=self.layer_idx,
                        start_pos=start_pos,
                    )
                    if topk is not None:
                        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
                        pooled_expanded = pooled.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
                        pooled = torch.gather(pooled_expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)
                full_kv = torch.cat([full_kv, pooled], dim=2)

        # ---- Extend the (sliding) mask with zero-pad over the compressed segment ----
        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_with_sink
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            full_kv,
            full_kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # ---- Inverse-rotate the rope slice on the output (reference model.py does this) ----
        rope_part = attn_output[..., -self.rope_head_dim :].transpose(1, 2)
        rope_part, _ = apply_rotary_pos_emb_interleave(rope_part, torch.zeros_like(rope_part), cos, -sin)
        attn_output = torch.cat([attn_output[..., : -self.rope_head_dim], rope_part.transpose(1, 2)], dim=-1)

        return self._grouped_output(attn_output), attn_weights


# -----------------------------------------------------------------------------
# Hyper-Connections — each layer owns one module that wraps a call and mixes
# streams via Sinkhorn-normalised weights (GPT-OSS-style __call__ usage).
# -----------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """Wraps an inner nn.Module call, collapsing ``hc_mult`` residual streams to a
    single tensor, running the inner block, then expanding back out.

    Forward signature: ``hc(hidden_states, inner, *args, **kwargs)``.
    Inner is expected to return either a tensor or a ``(tensor, *rest)`` tuple; the
    first element is combined with the residual streams, the rest is passed through.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, self.hc_mult * self.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_scale = nn.Parameter(torch.empty(3))

    def _sinkhorn(self, mixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        pre_scale, post_scale, comb_scale = self.hc_scale.unbind(0)
        pre = torch.sigmoid(mixes[..., :hc] * pre_scale + self.hc_base[:hc]) + self.eps
        post = torch.sigmoid(mixes[..., hc : 2 * hc] * post_scale + self.hc_base[hc : 2 * hc]) + self.eps
        comb = (
            torch.sigmoid(
                mixes[..., 2 * hc :].view(*mixes.shape[:-1], hc, hc) * comb_scale + self.hc_base[2 * hc :].view(hc, hc)
            )
            + self.eps
        )
        for _ in range(self.iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb

    def forward(self, hidden_states: torch.Tensor, inner: Callable, *args, **kwargs):
        # hidden_states: [B, S, hc_mult, hidden]
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn) * rsqrt
        pre_w, post_w, comb_w = self._sinkhorn(mixes)
        reduced = (pre_w.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        out = inner(reduced, *args, **kwargs)
        tensor_out, rest = (out[0], out[1:]) if isinstance(out, tuple) else (out, ())
        expanded = post_w.unsqueeze(-1) * tensor_out.unsqueeze(-2)
        expanded = expanded + torch.matmul(comb_w, hidden_states)
        expanded = expanded.to(hidden_states.dtype)
        return (expanded, *rest) if rest else expanded


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse before the LM head."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# -----------------------------------------------------------------------------
# MoE — Mixtral experts + classic top-k router + frozen hash router.
# -----------------------------------------------------------------------------


class DeepseekV4MLP(nn.Module):
    """SwiGLU MLP with packed ``gate_up_proj`` — shared expert and any dense MLP."""

    def __init__(self, config: DeepseekV4Config, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.moe_intermediate_size
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class DeepseekV4Experts(MixtralExperts):
    """Routed experts; clamps gate/up like GPT-OSS' :class:`GptOssExperts`."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.num_experts = config.n_routed_experts
        self.intermediate_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))


class DeepseekV4TopKRouter(nn.Module):
    """Score-based top-k router. Gate logits go through ``scoring_func`` (from
    ``ACT2FN``), then a per-expert correction bias nudges the selection — identical
    to DeepSeek V3's ``noaux_tc`` but without expert groups.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score = _resolve_activation(config.scoring_func)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        # V4 ships with `norm_topk_prob=True` and `scoring_func="sqrtsoftplus"`; renorm
        # is unconditional in the reference (softmax is not used upstream).
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(nn.Module):
    """First ``num_hash_layers`` layers use this: expert **indices** come from a
    frozen ``tid2eid[input_ids]`` lookup keyed by the input token id. The learned
    ``weight`` still produces scoring values used to weight each expert's activation.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score = _resolve_activation(config.scoring_func)
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    """Dispatches to a top-k or a hash router based on ``layer_idx < num_hash_layers``;
    sums the routed output with a single shared expert.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError("Hash-routing layers require `input_ids` to be threaded through the model.")
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


# Post-hoc patch: override MixtralExperts' `_apply_gate`-style chunk to apply
# swiglu_limit clamping. We subclass without decorators (the parent already has
# @use_experts_implementation) and override the forward path inline.
_MixtralExpertsForward = MixtralExperts.forward


def _v4_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Mixtral experts loop with pre-activation clamp on gate/up (swiglu_limit)."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    limit = getattr(self, "swiglu_limit", 0.0)
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        current = self.act_fn(gate) * up
        current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
    return final_hidden_states


DeepseekV4Experts.forward = _v4_experts_forward


# -----------------------------------------------------------------------------
# Decoder layer.
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

    def _attn_inner(self, hidden_states, **kwargs):
        hidden_states = self.input_layernorm(hidden_states)
        return self.self_attn(hidden_states, **kwargs)

    def _mlp_inner(self, hidden_states, input_ids):
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states, input_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor],
        rotary_main: nn.Module,
        rotary_compress: nn.Module,
        attention_mask: torch.Tensor | None,
        input_ids: torch.Tensor | None,
        past_key_values: Cache | None = None,
        start_pos: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self.attn_hc(
            hidden_states,
            self._attn_inner,
            position_embeddings=position_embeddings,
            position_embeddings_compress=position_embeddings_compress,
            rotary_main=rotary_main,
            rotary_compress=rotary_compress,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            start_pos=start_pos,
            **kwargs,
        )
        # self_attn returned (attn_output, attn_weights); hc unpacked attn_output and kept the rest.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = self.mlp_hc(hidden_states, self._mlp_inner, input_ids=input_ids)
        return hidden_states


# -----------------------------------------------------------------------------
# Pre-trained base.
# -----------------------------------------------------------------------------


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _keep_in_fp32_modules_strict = ["hc_fn", "hc_base", "hc_scale", "tid2eid"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                init.zeros_(module.bias)
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, (DeepseekV4HyperConnection, DeepseekV4HyperHead)):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, DeepseekV4Compressor):
            init.zeros_(module.ape)


# -----------------------------------------------------------------------------
# Model — owns two rotary embeddings (main + compressed) and the HC head.
# -----------------------------------------------------------------------------


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HyperHead(config)
        self.rotary_emb = _build_rotary(config, config.rope_theta, with_yarn=False)
        self.rotary_emb_compress = _build_rotary(
            config, config.compress_rope_theta, with_yarn=config.rope_scaling is not None
        )
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
            past_key_values = DeepseekV4Cache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        position_embeddings_compress = self.rotary_emb_compress(inputs_embeds, position_ids=position_ids)
        start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_embeddings_compress=position_embeddings_compress,
                rotary_main=self.rotary_emb,
                rotary_compress=self.rotary_emb_compress,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                start_pos=start_pos,
                **kwargs,
            )

        # Final HC collapse + pre-head norm live on the Model (matches the standard
        # `Model(...) → hidden_states → lm_head` contract used by Llama/Mixtral).
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV4Config):
        PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
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
        hidden_states = outputs.last_hidden_state
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
    "DeepseekV4Cache",
]
