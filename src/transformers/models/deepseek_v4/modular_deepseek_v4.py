# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb_interleave,
)
from ..llama.modeling_llama import repeat_kv
from ...integrations import use_experts_implementation
from ..gpt_oss.modeling_gpt_oss import GptOssExperts
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(DeepseekV3Config):
    r"""
    compress_ratios (`list[int]`): Per-layer compression schedule in ``{0, 4, 128}``.
        ``0`` = pure local SWA; ``4`` = overlap-window compress + Indexer; ``128`` = disjoint-window compress.
    compress_rope_theta (`float`): RoPE base for Compressor layers (paired with ``rope_scaling`` for YaRN).
    hc_mult (`int`): Hyper-Connection stream count (always active).
    num_hash_layers (`int`): First N layers route via a frozen ``tid2eid[input_ids]`` lookup.
    scoring_func (`str`): Router activation — ``sqrtsoftplus``, ``softmax``, or ``sigmoid``.
    swiglu_limit (`float`): Clip routed experts' gate/up pre-activations.
    sliding_window (`int`): Local window size used on every layer.
    o_groups (`int`), o_lora_rank (`int`): Grouped low-rank output projection.
    index_n_heads, index_head_dim, index_topk (`int`): Indexer hyperparameters.
    hc_sinkhorn_iters (`int`), hc_eps (`float`): Sinkhorn normalisation knobs.
    num_nextn_predict_layers (`int`): MTP layer count in the upstream checkpoint (not instantiated here).
    compress_rope_parameters (`dict`, *optional*): Filled in ``__post_init__``.
    """

    model_type = "deepseek_v4"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    # V4's module tree differs from V3's: attention/mlp live under the HyperConnection
    # wrappers (``attn_hc.inner`` / ``mlp_hc.inner``) and there are no dense-MLP layers.
    base_model_tp_plan = {
        "layers.*.attn_hc.inner.wq_a": "colwise",
        "layers.*.attn_hc.inner.wq_b": "colwise",
        "layers.*.attn_hc.inner.wkv": "colwise",
        "layers.*.attn_hc.inner.wo_a": "rowwise",
        "layers.*.attn_hc.inner.wo_b": "rowwise",
        "layers.*.mlp_hc.inner.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp_hc.inner.experts.down_proj": "rowwise",
        "layers.*.mlp_hc.inner.experts": "moe_tp_experts",
        "layers.*.mlp_hc.inner.shared_experts.gate_proj": "colwise",
        "layers.*.mlp_hc.inner.shared_experts.up_proj": "colwise",
        "layers.*.mlp_hc.inner.shared_experts.down_proj": "rowwise",
    }

    # V4 reshapes the attention: single-head KV, grouped low-rank output, no MLA decomposition.
    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0

    # V4-specific.
    compress_ratios: list[int] | None = None
    compress_rope_theta: float = 160000.0
    compress_rope_parameters: dict | None = None
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1

    # Fields carried from DeepseekV3Config but unused in V4 — kept ``None`` so the
    # MLA paths never fire (V3 fields that depend on them are guarded by truthiness checks).
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    first_k_dense_replace: int | None = None
    rope_interleave: bool | None = True

    # Router-side extras inherited from Mixtral config path.
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    # Rotary config. ``rope_parameters`` (alias ``rope_scaling``) is the HF standard dict;
    # ``partial_rotary_factor`` tells the shared rope-init path to size cos/sin to
    # ``qk_rope_head_dim`` instead of the full ``head_dim``.
    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    def __post_init__(self, **kwargs):
        n = self.num_hidden_layers
        # Upstream configs ship ``num_hidden_layers + num_nextn_predict_layers`` entries
        # (the trailing MTP entries are for an MTP block we don't instantiate); accept
        # either length and keep only the first ``num_hidden_layers``.
        if self.compress_ratios is None:
            self.compress_ratios = [0] + [4 if i % 2 else 128 for i in range(max(n - 2, 0))] + ([0] if n >= 2 else [])
        self.compress_ratios = list(self.compress_ratios[:n])
        if len(self.compress_ratios) != n:
            raise ValueError(f"`compress_ratios` must cover at least {n} layers, got {len(self.compress_ratios)}.")
        for r in self.compress_ratios:
            if r not in (0, 4, 128):
                raise ValueError(f"Unsupported compress_ratio={r}; expected 0, 4, or 128.")
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        # RoPE is only applied to the last ``qk_rope_head_dim`` dims of each head; the
        # shared rope-init path picks that up from ``partial_rotary_factor``.
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
        # Skip ``DeepseekV3Config.__post_init__`` — it pins ``head_dim`` to
        # ``qk_rope_head_dim`` for the MLA rotary, which would stomp V4's head_dim=512.
        PreTrainedConfig.__post_init__(self, **kwargs)
        # The compressed-segment rope shares structure with the main dict but overrides
        # the base ``rope_theta``; build it lazily here so it round-trips through to_dict.
        self.compress_rope_parameters = {**self.rope_parameters, "rope_theta": self.compress_rope_theta}


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4RotaryEmbedding(DeepseekV3RotaryEmbedding):
    """Honours ``partial_rotary_factor`` on the default rope path as well, so V4's
    attention (with ``head_dim != qk_rope_head_dim``) gets correctly-sized cos/sin.
    """

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


class DeepseekV4SlidingLayer(DynamicSlidingWindowLayer):
    """Sliding-window cache layer that stores K=V once (V4 attention shares the tensor)."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full


class DeepseekV4Cache(DynamicCache):
    """DynamicCache + per-layer compressor / indexer state dicts and K=V sliding layers."""

    def __init__(self, config: DeepseekV4Config | None = None):
        super().__init__(config=config)
        n = getattr(config, "num_hidden_layers", 0) if config is not None else 0
        if config is not None:
            self.layers = [DeepseekV4SlidingLayer(config.sliding_window) for _ in range(n)]
        self.compressor_state: list[dict | None] = [None] * n
        self.indexer_state: list[dict | None] = [None] * n


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear. The ``weight`` parameter is shaped like a
    standard ``nn.Linear`` (``[out_features, in_features_per_group]``) so quantizers
    keyed on ``nn.Linear.weight`` still pick it up; ``forward`` does per-group bmm.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


def _apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int) -> torch.Tensor:
    """Split ``x`` along its head dim into the nope-slice (first dims) and the rope-slice
    (last ``rope_head_dim`` dims), apply interleaved RoPE to the rope slice, and glue
    the two back together.
    """
    nope, rope = x[..., :-rope_head_dim], x[..., -rope_head_dim:]
    rope, _ = apply_rotary_pos_emb_interleave(rope, torch.zeros_like(rope), cos, sin)
    return torch.cat([nope, rope], dim=-1)


class DeepseekV4Indexer(nn.Module):
    """Lives inside ``DeepseekV4Compressor`` when ``compress_ratio == 4``. Runs a smaller
    (``index_head_dim``) companion compressor over the same hidden states and scores those
    pooled positions against the attention query to pick the top-k most informative.

    The positions the Indexer selects align with the outer ``DeepseekV4Compressor``'s pool
    (both pool with the same ratio over the same hidden states), so the indices it returns
    can be used directly to gather from the outer pool.
    """

    def __init__(self, config: DeepseekV4Config, rotary: nn.Module | None = None):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        # Separate pool at the indexer's (smaller) head dim — its pooled positions align
        # one-for-one with the outer compressor's, so we reuse the returned top-k indices.
        self.compressor = DeepseekV4Compressor(
            config, compress_ratio=4, head_dim=self.head_dim, _owns_indexer=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: "DeepseekV4Cache | None",
        layer_idx: int,
        start_pos: int,
    ) -> torch.LongTensor | None:
        pooled = self.compressor(
            hidden_states,
            q_residual=None,
            rotary=rotary,
            position_embeddings=position_embeddings,
            cache=cache,
            layer_idx=layer_idx,
            start_pos=start_pos,
            state_key="indexer_state",
        )
        if pooled is None:
            return None
        pooled_kv = pooled.squeeze(1)  # [B, T, head_dim]
        batch, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        return index_scores.topk(min(self.index_topk, index_scores.shape[-1]), dim=-1).indices


class DeepseekV4Compressor(nn.Module):
    """Per-layer long-range KV branch: pools ``compress_ratio`` consecutive tokens into
    one compressed KV, optionally filtered by an Indexer (for ``compress_ratio == 4``
    layers). Attention calls ``compressor(...)`` to get the long-range segment already
    ready to concatenate with its sliding-window KV.

    Pooling weights are softmax-normalised per window:

        weights = softmax(wgate(x) + ape, dim=window)
        pooled  = sum(weights * wkv(x), dim=window)

    RoPE is applied to the last ``qk_rope_head_dim`` dims of each pooled token with the
    caller-supplied rotary (typically the model's compressed-theta YaRN rotary).
    """

    def __init__(
        self, config: DeepseekV4Config, compress_ratio: int, head_dim: int, _owns_indexer: bool = True
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(compress_ratio, head_dim))
        self.kv_norm = DeepseekV4RMSNorm(head_dim, eps=config.rms_norm_eps)
        # Indexer is part of the compressor because it never exists without one: only
        # the ratio-4 variant gates its pooled KV through a learned top-k selector.
        # ``_owns_indexer=False`` is used when we build the companion compressor the
        # Indexer itself owns (it must not recurse into another Indexer).
        self.indexer: DeepseekV4Indexer | None = (
            DeepseekV4Indexer(config) if (compress_ratio == 4 and _owns_indexer) else None
        )

    def _pool_window(self, kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Reshape ``kv`` / ``gate`` into ``compress_ratio``-sized windows and sum-pool."""
        batch, length, _ = kv.shape
        r = self.compress_ratio
        kv = kv.view(batch, length // r, r, self.head_dim)
        gate = gate.view(batch, length // r, r, self.head_dim) + self.ape
        return (kv * gate.softmax(dim=2)).sum(dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: "DeepseekV4Cache | None",
        layer_idx: int,
        start_pos: int,
        state_key: str = "compressor_state",
    ) -> torch.Tensor | None:
        """Returns the long-range KV segment for this layer — already indexer-filtered
        if ``compress_ratio == 4``. Shape: ``[B, 1, N_compressed_or_topk, head_dim]``.
        Returns ``None`` if not enough tokens have arrived to close a window yet.
        """
        r = self.compress_ratio
        batch, seq_len, _ = hidden_states.shape
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)

        # Caches without the per-layer compressor store (e.g. generation's DynamicCache)
        # skip the stateful incremental path and just pool over the current call.
        store = getattr(cache, state_key, None)
        state = store[layer_idx] if store is not None else None
        if state is not None and state["buffer_kv"].shape[1]:
            kv = torch.cat([state["buffer_kv"], kv], dim=1)
            gate = torch.cat([state["buffer_gate"], gate], dim=1)
        usable = (kv.shape[1] // r) * r
        if usable == 0:
            if store is not None:
                store[layer_idx] = {
                    "buffer_kv": kv,
                    "buffer_gate": gate,
                    "pooled_kv": state["pooled_kv"] if state is not None else None,
                }
            return None

        # Pool and rope the newly-closed windows.
        pooled = self.kv_norm(self._pool_window(kv[:, :usable], gate[:, :usable]))
        base = max(0, start_pos) + (kv.shape[1] - usable - seq_len if kv.shape[1] > seq_len else 0)
        positions = (torch.arange(pooled.shape[1], device=pooled.device) * r + base).unsqueeze(0).expand(batch, -1)
        cos, sin = rotary(pooled, positions)
        pooled = _apply_partial_rope(pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)

        # Update the running compressed-KV cache.
        if store is not None:
            prev = state["pooled_kv"] if state is not None else None
            store[layer_idx] = {
                "buffer_kv": kv[:, usable:],
                "buffer_gate": gate[:, usable:],
                "pooled_kv": pooled if prev is None else torch.cat([prev, pooled], dim=1),
            }
            pooled = store[layer_idx]["pooled_kv"]

        if pooled.shape[1] == 0:
            return None

        # Shape to [B, 1, N, head_dim] — single KV head, broadcast to query heads downstream.
        pooled = pooled.unsqueeze(1)
        if self.indexer is not None:
            # Pick the top-k most informative compressed positions per query token. The
            # Indexer runs its own (smaller-dim) companion compressor; its pool aligns
            # one-for-one with ours so the returned indices are valid for gathering here.
            topk = self.indexer(
                hidden_states, q_residual, rotary, position_embeddings, cache, layer_idx, start_pos
            )
            if topk is not None:
                expanded = pooled.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
                idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
                pooled = torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)
        return pooled


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
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    return torch.matmul(probs, value_states).transpose(1, 2).contiguous(), probs


class DeepseekV4Attention(DeepseekV3Attention):
    """SWA + Compressor + Indexer + attention sink. Single-head KV (``num_key_value_heads=1``),
    grouped low-rank output. No MLA decomposition, so we override ``__init__`` and ``forward``
    rather than reusing V3's kv_a/kv_b projections — but the shared V3 scaffolding
    (config plumbing, ``layer_idx``, ``is_causal``) is inherited.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim) if self.compress_ratio else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor],
        rotary_compress: nn.Module,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        start_pos: int = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)

        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim)
        kv = _apply_partial_rope(kv, cos, sin, self.rope_head_dim)

        # V4 cache layer stores K=V once — returns the shared tensor for both positions.
        if past_key_values is not None:
            kv, _ = past_key_values.update(kv, kv, self.layer_idx)
        full_kv = kv

        if self.compressor is not None:
            pooled = self.compressor(
                hidden_states,
                q_residual=q_residual,
                rotary=rotary_compress,
                position_embeddings=position_embeddings_compress,
                cache=past_key_values,
                layer_idx=self.layer_idx,
                start_pos=start_pos,
            )
            if pooled is not None:
                full_kv = torch.cat([full_kv, pooled], dim=2)

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

        # De-rotate the rope slice on the output (reference V4 `model.py`).
        attn_output = _apply_partial_rope(attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim).transpose(1, 2)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2)), attn_weights


class DeepseekV4HyperConnection(nn.Module):
    """Hyper-Connections (https://huggingface.co/papers/2409.19606).

    Replaces the usual residual connection around a sub-block with ``hc_mult`` parallel
    residual streams. A learned Sinkhorn-normalised mixer combines streams on entry and
    re-assembles them on exit, allowing the block to operate on a single-stream input
    while the surrounding streams still carry information through the layer.

    This module is the "site" wrapper: it owns ``inner`` (the attention or MLP block)
    and ``norm`` (the pre-block RMSNorm). ``forward`` performs the full sequence:

        in_streams : [B, S, hc_mult, hidden]
        ──► flatten + RMSNorm + F.linear(..., hc_fn) ─► mix logits
        ──► Sinkhorn split → (pre, post, comb)
        ──► collapsed = sum_streams(pre * in_streams)                # [B, S, hidden]
        ──► out = inner(norm(collapsed))                             # the block
        ──► out_streams = post * out + comb @ in_streams             # [B, S, hc_mult, hidden]

    Where:
      * ``pre``  [hc_mult]           — per-stream weights used to collapse the input.
      * ``post`` [hc_mult]           — per-stream weights placing the block output back.
      * ``comb`` [hc_mult, hc_mult]  — a doubly-stochastic stream-mixing matrix.

    ``pre``, ``post``, ``comb`` are carved out of a single
    ``(2 + hc_mult) * hc_mult``-wide vector of mix logits and then passed through a
    small Sinkhorn loop (``hc_sinkhorn_iters``) to enforce the row/column sums.
    """

    def __init__(self, config: DeepseekV4Config, inner: nn.Module, norm: nn.Module):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.inner = inner
        self.norm = norm

        # The mix logits pack `pre`, `post`, and the `comb` matrix end-to-end so we only
        # need a single linear on the flattened streams. `num_mix_logits` is that total.
        self.num_mix_logits = (2 + self.hc_mult) * self.hc_mult
        self.hc_fn = nn.Parameter(torch.empty(self.num_mix_logits, self.hc_mult * self.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.num_mix_logits))
        # One learned scale per sub-part (pre, post, comb), broadcast across all logits.
        self.hc_scale = nn.Parameter(torch.empty(3))

    def _mix_logits(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        """RMSNorm-rescale the concatenated stream vector, then project to mix logits."""
        flat = hidden_streams.flatten(start_dim=2).float()  # [B, S, hc_mult * hidden]
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        return F.linear(flat, self.hc_fn) * rsqrt  # [B, S, num_mix_logits]

    def _split_mix(self, mix_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice ``mix_logits`` into ``(pre, post, comb)`` and Sinkhorn-normalise ``comb``."""
        hc = self.hc_mult
        pre_scale, post_scale, comb_scale = self.hc_scale.unbind(0)
        pre_base, post_base = self.hc_base[:hc], self.hc_base[hc : 2 * hc]
        comb_base = self.hc_base[2 * hc :].view(hc, hc)

        pre = torch.sigmoid(mix_logits[..., :hc] * pre_scale + pre_base) + self.hc_eps
        post = torch.sigmoid(mix_logits[..., hc : 2 * hc] * post_scale + post_base) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix_logits[..., 2 * hc :].view(*mix_logits.shape[:-1], hc, hc) * comb_scale + comb_base
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    def forward(self, hidden_streams: torch.Tensor, **kwargs) -> torch.Tensor:
        """Args:
            hidden_streams: ``[B, S, hc_mult, hidden]`` — the input residual streams.
        Returns:
            ``[B, S, hc_mult, hidden]`` — output streams after running ``inner``.
        """
        pre, post, comb = self._split_mix(self._mix_logits(hidden_streams))

        # 1. Collapse `hc_mult` streams into one (weighted sum with `pre`).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)

        # 2. Run the block on the collapsed representation.
        inner_output = self.inner(self.norm(collapsed), **kwargs)
        if isinstance(inner_output, tuple):
            inner_output = inner_output[0]  # tuples come back from self-attention forwards

        # 3. Expand back to `hc_mult` streams:
        #    - `post[:, :, :, None] * inner_output` re-assigns the block output to each stream
        #    - `comb @ hidden_streams` mixes the untouched residual streams
        next_streams = post.unsqueeze(-1) * inner_output.unsqueeze(-2)
        next_streams = next_streams + torch.matmul(comb, hidden_streams)
        return next_streams.to(hidden_streams.dtype)


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by ``DeepseekV4Model`` before the shared RMSNorm."""

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


class DeepseekV4MLP(Qwen2MoeMLP):
    """Shared expert — plain SwiGLU MLP, ``moe_intermediate_size`` hidden."""

    def __init__(self, config: DeepseekV4Config, intermediate_size: int | None = None):
        super().__init__(config, intermediate_size or config.moe_intermediate_size)


@use_experts_implementation
class DeepseekV4Experts(GptOssExperts):
    """Routed experts reuse GPT-OSS' expert machinery (packed ``gate_up_proj`` + per-expert
    iteration, ``_apply_gate`` hook). V4 differs in: no biases, Mixtral-style ``chunk(2)``
    gate/up split, SiLU activation, and ``swiglu_limit`` clamping before the activation.
    """

    def __init__(self, config: DeepseekV4Config):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        self.act_fn = ACT2FN[config.hidden_act]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
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
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            current = self._apply_gate(gate_up)
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(MixtralTopKRouter):
    """Classic Mixtral-style top-k routing with two V4 tweaks: the softmax is replaced
    by a configurable ``scoring_func`` (``sqrtsoftplus`` for V4 checkpoints), and the
    top-k selection is biased by a per-expert learnable correction (same ``noaux_tc``
    idea as DeepSeek V3, without the expert groups).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(DeepseekV4TopKRouter):
    """First ``num_hash_layers`` layers: expert indices come from a frozen ``tid2eid``
    lookup keyed by the input token id. The learned gate ``weight`` still produces
    scoring values used to weight each selected expert's contribution.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        # Hash routing has no top-k / correction-bias selection — indices are static.
        del self.bias
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
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None, **_) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepseekV4's hash-routing layers need `input_ids` to look up expert indices. "
                    "The `inputs_embeds`-only inference path is not supported for models with "
                    "`num_hash_layers > 0`."
                )
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_hc = DeepseekV4HyperConnection(
            config,
            DeepseekV4Attention(config, layer_idx),
            DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )
        self.mlp_hc = DeepseekV4HyperConnection(
            config,
            DeepseekV4SparseMoeBlock(config, layer_idx),
            DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        hidden_states = self.attn_hc(hidden_states, **kwargs)
        return self.mlp_hc(hidden_states, **kwargs)


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _keep_in_fp32_modules_strict = ["hc_fn", "hc_base", "hc_scale"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]
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
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                init.zeros_(module.bias)
            if isinstance(module, DeepseekV4HashRouter):
                # ``tid2eid`` is a persistent int64 lookup table; default-init is zeros
                # (the first expert), meaningful values come from the pretrained checkpoint.
                module.tid2eid.zero_()
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
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        compress_config = copy.copy(config)
        compress_config.rope_parameters = config.compress_rope_parameters
        self.rotary_emb_compress = DeepseekV4RotaryEmbedding(compress_config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        cos_sin = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        cos_sin_compress = self.rotary_emb_compress(inputs_embeds, position_ids=position_ids)
        start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=cos_sin,
                position_embeddings_compress=cos_sin_compress,
                rotary_main=self.rotary_emb,
                rotary_compress=self.rotary_emb_compress,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                start_pos=start_pos,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV4Config):
        PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Cache",
]
