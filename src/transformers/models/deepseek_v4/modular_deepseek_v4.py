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
from ...integrations import use_experts_implementation
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
)
from ..gpt_oss.modeling_gpt_oss import GptOssExperts
from ..llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
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

    # V4 has no dense-MLP layers (all MoE), and the HC mixers are layer-level parameters
    # (``hc_attn_*`` / ``hc_ffn_*``) not shardable. Otherwise the plan is V3-style.
    base_model_tp_plan = {
        "layers.*.self_attn.wq_a": "colwise",
        "layers.*.self_attn.wq_b": "colwise",
        "layers.*.self_attn.wkv": "colwise",
        "layers.*.self_attn.wo_a": "rowwise",
        "layers.*.self_attn.wo_b": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
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
    """Inherits the V3 rotary embedding; the only difference is that V4's
    ``compute_default_rope_parameters`` honours ``partial_rotary_factor`` so cos/sin
    comes out sized to ``qk_rope_head_dim`` instead of the full ``head_dim=512``.
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
    """DynamicCache + K=V sliding layers + per-layer V4 compressor state.

    State lives on the cache (per layer, per branch — one set for the attention
    compressor, one for the indexer compressor). Two kinds of state:

      * **Pre-pool buffer** (``buffer_kv`` / ``buffer_gate``) — tokens arrived after
        the last closed window that aren't yet enough to form the next one.
      * **Pooled cache** (``pooled``) — the running list of compressed tokens emitted
        so far, one per closed window.

    Compressor / Indexer modules stay stateless and just call
    :meth:`accumulate_windows` and :meth:`update_pool` on the cache.
    """

    def __init__(self, config: DeepseekV4Config | None = None):
        super().__init__(config=config)
        n = getattr(config, "num_hidden_layers", 0) if config is not None else 0
        if config is not None:
            self.layers = [DeepseekV4SlidingLayer(config.sliding_window) for _ in range(n)]
        self.compressor_state: list[dict] = []
        self.indexer_state: list[dict] = []

    def _branch_state(self, state_key: str, layer_idx: int) -> dict:
        store: list[dict] = getattr(self, state_key, None)
        if store is None:
            # Generation's default ``DynamicCache`` arrives without these; attach lazily.
            store = []
            setattr(self, state_key, store)
        while len(store) <= layer_idx:
            store.append({"buffer_kv": None, "buffer_gate": None, "pooled": None})
        return store[layer_idx]

    def accumulate_windows(
        self,
        kv: torch.Tensor,
        gate: torch.Tensor,
        layer_idx: int,
        state_key: str,
        ratio: int,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Merge newly-projected (``kv``, ``gate``) with the per-layer buffered tail and
        return the window-aligned chunk (``length // ratio * ratio``). The remainder
        stays in the buffer for the next call. ``pool_base`` is the absolute token
        position of the first pooled window returned.
        """
        state = self._branch_state(state_key, layer_idx)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = torch.cat([buf_kv, kv], dim=1)
            gate = torch.cat([buf_gate, gate], dim=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def update_pool(self, new_pooled: torch.Tensor, layer_idx: int, state_key: str) -> torch.Tensor:
        """Append ``new_pooled`` to the running pool for this layer / branch. Returns
        the full pool (empty shape ``[B, 0, D]`` if nothing has been pooled yet —
        never ``None``)."""
        state = self._branch_state(state_key, layer_idx)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else torch.cat([pool, new_pooled], dim=1)
            state["pooled"] = pool
        if pool is None:
            pool = new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return pool

    @classmethod
    def adopt(cls, cache: "Cache | None") -> "Cache":
        """Coerce an incoming cache so the compressor / indexer can stash their state.

        * ``None`` → fresh empty :class:`DeepseekV4Cache` (forward-pass scratch space).
        * ``DynamicCache`` (generation's default) → class reinterpreted in place.
        * Already a ``DeepseekV4Cache`` → no-op.
        * Any other cache (``StaticCache``, etc.) is returned as-is with the
          compressor-state helpers attached via ``__class__`` switch only when the
          class hierarchy allows it — otherwise we just use the cache's state store
          lazily (``_branch_state`` creates the dict on first access).
        """
        if isinstance(cache, cls):
            return cache
        if cache is None:
            return cls()
        if isinstance(cache, DynamicCache):
            cache.__class__ = cls  # safe: cls extends DynamicCache.
            return cache
        # Any other cache type (StaticCache, etc.): bolt on the three V4 methods as
        # bound attributes. The state dicts are created lazily by ``_branch_state``.
        for name in ("_branch_state", "accumulate_windows", "update_pool"):
            if not hasattr(cache, name):
                setattr(cache, name, getattr(cls, name).__get__(cache))
        return cache


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
    """Split ``x`` along its last dim into the nope-slice (first dims) and the rope-slice
    (last ``rope_head_dim`` dims), rotate the rope slice with the standard Llama / GPT-NeoX
    ``apply_rotary_pos_emb`` (``rotate_half`` + ``cat(freqs, freqs)``-shaped cos/sin), and
    glue the two back together.
    """
    nope, rope = x[..., :-rope_head_dim], x[..., -rope_head_dim:]
    rope, _ = apply_rotary_pos_emb(rope, torch.zeros_like(rope), cos, sin)
    return torch.cat([nope, rope], dim=-1)


def _pool_windows(kv: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor, ratio: int, head_dim: int) -> torch.Tensor:
    """Softmax-gated sum-pool: reshape into ``ratio``-sized windows and collapse.

    weights = softmax(gate + ape, dim=window)
    pooled  = sum(weights * kv, dim=window)
    """
    batch, length, _ = kv.shape
    kv = kv.view(batch, length // ratio, ratio, head_dim)
    gate = gate.view(batch, length // ratio, ratio, head_dim) + ape.to(gate.dtype)
    return (kv * gate.softmax(dim=2)).sum(dim=2)


def _rope_pool_positions(
    pool_length: int, pool_base: int, ratio: int, device: torch.device, batch: int
) -> torch.Tensor:
    """Absolute positions of the pooled tokens: ``[pool_base, pool_base + ratio, …]``."""
    return (torch.arange(pool_length, device=device) * ratio + pool_base).unsqueeze(0).expand(batch, -1)


class DeepseekV4Indexer(nn.Module):
    """Picks the top-k compressed positions per query. Owned by ``DeepseekV4Compressor``
    when ``compress_ratio == 4``. Pools the same windows as the outer compressor but at
    ``index_head_dim``, then scores the pooled positions against the attention query.

    Because the indexer's pool uses the same ``compress_ratio`` over the same hidden
    states as the outer compressor, its pooled positions align one-for-one — the top-k
    indices it returns index into the outer compressor's pool just fine.

    All cache-state management lives on :class:`DeepseekV4Cache` (the indexer calls
    ``cache.accumulate_windows`` / ``cache.update_pool`` with ``state_key="indexer_state"``).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_ratio = 4
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(self.compress_ratio, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: "DeepseekV4Cache",
        layer_idx: int,
        start_pos: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape

        # Run our own pool at ``index_head_dim`` — positions align 1-for-1 with the
        # outer compressor's, so the returned top-k indices apply to its pool too.
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "indexer_state", self.compress_ratio, start_pos
        )
        new_pooled = self.kv_norm(_pool_windows(ready_kv, ready_gate, self.ape, self.compress_ratio, self.head_dim))
        if new_pooled.shape[1] > 0:
            positions = _rope_pool_positions(
                new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, new_pooled.shape[0]
            )
            cos, sin = rotary(new_pooled, positions)
            new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        pooled_kv = cache.update_pool(new_pooled, layer_idx, "indexer_state")

        # Score queries against the running pool.
        cos, sin = position_embeddings
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_partial_rope(q, cos, sin, self.rope_head_dim).transpose(1, 2)
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        topk = min(self.index_topk, pooled_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4Compressor(nn.Module):
    """Per-layer long-range KV branch. Pools ``compress_ratio`` consecutive tokens into
    one compressed KV and (when ``compress_ratio == 4``) narrows the running pool via
    a learned top-k Indexer. Attention concatenates the returned tensor onto its
    sliding-window KV.

    Cache-state (buffered pre-pool tokens, running pooled cache) is owned by
    :class:`DeepseekV4Cache` — this module only runs the math.
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
        self.indexer: DeepseekV4Indexer | None = DeepseekV4Indexer(config) if compress_ratio == 4 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        rotary: nn.Module,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache: "DeepseekV4Cache",
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        """Returns the long-range KV segment for this layer, shape
        ``[B, 1, N_compressed_or_topk, head_dim]`` (possibly empty in N if not enough
        tokens have arrived to close the first window).
        """
        batch, seq_len, _ = hidden_states.shape

        # Accumulate windows through the cache, pool the ready chunk, update the pool.
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        ready_kv, ready_gate, pool_base = cache.accumulate_windows(
            kv, gate, layer_idx, "compressor_state", self.compress_ratio, start_pos
        )
        new_pooled = self.kv_norm(_pool_windows(ready_kv, ready_gate, self.ape, self.compress_ratio, self.head_dim))
        positions = _rope_pool_positions(new_pooled.shape[1], pool_base, self.compress_ratio, new_pooled.device, batch)
        cos, sin = rotary(new_pooled, positions)
        new_pooled = _apply_partial_rope(new_pooled.unsqueeze(1), cos, sin, self.rope_head_dim).squeeze(1)
        pooled = cache.update_pool(new_pooled, layer_idx, "compressor_state").unsqueeze(1)

        if self.indexer is not None:
            topk = self.indexer(hidden_states, q_residual, rotary, position_embeddings, cache, layer_idx, start_pos)
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
            # The decoder layer's gradient-checkpointing wrapper strips ``past_key_values``
            # on the recompute pass. Fall back to an ephemeral V4 cache so the compressor
            # has somewhere to stage its window buffers during a stateless forward.
            compressor_cache = DeepseekV4Cache.adopt(past_key_values)
            pooled = self.compressor(
                hidden_states,
                q_residual=q_residual,
                rotary=rotary_compress,
                position_embeddings=position_embeddings_compress,
                cache=compressor_cache,
                layer_idx=self.layer_idx,
                start_pos=start_pos,
            )
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

        # De-rotate the rope slice on the output. V4 shares K and V (``wkv`` projects to a
        # single tensor), so V's last ``qk_rope_head_dim`` dims carry the same per-token
        # RoPE rotation as K. Attention sums V-rotated values across all attended
        # positions, so the output's rope slice is a mixture of rotated content values.
        # Applying the conjugate rotation ``(cos, -sin)`` at the *query* position pulls
        # the content back into a position-independent frame before the output projection
        # mixes heads — without this step, ``wo_a`` / ``wo_b`` would see position-entangled
        # content in the rope slice and couldn't learn a clean projection.
        attn_output = _apply_partial_rope(attn_output.transpose(1, 2), cos, -sin, self.rope_head_dim).transpose(1, 2)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2)), attn_weights


class DeepseekV4HyperConnection(nn.Module):
    r"""Per-site Hyper-Connection mixer. Owns the learned parameters (``fn``, ``base``,
    ``scale``) that turn the incoming ``hc_mult`` residual streams into collapse / expand
    weights; the decoder layer instantiates two of these (one for the attention site, one
    for the mlp site).

    ASCII shape guide — ``B`` = batch, ``S`` = seq, ``H`` = hc_mult, ``D`` = hidden_size::

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
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_streams.flatten(start_dim=2).float()  # [B, S, H*D]
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        # HC mixer params are kept in fp32 for Sinkhorn stability — cast defensively.
        mix = F.linear(flat, self.fn.float()) * rsqrt  # [B, S, (2+H)*H]
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
        return pre, post, comb


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
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
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
        super().__init__()
        del self.gate_up_proj_bias
        del self.down_proj_bias
        del self.alpha
        self.limit = config.swiglu_limit
        self.act_fn = ACT2FN[config.hidden_act]

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
            # skip masking index
            if expert_idx == self.num_experts:
                continue
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
        # Correction bias biases the argmax only — not a gradient-carrying parameter, so
        # store as a buffer (same convention as DeepseekV3's ``e_score_correction_bias``).
        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    """First ``num_hash_layers`` layers route via a frozen ``tid2eid`` lookup keyed by
    the input token id. The learned gate ``weight`` still produces scoring values used
    to weight each selected expert's activation; the selection itself is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
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
    r"""Hyper-Connection (https://huggingface.co/papers/2409.19606) decoder layer.

    Classic residual decoder layer::

        h ──► norm ──► self_attn ──► + ──► norm ──► mlp ──► +
        └──────── residual ────────┘   └─────── residual ───┘

    V4 decoder layer (``H = hc_mult`` parallel residual streams throughout)::

                attention site                                    mlp site
        ┌────────────────────────────────────────┐    ┌────────────────────────────────────────┐
        │  hidden_streams [B, S, H, D]           │    │  hidden_streams [B, S, H, D]           │
        │        │                               │    │        │                               │
        │  attn_hc.compute_weights ─► (pre, post, comb)  │  ffn_hc.compute_weights ─► (pre, post, comb) │
        │        │                               │    │        │                               │
        │   Σ pre·streams  (collapse)            │    │   Σ pre·streams  (collapse)            │
        │        │                               │    │        │                               │
        │   input_layernorm                      │    │   post_attention_layernorm             │
        │        │                               │    │        │                               │
        │   self_attn                            │    │   mlp  (MoE routed + shared)           │
        │        │                               │    │        │                               │
        │   post·output + comb·streams  (expand) │    │   post·output + comb·streams  (expand) │
        │        │                               │    │        │                               │
        │        ▼                               │    │        ▼                               │
        │  new hidden_streams  ──────────────────┘    │  new hidden_streams                    │
        └────────────────────────────────────────┘    └────────────────────────────────────────┘

    The two :class:`DeepseekV4HyperConnection` instances own one packed linear + bias +
    scale per site. Checkpoint keys (``hc_attn_*`` / ``hc_ffn_*`` from the upstream
    reference) are bridged to the ``attn_hc.*`` / ``ffn_hc.*`` module tree via
    ``conversion_mapping.py``.
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

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        # hidden_states shape throughout this layer: [B, S, hc_mult, hidden].

        # --- Attention site: collapse → norm → attn → expand ---
        pre, post, comb = self.attn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        # Expand: each new stream = post[h] * block_output + Σ_k comb[h, k] * streams[k].
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        # --- MLP site: same pattern ---
        pre, post, comb = self.ffn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=kwargs.get("input_ids"))
        dtype = hidden_states.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _keep_in_fp32_modules_strict = ["attn_hc", "ffn_hc"]
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
            if isinstance(module, DeepseekV4TopKRouter):
                module.bias.zero_()  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                module.tid2eid.zero_()  # buffer; real values come from the checkpoint
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
        elif isinstance(module, DeepseekV4Indexer):
            init.zeros_(module.ape)
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
        # When a cache was supplied (or use_cache asked for one) make sure it's a
        # DeepseekV4Cache so the compressor/indexer methods are there. If the caller
        # explicitly passed ``None`` with ``use_cache=False`` we leave it ``None`` so the
        # generation loop's "no cache expected" invariant holds; each Compressor call
        # then adopts a forward-scoped ephemeral cache internally.
        if past_key_values is not None:
            past_key_values = DeepseekV4Cache.adopt(past_key_values)
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
