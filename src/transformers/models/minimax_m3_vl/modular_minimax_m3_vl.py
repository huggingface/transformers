# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
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
"""MiniMax M3 VL: vision tower + M3 (mixed sparse/dense MoE) text backbone."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicLayer
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig
from ..gemma3.modeling_gemma3 import Gemma3RMSNorm
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaModel
from ..minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
from ..minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2Experts,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2PreTrainedModel,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2SparseMoeBlock,
    MiniMaxM2TopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLTextConfig(MiniMaxM2Config):
    r"""
    dense_intermediate_size (`int`, *optional*, defaults to 12288):
        Intermediate size of the dense MLP used on layers where ``moe_layer_freq[i] == 0``.
    shared_intermediate_size (`int`, *optional*, defaults to 3072):
        Intermediate size of a single shared expert in the MoE layers.
    use_routing_bias (`bool`, *optional*, defaults to `True`):
        Whether the MoE router adds a learned per-expert bias before top-k selection.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of head channels rotated by RoPE; the remaining channels are passed through unchanged.
    swiglu_alpha (`float`, *optional*, defaults to 1.702):
        Sigmoid gain of the SwiGLU-OAI activation.
    swiglu_limit (`float`, *optional*, defaults to 7.0):
        Clamp bound applied to the gate and up projections of the SwiGLU-OAI activation.
    moe_layer_freq (`list[int]`, *optional*):
        Per-layer flags (`0`/`1`) selecting a dense MLP (`0`) or a sparse MoE block (`1`).
    sparse_attention_config (`dict`, *optional*):
        Configuration of the lightning sparse attention (top-k, indexer dims, local/init window, frequency).
    num_mtp_modules (`int`, *optional*, defaults to 0):
        Number of multi-token-prediction modules in the checkpoint; ignored at inference.
    """

    model_type = "minimax_m3_vl_text"
    base_config_key = "text_config"

    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 524288
    vocab_size: int = 200064
    rms_norm_eps: float = 1e-06
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    rotary_dim: int = 64
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    moe_layer_freq: list[int] | None = None
    sparse_attention_config: dict | None = None
    layer_types: list[str] | None = None
    num_mtp_modules: int = 0
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    rope_parameters: RopeParameters | dict | None = None


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLVisionConfig(PreTrainedConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period of the vision tower's 3D rotary position embedding.
    """

    model_type = "minimax_m3_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 2016
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-05
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLConfig(PreTrainedConfig):
    r"""Composite config for MiniMax M3 VL (vision tower + M3 LLM)."""

    model_type = "minimax_m3_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 200025
    video_token_index: int = 200026
    projector_hidden_size: int = 6144
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # The snapshot bundles its sub-configs with their original model_types
        # (e.g. ``clip_vision_model`` for the ViT, ``minimax_m2`` for the LLM).
        # We always rebuild them as our own classes so the resulting attributes
        # carry the fields our model code expects.
        if isinstance(self.vision_config, dict):
            self.vision_config.pop("model_type", None)
            self.vision_config = MiniMaxM3VLVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = MiniMaxM3VLVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config.pop("model_type", None)
            self.text_config = MiniMaxM3VLTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = MiniMaxM3VLTextConfig()

        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__post_init__(**kwargs)


# ---------------------------------------------------------------------------
# Per-layer cache for sparse-attention layers
# ---------------------------------------------------------------------------


class MiniMaxM3VLSparseCacheLayer(DynamicLayer):
    """Cache layer for M3 sparse-attention layers: standard DynamicLayer for the
    main attention plus ``idx_keys`` / ``idx_values`` slots for the lightning
    indexer's K/V (one head, ``sparse_index_dim`` per token).

    Same dispatch story as DeepSeek-V4's ``DeepseekV4CSACache``: the class
    registers itself via ``layer_type = "minimax_m3_sparse"`` so
    ``DynamicCache(config=text_config)`` picks it for each layer where
    ``text_config.layer_types[i] == "minimax_m3_sparse"``.
    """

    layer_type = "minimax_m3_sparse"

    def __init__(self, config: PreTrainedConfig | None = None):
        super().__init__(config)
        self.idx_keys: torch.Tensor | None = None
        self.idx_values: torch.Tensor | None = None

    def update_index(
        self, idx_k: torch.Tensor, idx_v: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Append the new token's ``idx_k`` / ``idx_v`` to the cache and return
        the full cached history. ``idx_v`` is ``None`` for layers with
        ``disable_index_value`` set; we keep it ``None`` end-to-end.
        """
        if self.idx_keys is None:
            self.idx_keys = idx_k
        else:
            self.idx_keys = torch.cat([self.idx_keys, idx_k], dim=-2)

        if idx_v is None:
            return self.idx_keys, None
        if self.idx_values is None:
            self.idx_values = idx_v
        else:
            self.idx_values = torch.cat([self.idx_values, idx_v], dim=-2)
        return self.idx_keys, self.idx_values

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys.index_select(0, beam_idx.to(self.idx_keys.device))
        if self.idx_values is not None:
            self.idx_values = self.idx_values.index_select(0, beam_idx.to(self.idx_values.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys.repeat_interleave(repeats, dim=0)
        if self.idx_values is not None:
            self.idx_values = self.idx_values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys[indices, ...]
        if self.idx_values is not None:
            self.idx_values = self.idx_values[indices, ...]

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.idx_keys is not None and self.idx_keys.shape[-2] > max_length:
            self.idx_keys = self.idx_keys[..., :max_length, :]
        if self.idx_values is not None and self.idx_values.shape[-2] > max_length:
            self.idx_values = self.idx_values[..., :max_length, :]


# ---------------------------------------------------------------------------
# Text branch: activation + norm helpers
# ---------------------------------------------------------------------------


def _swigluoai(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """SwiGLU with output clamp (M3-style, non-interleaved gate/up layout)."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


class MiniMaxM3VLRMSNorm(Gemma3RMSNorm):
    """Gemma-style RMSNorm: normalizes in fp32 and scales by ``weight + 1``."""


# ---------------------------------------------------------------------------
# Text branch: dense MLP + MoE
# ---------------------------------------------------------------------------


class MiniMaxM3VLDenseMLP(nn.Module):
    """Dense feed-forward used for layers where ``moe_layer_freq[i] == 0``."""

    def __init__(self, config: MiniMaxM3VLTextConfig, intermediate_size: int | None = None):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        return self.down_proj(_swigluoai(gate_up, self.swiglu_alpha, self.swiglu_limit))


class MiniMaxM3VLExperts(MiniMaxM2Experts):
    """M3 experts: packed Mixtral layout + swigluoai gate."""

    def __init__(self, config: MiniMaxM3VLTextConfig):
        # We can't reuse the M2 init because it calls ACT2FN[config.hidden_act],
        # and ``swigluoai`` is not registered there (the alpha/limit clamp makes
        # it stateful enough that bundling it into the experts is cleaner).
        nn.Module.__init__(self)
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        return _swigluoai(gate_up, self.swiglu_alpha, self.swiglu_limit)


class MiniMaxM3VLTopKRouter(MiniMaxM2TopKRouter):
    pass


class MiniMaxM3VLSparseMoeBlock(MiniMaxM2SparseMoeBlock):
    """M3 MoE block: M2 base + shared expert (``n_shared_experts`` is always >= 1)."""

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.shared_experts = MiniMaxM3VLDenseMLP(
            config, intermediate_size=config.shared_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states_flat, self.e_score_correction_bias)
        top_k_weights = top_k_weights * self.routed_scaling_factor
        routed = self.experts(hidden_states_flat, top_k_index, top_k_weights)
        routed = routed + self.shared_experts(hidden_states_flat)
        return routed.reshape(batch_size, sequence_length, hidden_dim)


# ---------------------------------------------------------------------------
# Text branch: attention (partial RoPE + per-head Gemma QK norm)
# ---------------------------------------------------------------------------


def _apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate only the first ``rotary_dim`` channels of each head."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_r, sin_r = cos[..., :rotary_dim], sin[..., :rotary_dim]
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos_r, sin_r)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class MiniMaxM3VLRotaryEmbedding(MiniMaxM2RotaryEmbedding):
    pass


class MiniMaxM3VLAttention(MiniMaxM2Attention):
    """M3 attention: per-head Gemma QK-norm + partial RoPE.

    Overrides the inherited FlexOlmo-style forward because:
      * per-head QK norm requires reshaping to ``[..., num_heads, head_dim]``
        before the norm (FlexOlmo applies a flat per-layer norm);
      * RoPE rotates only the first ``rotary_dim`` channels of each head.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.rotary_dim = config.rotary_dim
        # Replace the inherited (per-layer) q_norm/k_norm with per-head Gemma norms.
        self.q_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_partial_rope(query_states, key_states, cos, sin, self.rotary_dim)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class MiniMaxM3VLIndexer(nn.Module):
    r"""Lightning Indexer for MiniMax M3 sparse attention.

    Picks the top-``sparse_topk_blocks`` key blocks of size ``sparse_block_size``
    per query, plus the first ``sparse_init_block`` blocks and the
    ``sparse_local_block`` blocks immediately preceding the query block, which
    are always visible. Returns:

      * ``idx_output`` — projection of the index branch's own SDPA over
        ``(idx_q, idx_k, idx_v)``, to be added to the main attention output.
      * ``block_bias`` — ``[B, 1, S, S]`` mask with ``0`` at every (query, key)
        pair allowed under the top-k + init + local rule, ``-inf`` elsewhere,
        ready to add onto ``attention_mask`` before the main SDPA (same trick
        as [`DeepseekV4Indexer`]).

    When ``disable_index_value`` is set for this layer, the indexer still
    scores blocks (with ``idx_q``/``idx_k``) but skips the value branch, so
    ``idx_output`` is ``None``.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, disable_index_value: bool = False):
        super().__init__()
        sparse_cfg = config.sparse_attention_config
        self.disable_index_value = bool(disable_index_value)
        self.num_heads = int(sparse_cfg["sparse_num_index_heads"])
        self.head_dim = int(sparse_cfg["sparse_index_dim"])
        self.block_size = int(sparse_cfg["sparse_block_size"])
        self.topk_blocks = int(sparse_cfg["sparse_topk_blocks"])
        self.init_blocks = int(sparse_cfg.get("sparse_init_block", 0))
        self.local_blocks = int(sparse_cfg.get("sparse_local_block", 1))
        self.score_type = sparse_cfg.get("sparse_score_type", "max")

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.q_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if not self.disable_index_value:
            self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def _block_bias(self, idx_q: torch.Tensor, idx_k: torch.Tensor, q_positions: torch.Tensor) -> torch.Tensor:
        r"""Build the ``[B, 1, S_q, S_k]`` top-k + init + local additive mask.

        ``idx_q``: ``[B, H_idx, S_q, D]`` — only the *new* queries.
        ``idx_k``: ``[B, 1, S_k, D]`` — the *full* cached key history.
        ``q_positions``: ``[S_q]`` absolute positions of the queries, so the
        causal block threshold lines up with the cached history during decode.
        """
        B, H, Sq, _ = idx_q.shape
        Sk = idx_k.shape[2]
        block = self.block_size
        pad = (-Sk) % block
        if pad:
            idx_k = F.pad(idx_k, (0, 0, 0, pad))
        n_blocks = (Sk + pad) // block

        # Inner-product scores: ``[B, H, Sq, n_blocks * block]`` -> max over block.
        idx_k_h = idx_k.expand(-1, H, -1, -1)
        scores_qk = torch.matmul(idx_q.float(), idx_k_h.float().transpose(-1, -2))
        scores_qk = scores_qk.view(B, H, Sq, n_blocks, block)
        if self.score_type == "max":
            block_scores = scores_qk.amax(dim=-1)
        else:
            block_scores = scores_qk.softmax(dim=-1).sum(dim=-1)
        block_scores = block_scores.amax(dim=1)  # max over index heads -> [B, Sq, n_blocks]

        # Causality on absolute positions.
        q_block = q_positions // block  # [Sq]
        block_idx = torch.arange(n_blocks, device=idx_q.device)
        future_mask = block_idx.view(1, 1, -1) > q_block.view(1, -1, 1)
        block_scores = block_scores.masked_fill(future_mask, float("-inf"))

        topk = min(self.topk_blocks, n_blocks)
        topk_idx = block_scores.topk(topk, dim=-1).indices  # [B, Sq, topk]

        block_keep = torch.zeros((B, Sq, n_blocks), dtype=torch.bool, device=idx_q.device)
        block_keep.scatter_(-1, topk_idx, True)
        if self.init_blocks > 0:
            block_keep[..., : self.init_blocks] = True
        if self.local_blocks > 0:
            local_offset = torch.arange(self.local_blocks, device=idx_q.device)
            local_idx = (q_block.view(-1, 1) - local_offset.view(1, -1)).clamp(min=0)
            block_keep.scatter_(-1, local_idx.unsqueeze(0).expand(B, -1, -1), True)

        token_keep = block_keep.repeat_interleave(block, dim=-1)[..., :Sk]
        return torch.zeros_like(token_keep, dtype=idx_q.dtype).masked_fill_(~token_keep, float("-inf")).unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        bsz, slen, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(bsz, slen, self.num_heads, self.head_dim)
        idx_k = self.k_proj(hidden_states).view(bsz, slen, 1, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)  # [B, H_idx, Sq, D]
        idx_k = self.k_norm(idx_k).transpose(1, 2)  # [B, 1, Sq, D]
        cos, sin = position_embeddings
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

        idx_v = None
        if not self.disable_index_value:
            idx_v = self.v_proj(hidden_states).view(bsz, slen, 1, self.head_dim).transpose(1, 2)

        # Append to cache (or run stateless if no cache is provided).
        cache_layer: MiniMaxM3VLSparseCacheLayer | None = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        if cache_layer is not None:
            idx_k_full, idx_v_full = cache_layer.update_index(idx_k, idx_v)
        else:
            idx_k_full, idx_v_full = idx_k, idx_v

        # Absolute query positions; default to ``arange`` if the model didn't pass them.
        if position_ids is None:
            q_positions = torch.arange(slen, device=idx_q.device)
        else:
            q_positions = position_ids[0]

        block_bias = self._block_bias(idx_q, idx_k_full, q_positions)

        idx_output: torch.Tensor | None = None
        if not self.disable_index_value:
            idx_k_e = idx_k_full.expand(-1, self.num_heads, -1, -1)
            idx_v_e = idx_v_full.expand(-1, self.num_heads, -1, -1)
            # Build the index-branch causal mask explicitly (Sq < Sk during decode,
            # so ``is_causal=True`` would mis-align the diagonal).
            Sq, Sk = idx_q.shape[2], idx_k_e.shape[2]
            causal = torch.full((Sq, Sk), float("-inf"), device=idx_q.device, dtype=idx_q.dtype)
            k_pos = torch.arange(Sk, device=idx_q.device)
            causal = causal.masked_fill(k_pos.view(1, -1) <= q_positions.view(-1, 1), 0.0)
            idx_attn = F.scaled_dot_product_attention(idx_q, idx_k_e, idx_v_e, attn_mask=causal, is_causal=False)
            idx_attn = idx_attn.transpose(1, 2).reshape(bsz, slen, self.num_heads * self.head_dim)
            idx_output = self.o_proj(idx_attn)

        return idx_output, block_bias


class MiniMaxM3VLSparseAttention(MiniMaxM3VLAttention):
    """Sparse-attention layer: main attention masked to top-k key blocks by the
    [`MiniMaxM3VLIndexer`], plus the indexer's own attention output added as a
    residual (when ``disable_index_value`` is False).

    The block-selection follows the same scatter-into-block-bias pattern as
    [`DeepseekV4Indexer`]: build a ``[B, 1, S, S]`` additive mask that is ``0``
    where each query is allowed to attend (top-k + init + local blocks) and
    ``-inf`` elsewhere, then add it onto the standard ``attention_mask``
    before SDPA.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        layer_idx: int,
        disable_index_value: bool = False,
    ):
        super().__init__(config, layer_idx)
        self.indexer = MiniMaxM3VLIndexer(config, disable_index_value=disable_index_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        idx_output, block_bias = self.indexer(
            hidden_states,
            position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            layer_idx=self.layer_idx,
        )

        # Same trick as deepseek_v4: encode top-k block visibility as a `-inf`
        # additive mask, sum it onto the regular causal mask. SDPA then sees a
        # mask that already excludes every key outside the top-k blocks.
        if attention_mask is None:
            merged_mask = block_bias
        else:
            merged_mask = attention_mask.to(block_bias.dtype) + block_bias

        attn_output, attn_weights = super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=merged_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if idx_output is not None:
            attn_output = attn_output + idx_output
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Text branch: decoder layer + model
# ---------------------------------------------------------------------------


class MiniMaxM3VLDecoderLayer(GradientCheckpointingLayer):
    """M3 decoder layer: per-layer dense/MoE MLP and dense/sparse attention."""

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        eps = config.rms_norm_eps
        sparse_cfg = config.sparse_attention_config

        is_sparse_attn = bool(sparse_cfg["sparse_attention_freq"][layer_idx])
        if is_sparse_attn:
            self.self_attn = MiniMaxM3VLSparseAttention(
                config,
                layer_idx,
                disable_index_value=bool(sparse_cfg["sparse_disable_index_value"][layer_idx]),
            )
        else:
            self.self_attn = MiniMaxM3VLAttention(config, layer_idx)

        if config.moe_layer_freq[layer_idx]:
            self.mlp = MiniMaxM3VLSparseMoeBlock(config)
        else:
            self.mlp = MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)

        self.input_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # ``position_ids`` is consumed by the sparse-attention indexer (used for
        # absolute-position causality on the cached idx_k history). The dense
        # attention path ignores it.
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMaxM3VLPreTrainedModel(MiniMaxM2PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    input_modalities = ("image", "video", "text")
    # MTP modules ship in the upstream checkpoint but aren't part of this port.
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, MiniMaxM3VLExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM3VLTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM3VLSparseMoeBlock):
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, MiniMaxM3VLRMSNorm):
            init.zeros_(module.weight)


class MiniMaxM3VLTextModel(MiniMaxM2Model):
    """Stand-alone text backbone (no LM head). Used by [`MiniMaxM3VLModel`]."""

    config: MiniMaxM3VLTextConfig

    def __init__(self, config: MiniMaxM3VLTextConfig):
        # Derive layer_types from sparse_attention_freq so DynamicCache(config=...)
        # dispatches the per-layer sparse cache for sparse-attention layers.
        if config.layer_types is None and config.sparse_attention_config is not None:
            freq = config.sparse_attention_config.get("sparse_attention_freq")
            if freq is not None:
                config.layer_types = ["minimax_m3_sparse" if f else "full_attention" for f in freq]
        super().__init__(config)
        self.layers = nn.ModuleList([MiniMaxM3VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxM3VLForCausalLM(MiniMaxM2ForCausalLM):
    """Text-only causal LM head."""

    config: MiniMaxM3VLTextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMaxM3VLTextConfig):
        # The M2 parent's substitution table would point ``self.model`` at the
        # composite VL model; force the text backbone here.
        super().__init__(config)
        self.model = MiniMaxM3VLTextModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class MiniMaxM3VLVisionEmbeddings(nn.Module):
    """Conv3d patch embedding over a flat ``[N_patches, C * T * P * P]`` input."""

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=config.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        n = pixel_values.shape[0]
        pixel_values = pixel_values.view(
            n, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        out = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        return out.reshape(n, -1)


class MiniMaxM3VL3DRotaryEmbedding(nn.Module):
    """3D RoPE over (T, H, W); splits ``head_dim`` into three axes."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        d_per_axis = (head_dim // 3) // 2 * 2
        self.dims = (d_per_axis, d_per_axis, head_dim - 2 * d_per_axis)
        self.theta = theta

    def _axis_inv_freq(self, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dim == 0:
            return torch.empty(0, device=device, dtype=dtype)
        return 1.0 / (self.theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))

    def forward(
        self, grid_thw: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords_list = []
        for t, h, w in grid_thw.tolist():
            ti = torch.arange(t, device=device, dtype=dtype).repeat_interleave(h * w)
            hi = torch.arange(h, device=device, dtype=dtype).repeat_interleave(w).repeat(t)
            wi = torch.arange(w, device=device, dtype=dtype).repeat(t * h)
            coords_list.append(torch.stack([ti, hi, wi], dim=-1))
        coords = torch.cat(coords_list, dim=0)

        parts_cos: list[torch.Tensor] = []
        parts_sin: list[torch.Tensor] = []
        for axis, dim in enumerate(self.dims):
            if dim == 0:
                continue
            inv = self._axis_inv_freq(dim, device, dtype)
            freqs = coords[:, axis : axis + 1] * inv[None, :]
            parts_cos.append(freqs.cos().repeat_interleave(2, dim=-1))
            parts_sin.append(freqs.sin().repeat_interleave(2, dim=-1))
        return torch.cat(parts_cos, dim=-1), torch.cat(parts_sin, dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_vision_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class MiniMaxM3VLVisionAttention(nn.Module):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        q, k = _apply_vision_rope(q, k, *position_embeddings)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout)
        return self.out_proj(out.transpose(1, 2).reshape(bsz, seq, self.embed_dim))


class MiniMaxM3VLVisionMLP(nn.Module):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MiniMaxM3VLVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = MiniMaxM3VLVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MiniMaxM3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), position_embeddings, attention_mask
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


@auto_docstring
class MiniMaxM3VLVisionModel(MiniMaxM3VLPreTrainedModel):
    """CLIP-like vision tower with Conv3d patch embed + 3D RoPE."""

    config: MiniMaxM3VLVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.embeddings = MiniMaxM3VLVisionEmbeddings(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MiniMaxM3VL3DRotaryEmbedding(head_dim, theta=config.rope_theta)
        # Snapshot keeps the CLIP-style ``pre_layrnorm`` (yes, that's the upstream
        # spelling) applied to patch embeddings before the encoder stack. There is
        # *no* post-encoder norm — the last encoder layer feeds the projector
        # directly.
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([MiniMaxM3VLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        embeds = self.embeddings(pixel_values)
        cos, sin = self.rotary_emb(image_grid_thw, device=embeds.device, dtype=torch.float32)
        hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=(cos, sin))
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:, 0],
        )


# ---------------------------------------------------------------------------
# Multimodal: patch merger + projector + composite model
# ---------------------------------------------------------------------------


class MiniMaxM3VLPatchMerger(nn.Module):
    """Group ``spatial_merge_size**2`` patches into the channel dim, then a 2-MLP."""

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__()
        text_hidden = config.text_config.hidden_size
        merge_size = config.vision_config.spatial_merge_size
        self.spatial_merge_size = merge_size
        self.linear_1 = nn.Linear(text_hidden * (merge_size**2), config.projector_hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.projector_hidden_size, text_hidden, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        x = image_features.reshape(image_features.shape[0] // (self.spatial_merge_size**2), -1)
        return self.linear_2(F.gelu(self.linear_1(x)))


class MiniMaxM3VLMultiModalProjector(nn.Module):
    """2-layer projector from vision hidden_size to text hidden_size."""

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.projector_hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(image_features)))


@dataclass
@auto_docstring(custom_intro="MiniMax M3 VL model output (without LM head).")
class MiniMaxM3VLModelOutputWithPast(BaseModelOutputWithPast):
    r"""image_hidden_states: Image features from the vision tower after projection."""

    image_hidden_states: torch.FloatTensor | None = None


@dataclass
@auto_docstring(custom_intro="MiniMax M3 VL causal LM output.")
class MiniMaxM3VLCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None


@auto_docstring(custom_intro="MiniMax M3 VL backbone (vision + projector + text), without LM head.")
class MiniMaxM3VLModel(LlavaModel):
    config: MiniMaxM3VLConfig

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__(config)
        self.vision_tower = MiniMaxM3VLVisionModel(config.vision_config)
        self.multi_modal_projector = MiniMaxM3VLMultiModalProjector(config)
        self.patch_merge_mlp = MiniMaxM3VLPatchMerger(config)
        self.language_model = MiniMaxM3VLTextModel(config.text_config)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        vision_out = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        # vision_out is [1, seq, vision_hidden] -> project -> spatial merge.
        hidden_states = self.multi_modal_projector(vision_out.last_hidden_state.squeeze(0))
        return self.patch_merge_mlp(hidden_states)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniMaxM3VLModelOutputWithPast:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return MiniMaxM3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            image_hidden_states=image_features,
        )


@auto_docstring(custom_intro="MiniMax M3 VL full model with LM head (text + vision).")
class MiniMaxM3VLForConditionalGeneration(LlavaForConditionalGeneration):
    config: MiniMaxM3VLConfig

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniMaxM3VLCausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return MiniMaxM3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "MiniMaxM3VLConfig",
    "MiniMaxM3VLTextConfig",
    "MiniMaxM3VLVisionConfig",
    "MiniMaxM3VLForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLPreTrainedModel",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLVisionModel",
]
