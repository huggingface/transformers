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
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..clip.modeling_clip import CLIPAttention, CLIPEncoder, CLIPEncoderLayer, CLIPMLP
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

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)
        # ``layer_types`` is the canonical per-layer attention dispatch: it tells
        # ``DynamicCache(config=...)`` which layers want the sparse cache and lets
        # the decoder pick ``MiniMaxM3VLSparseAttention`` vs. plain attention. We
        # derive it once here from ``sparse_attention_freq`` so model code never
        # has to re-read the sparse-attention frequency list.
        if self.layer_types is None and self.sparse_attention_config is not None:
            freq = self.sparse_attention_config.get("sparse_attention_freq")
            if freq is not None:
                self.layer_types = ["minimax_m3_sparse" if f else "full_attention" for f in freq]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


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
    hidden_act: str = "gelu"
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
    """Cache layer for M3 sparse-attention layers: a standard ``DynamicLayer``
    for the main attention plus an ``idx_keys`` slot holding the lightning
    indexer's keys (one head, ``sparse_index_dim`` per token).

    Same dispatch story as DeepSeek-V4's ``DeepseekV4CSACache``: the class
    registers itself via ``layer_type = "minimax_m3_sparse"`` so
    ``DynamicCache(config=text_config)`` picks it for each layer where
    ``text_config.layer_types[i] == "minimax_m3_sparse"``.
    """

    layer_type = "minimax_m3_sparse"

    def __init__(self, config: PreTrainedConfig | None = None):
        super().__init__(config)
        self.idx_keys: torch.Tensor | None = None

    def update_index(self, idx_k: torch.Tensor) -> torch.Tensor:
        """Append the new token's ``idx_k`` to the cache and return the full history."""
        self.idx_keys = idx_k if self.idx_keys is None else torch.cat([self.idx_keys, idx_k], dim=-2)
        return self.idx_keys

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys.index_select(0, beam_idx.to(self.idx_keys.device))

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys[indices, ...]

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.idx_keys is not None and self.idx_keys.shape[-2] > max_length:
            self.idx_keys = self.idx_keys[..., :max_length, :]


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
    """M3 experts: packed Mixtral layout, gated by the SwiGLU-OAI activation.

    The only architectural difference from [`MiniMaxM2Experts`] is the gate: M3
    fuses gate and up with the clamped SwiGLU-OAI activation, which needs both
    halves at once, so we route through the ``_apply_gate`` hook (also used by
    the kernelized expert implementations) instead of a plain ``act_fn``.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig):
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

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_state = self._apply_gate(F.linear(current_state, self.gate_up_proj[expert_idx]))
            current_state = F.linear(current_state, self.down_proj[expert_idx])
            current_state = current_state * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_state.to(final_hidden_states.dtype))

        return final_hidden_states


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

    Scores key *blocks* of size ``sparse_block_size`` against each query with a
    small ``sparse_num_index_heads``-head dot-product branch, then keeps, per
    query, the top-``sparse_topk_blocks`` blocks plus the first
    ``sparse_init_block`` blocks and the ``sparse_local_block`` blocks
    immediately preceding the query (always visible). It returns a
    ``[B, 1, S_q, S_k]`` additive ``block_bias`` that is ``0`` at every allowed
    (query, key) pair and ``-inf`` elsewhere, to be summed onto the main
    attention mask — the same scatter-into-``-inf``-bias trick as
    [`DeepseekV4Indexer`].

    Like DeepSeek-V4's indexer this is purely a *selection* branch: it has no
    value projection and produces no residual output of its own (the upstream
    checkpoint disables the index-value path on every sparse layer).
    """

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__()
        sparse_cfg = config.sparse_attention_config
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

    def _block_bias(self, idx_q: torch.Tensor, idx_k: torch.Tensor, q_positions: torch.Tensor) -> torch.Tensor:
        r"""Build the ``[B, 1, S_q, S_k]`` top-k + init + local additive mask.

        ``idx_q``: ``[B, H_idx, S_q, D]`` — only the *new* queries.
        ``idx_k``: ``[B, 1, S_k, D]`` — the *full* cached key history.
        ``q_positions``: ``[S_q]`` absolute positions of the queries, so the
        causal block threshold lines up with the cached history during decode.
        """
        batch, heads, q_len, _ = idx_q.shape
        k_len = idx_k.shape[2]
        block = self.block_size
        n_blocks = -(-k_len // block)  # ceil-div
        pad = n_blocks * block - k_len

        # Per-(head, query) block score = pool over the block's key tokens, then
        # reduce over index heads. Pad keys with ``-inf`` so the trailing partial
        # block never wins a top-k slot on padding.
        scores = torch.matmul(idx_q.float(), idx_k.expand(-1, heads, -1, -1).float().transpose(-1, -2))
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        scores = scores.view(batch, heads, q_len, n_blocks, block)
        block_scores = scores.amax(dim=-1) if self.score_type == "max" else scores.softmax(dim=-1).sum(dim=-1)
        block_scores = block_scores.amax(dim=1)  # -> [B, S_q, n_blocks]

        # Block-level causality on absolute positions.
        q_block = q_positions // block  # [S_q]
        future = torch.arange(n_blocks, device=idx_q.device).view(1, 1, -1) > q_block.view(1, -1, 1)
        block_scores = block_scores.masked_fill(future, float("-inf"))

        # Same scatter-into-``-inf``-bias idea as deepseek_v4: start all-masked,
        # punch ``0`` at the selected blocks (top-k, then the always-on init and
        # local windows), then re-mask the future so a short history can't leak.
        bias = block_scores.new_full((batch, q_len, n_blocks), float("-inf"))
        bias.scatter_(-1, block_scores.topk(min(self.topk_blocks, n_blocks), dim=-1).indices, 0.0)
        if self.init_blocks > 0:
            bias[..., : self.init_blocks] = 0.0
        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=idx_q.device)
            local = (q_block.view(-1, 1) - local.view(1, -1)).clamp(min=0)  # [S_q, local]
            bias.scatter_(-1, local.unsqueeze(0).expand(batch, -1, -1), 0.0)
        bias = bias.masked_fill(future, float("-inf"))

        token_bias = bias.repeat_interleave(block, dim=-1)[..., :k_len]
        return token_bias.to(idx_q.dtype).unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        bsz, slen, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(bsz, slen, self.num_heads, self.head_dim)
        idx_k = self.k_proj(hidden_states).view(bsz, slen, 1, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)  # [B, H_idx, Sq, D]
        idx_k = self.k_norm(idx_k).transpose(1, 2)  # [B, 1, Sq, D]
        cos, sin = position_embeddings
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

        # Cache the indexer keys (one head, ``sparse_index_dim`` per token) so a
        # decode-step query scores against the full history.
        cache_layer: MiniMaxM3VLSparseCacheLayer | None = (
            past_key_values.layers[layer_idx] if past_key_values is not None else None
        )
        idx_k = cache_layer.update_index(idx_k) if cache_layer is not None else idx_k

        q_positions = torch.arange(slen, device=idx_q.device) if position_ids is None else position_ids[0]
        return self._block_bias(idx_q, idx_k, q_positions)


class MiniMaxM3VLSparseAttention(MiniMaxM3VLAttention):
    """Sparse-attention layer: standard M3 attention restricted to the key
    blocks selected by the [`MiniMaxM3VLIndexer`].

    The indexer returns a ``[B, 1, S, S]`` additive ``block_bias`` (``0`` where a
    query may attend, ``-inf`` elsewhere); we sum it onto the regular causal
    ``attention_mask`` so SDPA already excludes every key outside the top-k
    blocks — the same approach as [`DeepseekV4Attention`].
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = MiniMaxM3VLIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        block_bias = self.indexer(
            hidden_states,
            position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            layer_idx=self.layer_idx,
        )
        if attention_mask is not None:
            block_bias = block_bias + attention_mask.to(block_bias.dtype)

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=block_bias,
            past_key_values=past_key_values,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Text branch: decoder layer + model
# ---------------------------------------------------------------------------


class MiniMaxM3VLDecoderLayer(GradientCheckpointingLayer):
    """M3 decoder layer: per-layer dense/MoE MLP and dense/sparse attention."""

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        eps = config.rms_norm_eps

        if config.layer_types[layer_idx] == "minimax_m3_sparse":
            self.self_attn = MiniMaxM3VLSparseAttention(config, layer_idx)
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


class MiniMaxM3VLVisionAttention(CLIPAttention):
    """CLIP-style vision attention; the only difference from [`CLIPAttention`] is
    that queries and keys are rotated by the tower's 3D RoPE before the
    (interface-dispatched) scaled dot-product attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        queries = self.q_proj(hidden_states).view(hidden_shape)
        keys = self.k_proj(hidden_states).view(hidden_shape)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        queries, keys = _apply_vision_rope(queries, keys, *position_embeddings)
        queries, keys = queries.transpose(1, 2), keys.transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output), attn_weights


class MiniMaxM3VLVisionMLP(CLIPMLP):
    pass


class MiniMaxM3VLVisionEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.self_attn = MiniMaxM3VLVisionAttention(config)
        self.mlp = MiniMaxM3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            self.layer_norm1(hidden_states), position_embeddings, attention_mask, **kwargs
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class MiniMaxM3VLVisionEncoder(CLIPEncoder):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniMaxM3VLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, position_embeddings, attention_mask, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


class MiniMaxM3VLVisionTransformer(nn.Module):
    """CLIP-style vision transformer with a Conv3d patch embed and 3D RoPE.

    No post-encoder norm: the last encoder layer feeds the projector directly.
    ``pre_layrnorm`` keeps the upstream CLIP spelling.
    """

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.embeddings = MiniMaxM3VLVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = MiniMaxM3VLVisionEncoder(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MiniMaxM3VL3DRotaryEmbedding(head_dim, theta=config.rope_theta)

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        embeds = self.embeddings(pixel_values)
        cos, sin = self.rotary_emb(image_grid_thw, device=embeds.device, dtype=torch.float32)
        hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
        cos, sin = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, position_embeddings=(cos, sin), **kwargs)
        last_hidden_state = encoder_outputs.last_hidden_state
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=last_hidden_state[:, 0])


@auto_docstring
class MiniMaxM3VLVisionModel(MiniMaxM3VLPreTrainedModel):
    """CLIP-like vision tower with Conv3d patch embed + 3D RoPE."""

    config: MiniMaxM3VLVisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {
        "hidden_states": MiniMaxM3VLVisionEncoderLayer,
        "attentions": MiniMaxM3VLVisionAttention,
    }

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.vision_model = MiniMaxM3VLVisionTransformer(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)


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
        self.patch_merge = MiniMaxM3VLPatchMerger(config)
        self.language_model = MiniMaxM3VLTextModel(config.text_config)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of each image's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        # Return the raw vision-tower output (so callers can inspect hidden states /
        # attentions) while stashing the projected + spatially-merged features —
        # ready to scatter into the text embeddings — in ``pooler_output``.
        vision_outputs = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        hidden_states = self.multi_modal_projector(vision_outputs.last_hidden_state.squeeze(0))
        vision_outputs.pooler_output = self.patch_merge(hidden_states)
        return vision_outputs

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
            image_features = self.get_image_features(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
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
