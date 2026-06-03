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

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig, AutoModel
from ..minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
from ..minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2DecoderLayer,
    MiniMaxM2Experts,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2PreTrainedModel,
    MiniMaxM2RMSNorm,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2SparseMoeBlock,
    MiniMaxM2TopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from typing import Callable


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLTextConfig(MiniMaxM2Config):
    r"""Text-side config for MiniMax M3 VL.

    Extends [`MiniMaxM2Config`] with the M3-specific knobs: shared experts,
    swiglu-with-limit activation, partial rotary embeddings, per-head QK
    normalization, dense MLP layers for the first ``len([f for f in moe_layer_freq if f == 0])``
    layers, and the optional ``sparse_attention_config`` lightning-index branch.
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
    hidden_act: str = "swigluoai"
    max_position_embeddings: int = 524288
    vocab_size: int = 200064
    rms_norm_eps: float = 1e-06
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    use_routing_bias: bool = True
    scoring_func: Literal["sigmoid", "softmax"] = "sigmoid"
    routed_scaling_factor: float = 2.0
    use_qk_norm: bool = True
    qk_norm_type: Literal["per_layer", "per_head", "multi_head"] = "per_head"
    use_gemma_norm: bool = True
    attention_output_gate: bool = False
    rotary_dim: int = 64
    partial_rotary_factor: float = 0.5
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    moe_layer_freq: list[int] | None = None
    sparse_attention_config: dict | None = None
    num_mtp_modules: int = 0
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    rope_parameters: RopeParameters | dict | None = None


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLVisionConfig(PreTrainedConfig):
    r"""Vision-side config for MiniMax M3 VL.

    CLIP-style ViT with 3D RoPE on the (T, H, W) patch grid and a Conv3d
    patch embedding. Used by [`MiniMaxM3VLVisionModel`].
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
    rope_mode: str = "3d"
    vision_segment_max_frames: int = 4
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
    image_seq_length: int = 576
    process_image_mode: str = "dynamic_res"
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int | None = 6144
    multimodal_projector_bias: bool = True
    vision_feature_layer: int = -1
    vision_feature_select_strategy: Literal["default", "full"] = "full"
    img_token_compression_config: dict | None = None
    image_grid_pinpoints: str | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config.setdefault("model_type", "minimax_m3_vl_vision")
            self.vision_config = MiniMaxM3VLVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = MiniMaxM3VLVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config.setdefault("model_type", "minimax_m3_vl_text")
            self.text_config = MiniMaxM3VLTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = MiniMaxM3VLTextConfig()

        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        if self.img_token_compression_config is None:
            self.img_token_compression_config = {
                "image_token_compression_method": "patch_merge",
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            }

        super().__post_init__(**kwargs)


# ---------------------------------------------------------------------------
# Text branch: dense MLP + swigluoai
# ---------------------------------------------------------------------------


def _swigluoai(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """SwiGLU with output clamp, as in GPT-OSS / MiniMax M3.

    ``gate_up`` is ``[..., 2 * I]`` with gate first, then up (non-interleaved
    layout — matches sglang's ``swiglu_no_interleaved_with_alpha_and_limit``).
    """
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1.0) * glu


class MiniMaxM3VLDenseMLP(nn.Module):
    """Dense feed-forward used for the first ``moe_layer_freq == 0`` layers."""

    def __init__(self, config: MiniMaxM3VLTextConfig, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        inter = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
        self.hidden_act = config.hidden_act
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Linear(hidden_size, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        if self.hidden_act != "swigluoai":
            self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        if self.hidden_act == "swigluoai":
            x = _swigluoai(gate_up, self.swiglu_alpha, self.swiglu_limit)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            x = self.act_fn(gate) * up
        return self.down_proj(x)


class MiniMaxM3VLExperts(MiniMaxM2Experts):
    """M3 experts: same packed-weight layout as Mixtral, swigluoai activation.

    Overrides ``__init__`` fully because Mixtral's ``ACT2FN[config.hidden_act]``
    lookup fails for ``hidden_act="swigluoai"`` (not registered as a stock
    activation; the alpha/limit clamping makes it weight-dependent).
    """

    def __init__(self, config: MiniMaxM3VLTextConfig):
        nn.Module.__init__(self)
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.hidden_act = config.hidden_act
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "swigluoai":
            return _swigluoai(gate_up, self.swiglu_alpha, self.swiglu_limit)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


class MiniMaxM3VLTopKRouter(MiniMaxM2TopKRouter):
    pass


class MiniMaxM3VLSparseMoeBlock(MiniMaxM2SparseMoeBlock):
    """M3 MoE block: M2 base + optional shared experts merged into the output."""

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.n_shared_experts:
            self.shared_experts = MiniMaxM3VLDenseMLP(
                config,
                intermediate_size=config.shared_intermediate_size * self.n_shared_experts,
            )
        else:
            self.shared_experts = None

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
        if self.shared_experts is not None:
            routed = routed + self.shared_experts(hidden_states_flat)
        return routed.reshape(batch_size, sequence_length, hidden_dim)


# ---------------------------------------------------------------------------
# Text branch: attention (partial rope + per-head QK norm + optional sparse index)
# ---------------------------------------------------------------------------


def _apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate only the first ``rotary_dim`` channels of each head."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class MiniMaxM3VLRotaryEmbedding(MiniMaxM2RotaryEmbedding):
    pass


class MiniMaxM3VLRMSNorm(MiniMaxM2RMSNorm):
    pass


class MiniMaxM3VLGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm (multiplies by ``weight + 1``)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.variance_epsilon)
        return (x32 * (self.weight.float() + 1.0)).to(orig_dtype)


def _build_norm(dim: int, eps: float, gemma: bool) -> nn.Module:
    return MiniMaxM3VLGemmaRMSNorm(dim, eps=eps) if gemma else MiniMaxM3VLRMSNorm(dim, eps=eps)


class MiniMaxM3VLAttention(MiniMaxM2Attention):
    """M3 attention: partial RoPE + per-head QK-norm (Gemma-style optional).

    Replaces the inherited FlexOlmo-style attention forward, because:
      * per-head QK norm requires reshaping to ``[..., num_heads, head_dim]``
        before the norm (FlexOlmo applies a single per-layer norm to the flat
        ``[..., num_heads * head_dim]``);
      * RoPE is partial: only the first ``rotary_dim`` channels of each head
        are rotated.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.use_qk_norm = config.use_qk_norm
        self.qk_norm_type = config.qk_norm_type
        self.use_gemma_norm = config.use_gemma_norm
        self.rotary_dim = config.rotary_dim
        eps = config.rms_norm_eps
        # Replace the inherited (per-layer) q_norm/k_norm with the right shape for M3.
        if self.use_qk_norm and self.qk_norm_type == "per_head":
            self.q_norm = _build_norm(self.head_dim, eps, self.use_gemma_norm)
            self.k_norm = _build_norm(self.head_dim, eps, self.use_gemma_norm)
        elif self.use_qk_norm and self.qk_norm_type == "per_layer":
            self.q_norm = _build_norm(config.num_attention_heads * self.head_dim, eps, self.use_gemma_norm)
            self.k_norm = _build_norm(config.num_key_value_heads * self.head_dim, eps, self.use_gemma_norm)
        elif self.use_qk_norm:
            raise ValueError(f"Unsupported qk_norm_type {self.qk_norm_type!r}")

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_qk_norm:
            return q, k
        # q/k are [batch, num_heads, seq, head_dim]
        return self.q_norm(q), self.k_norm(k)

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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = self._apply_qk_norm(query_states, key_states)

        cos, sin = position_embeddings
        if self.rotary_dim and self.rotary_dim < self.head_dim:
            query_states, key_states = _apply_partial_rope(
                query_states, key_states, cos, sin, self.rotary_dim
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiniMaxM3VLSparseAttention(MiniMaxM3VLAttention):
    """Sparse-attention layer: same dense math + an extra index branch.

    The original sglang implementation runs a lightning-index attention that
    picks top-k key blocks per query. To keep the first transformers port
    purely PyTorch / kernel-free, the dense path is identical to
    [`MiniMaxM3VLAttention`] and the index branch (whose projections still
    need to load) contributes via a residual sum on the attention output. The
    sparse-selection optimization is deferred to a follow-up.
    """

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        layer_idx: int,
        disable_index_value: bool = False,
    ):
        super().__init__(config, layer_idx)
        sparse_cfg = config.sparse_attention_config or {}
        self.disable_index_value = bool(disable_index_value)
        self.total_idx_heads = int(sparse_cfg.get("sparse_num_index_heads", 4))
        self.idx_head_dim = int(sparse_cfg.get("sparse_index_dim", 128))
        self.index_q_proj = nn.Linear(
            config.hidden_size, self.total_idx_heads * self.idx_head_dim, bias=False
        )
        self.index_k_proj = nn.Linear(config.hidden_size, self.idx_head_dim, bias=False)
        if self.disable_index_value:
            self.index_v_proj = None
            self.index_o_proj = None
        else:
            self.index_v_proj = nn.Linear(config.hidden_size, self.idx_head_dim, bias=False)
            self.index_o_proj = nn.Linear(
                self.total_idx_heads * self.idx_head_dim, config.hidden_size, bias=False
            )
        eps = config.rms_norm_eps
        self.index_q_norm = _build_norm(self.idx_head_dim, eps, self.use_gemma_norm)
        self.index_k_norm = _build_norm(self.idx_head_dim, eps, self.use_gemma_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.disable_index_value:
            return attn_output, attn_weights

        # Lightning-index branch: full dense math (no top-k block selection in
        # this first port). The selection optimization is a follow-up that
        # plugs into ``attention_interface`` once kernels are wired up.
        bsz, slen, _ = hidden_states.shape
        idx_q = self.index_q_proj(hidden_states).view(bsz, slen, self.total_idx_heads, self.idx_head_dim)
        idx_k = self.index_k_proj(hidden_states).view(bsz, slen, 1, self.idx_head_dim)
        idx_v = self.index_v_proj(hidden_states).view(bsz, slen, 1, self.idx_head_dim)
        idx_q = self.index_q_norm(idx_q)
        idx_k = self.index_k_norm(idx_k)
        cos, sin = position_embeddings
        if cos.shape[-1] >= self.idx_head_dim:
            idx_q_t = idx_q.transpose(1, 2)
            idx_k_t = idx_k.transpose(1, 2)
            idx_q_t, idx_k_t = apply_rotary_pos_emb(
                idx_q_t, idx_k_t, cos[..., : self.idx_head_dim], sin[..., : self.idx_head_dim]
            )
            idx_q = idx_q_t.transpose(1, 2)
            idx_k = idx_k_t.transpose(1, 2)

        # Broadcast the single idx-k/idx-v head to all idx-q heads, then SDPA.
        idx_q_h = idx_q.transpose(1, 2)  # [B, H_idx, S, D_idx]
        idx_k_h = idx_k.transpose(1, 2).expand(-1, self.total_idx_heads, -1, -1)
        idx_v_h = idx_v.transpose(1, 2).expand(-1, self.total_idx_heads, -1, -1)
        idx_attn = F.scaled_dot_product_attention(
            idx_q_h, idx_k_h, idx_v_h, attn_mask=None, is_causal=True
        )
        idx_attn = idx_attn.transpose(1, 2).reshape(bsz, slen, self.total_idx_heads * self.idx_head_dim)
        idx_output = self.index_o_proj(idx_attn)
        return attn_output + idx_output, attn_weights


# ---------------------------------------------------------------------------
# Text branch: decoder layer + model
# ---------------------------------------------------------------------------


def _is_moe_layer(config: MiniMaxM3VLTextConfig, layer_idx: int) -> bool:
    if config.moe_layer_freq is None:
        return True
    return bool(config.moe_layer_freq[layer_idx])


def _is_sparse_attn_layer(config: MiniMaxM3VLTextConfig, layer_idx: int) -> bool:
    sa = config.sparse_attention_config
    if sa is None:
        return False
    freq = sa.get("sparse_attention_freq")
    if freq is None:
        return False
    return bool(freq[layer_idx])


def _is_disable_index_value(config: MiniMaxM3VLTextConfig, layer_idx: int) -> bool:
    sa = config.sparse_attention_config
    if sa is None:
        return False
    flags = sa.get("sparse_disable_index_value")
    if flags is None:
        return False
    return bool(flags[layer_idx])


class MiniMaxM3VLDecoderLayer(GradientCheckpointingLayer):
    """M3 decoder layer: chooses dense/MoE MLP and dense/sparse attention per layer."""

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_gemma_norm = config.use_gemma_norm
        eps = config.rms_norm_eps

        if _is_sparse_attn_layer(config, layer_idx):
            self.self_attn = MiniMaxM3VLSparseAttention(
                config, layer_idx, disable_index_value=_is_disable_index_value(config, layer_idx)
            )
        else:
            self.self_attn = MiniMaxM3VLAttention(config, layer_idx)

        if _is_moe_layer(config, layer_idx):
            # We expose the MoE under ``mlp`` so the existing ``MoeModelOutputWithPast``
            # plumbing in [`MiniMaxM3VLTextModel`] inherited from Mixtral is reused.
            self.mlp = MiniMaxM3VLSparseMoeBlock(config)
        else:
            self.mlp = MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)

        self.input_layernorm = _build_norm(config.hidden_size, eps, self.use_gemma_norm)
        self.post_attention_layernorm = _build_norm(config.hidden_size, eps, self.use_gemma_norm)

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
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxM3VLPreTrainedModel(MiniMaxM2PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    input_modalities = ("image", "video", "text")

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
        elif isinstance(module, (MiniMaxM3VLRMSNorm, MiniMaxM3VLGemmaRMSNorm)):
            init.zeros_(module.weight) if isinstance(module, MiniMaxM3VLGemmaRMSNorm) else init.ones_(module.weight)


class MiniMaxM3VLTextModel(MiniMaxM2Model):
    """Stand-alone text backbone (no LM head). Used by [`MiniMaxM3VLModel`]."""

    config: MiniMaxM3VLTextConfig

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        # Recreate the layers with the M3 decoder
        self.layers = nn.ModuleList(
            [MiniMaxM3VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Replace top-level norm with Gemma-style if needed
        if config.use_gemma_norm:
            self.norm = MiniMaxM3VLGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxM3VLForCausalLM(MiniMaxM2ForCausalLM):
    """Text-only causal LM head, mirrors ``language_model.*`` weight names."""

    config: MiniMaxM3VLTextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMaxM3VLTextConfig):
        # Use the *text* backbone — the M2 parent would otherwise pick up the
        # composite ``MiniMaxM3VLModel`` from the modular substitution table,
        # which expects a vision_config.
        super().__init__(config)
        self.model = MiniMaxM3VLTextModel(config)
        self.post_init()


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class MiniMaxM3VLVisionEmbeddings(nn.Module):
    """Conv3d patch embedding over a flattened ``[N_patches, C * T * Pp * Pp]`` input.

    The image processor produces a 2D tensor where each row is one flat patch
    (channels × temporal × patch × patch). We reshape it to ``[N, C, T, P, P]``,
    apply Conv3d, then squeeze back to ``[N, hidden_size]``.
    """

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.num_channels = config.num_channels
        self.patch_embedding = nn.Conv3d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [N_patches, C * T * P * P]
        n = pixel_values.shape[0]
        pixel_values = pixel_values.view(
            n, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        out = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))
        return out.reshape(n, -1)


class MiniMaxM3VL3DRotaryEmbedding(nn.Module):
    """3D RoPE over (T, H, W) for the vision tower.

    Splits ``rotary_dim`` into 3 equal parts (one per axis). Returns
    ``(cos, sin)`` tensors broadcastable over heads.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        # Allocate dims across (t, h, w) — round each to even so half-rotation works.
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
        coords = torch.cat(coords_list, dim=0)  # [seq, 3]

        parts_cos: list[torch.Tensor] = []
        parts_sin: list[torch.Tensor] = []
        for axis, dim in enumerate(self.dims):
            if dim == 0:
                continue
            inv = self._axis_inv_freq(dim, device, dtype)  # [dim/2]
            freqs = coords[:, axis : axis + 1] * inv[None, :]  # [seq, dim/2]
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
            parts_cos.append(cos)
            parts_sin.append(sin)
        cos = torch.cat(parts_cos, dim=-1)
        sin = torch.cat(parts_sin, dim=-1)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_vision_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    cos = cos[None, :, None, :]  # [1, seq, 1, rot_dim]
    sin = sin[None, :, None, :]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class MiniMaxM3VLVisionAttention(nn.Module):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
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
        # hidden_states: [batch, seq, hidden]
        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        q, k = _apply_vision_rope(q, k, cos, sin)
        # [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.dropout)
        out = out.transpose(1, 2).reshape(bsz, seq, self.embed_dim)
        return self.out_proj(out)


class MiniMaxM3VLVisionMLP(nn.Module):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


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
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


@auto_docstring
class MiniMaxM3VLVisionModel(MiniMaxM3VLPreTrainedModel):
    """CLIP-like vision tower with Conv3d patch embed and 3D RoPE.

    Accepts the packed image representation produced by
    [`MiniMaxM3VLImageProcessor`]:
      - ``pixel_values``: ``[total_patches, C * T * P * P]``
      - ``image_grid_thw``: ``[num_images, 3]``
    """

    config: MiniMaxM3VLVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = MiniMaxM3VLVisionEmbeddings(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MiniMaxM3VL3DRotaryEmbedding(head_dim, theta=config.rope_theta)
        self.layers = nn.ModuleList(
            [MiniMaxM3VLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        embeds = self.embeddings(pixel_values)
        cos, sin = self.rotary_emb(image_grid_thw, device=embeds.device, dtype=torch.float32)
        # Batch dim of 1 -- the vision tower runs over flat packed patches
        hidden_states = embeds.unsqueeze(0)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        all_hidden_states = [] if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, position_embeddings=(cos, sin))
        hidden_states = self.post_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states[:, 0],
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


# ---------------------------------------------------------------------------
# Multimodal: patch merger + projector + full conditional generation model
# ---------------------------------------------------------------------------


class MiniMaxM3VLPatchMerger(nn.Module):
    """Spatial patch merger: reshape ``spatial_merge_size**2`` neighboring patches
    into the channel dim, then a 2-layer MLP back to ``text_hidden_size``.
    """

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__()
        text_hidden = config.text_config.hidden_size
        merge_size = config.vision_config.spatial_merge_size
        mid = config.projector_hidden_size if config.projector_hidden_size is not None else text_hidden
        self.spatial_merge_size = merge_size
        self.linear_1 = nn.Linear(
            text_hidden * (merge_size**2), mid, bias=config.multimodal_projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(mid, text_hidden, bias=config.multimodal_projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        x = image_features.reshape(image_features.shape[0] // (self.spatial_merge_size**2), -1)
        return self.linear_2(self.act(self.linear_1(x)))


class MiniMaxM3VLMultiModalProjector(nn.Module):
    """2-layer projector from vision hidden_size to text hidden_size."""

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__()
        vision_hidden = config.vision_config.hidden_size
        text_hidden = config.text_config.hidden_size
        mid = config.projector_hidden_size if config.projector_hidden_size is not None else text_hidden
        self.linear_1 = nn.Linear(vision_hidden, mid, bias=config.multimodal_projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(mid, text_hidden, bias=config.multimodal_projector_bias)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(image_features)))


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


@auto_docstring(
    custom_intro="MiniMax M3 VL backbone (vision tower + projector + text model), without LM head."
)
class MiniMaxM3VLModel(MiniMaxM3VLPreTrainedModel):
    config: MiniMaxM3VLConfig

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__(config)
        # Vision tower + projector + patch merger.
        self.vision_tower = MiniMaxM3VLVisionModel(config.vision_config)
        self.multi_modal_projector = MiniMaxM3VLMultiModalProjector(config)
        self.patch_merge_mlp = MiniMaxM3VLPatchMerger(config)
        # Text backbone.
        self.language_model = MiniMaxM3VLTextModel(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        vision_out = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        # Vision tower returns ``[1, seq, vision_hidden]``; squeeze the batch dim.
        # Project vision_hidden -> text_hidden, then merge spatial_merge_size**2 patches.
        hidden_states = vision_out.last_hidden_state.squeeze(0)
        hidden_states = self.multi_modal_projector(hidden_states)
        return self.patch_merge_mlp(hidden_states)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.Tensor:
        if input_ids is None:
            image_emb = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_emb).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index
        return special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values, image_grid_thw=image_grid_thw
            ).to(inputs_embeds.device, inputs_embeds.dtype)
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
class MiniMaxM3VLForConditionalGeneration(MiniMaxM3VLPreTrainedModel, GenerationMixin):
    config: MiniMaxM3VLConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__(config)
        self.model = MiniMaxM3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
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
