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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicLayer, StaticLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, torch_compilable_check
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.import_utils import is_torchdynamo_compiling
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig
from ..clip.modeling_clip import CLIPMLP, CLIPAttention, CLIPEncoderLayer
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from ..gemma3.modeling_gemma3 import Gemma3RMSNorm
from ..laguna.modeling_laguna import LagunaSparseMoeBlock
from ..llama.modeling_llama import eager_attention_forward
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaMultiModalProjector,
)
from ..minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
from ..minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2PreTrainedModel,
    MiniMaxM2RotaryEmbedding,
    MiniMaxM2TopKRouter,
    apply_rotary_pos_emb,
)
from ..mixtral.modeling_mixtral import MixtralDecoderLayer
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor, Qwen2VLProcessorKwargs


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
    index_n_heads (`int`, *optional*, defaults to 4):
        Number of heads in the lightning indexer's dot-product scoring branch.
    index_head_dim (`int`, *optional*, defaults to 128):
        Per-head channel dimension of the lightning indexer.
    index_block_size (`int`, *optional*, defaults to 128):
        Number of key tokens pooled into a single scored block.
    index_topk_blocks (`int`, *optional*, defaults to 16):
        Number of top-scoring key blocks each query may attend to.
    index_local_blocks (`int`, *optional*, defaults to 1):
        Number of key blocks immediately preceding the query always kept visible.
    num_mtp_modules (`int`, *optional*, defaults to 0):
        Number of multi-token-prediction modules in the checkpoint; ignored at inference.
    """

    model_type = "minimax_m3_vl_text"
    base_config_key = "text_config"
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_up_proj_scale_inv": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_scale_inv": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

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
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    layer_types: list[str] | None = None
    num_mtp_modules: int = 0
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        # Older checkpoints ship the lightning-indexer hyperparameters as a nested
        # ``sparse_attention_config`` dict; fold it into the flat ``index_*`` fields
        # (and derive ``layer_types`` from its per-layer frequency) before the strict
        # parent init runs, so model code only ever reads flat config attributes.
        sparse_cfg = kwargs.pop("sparse_attention_config", None) or {}
        PreTrainedConfig.__post_init__(self, **kwargs)

        for flat, legacy in {
            "index_n_heads": "sparse_num_index_heads",
            "index_head_dim": "sparse_index_dim",
            "index_block_size": "sparse_block_size",
            "index_topk_blocks": "sparse_topk_blocks",
            "index_local_blocks": "sparse_local_block",
        }.items():
            if legacy in sparse_cfg:
                setattr(self, flat, sparse_cfg[legacy])

        # ``layer_types`` is the canonical per-layer attention dispatch: it tells
        # ``DynamicCache(config=...)`` which layers want the sparse cache and tells
        # ``MiniMaxM3VLAttention`` which layers build a sparse Lightning Indexer.
        if self.layer_types is None and "sparse_attention_freq" in sparse_cfg:
            self.layer_types = [
                "minimax_m3_sparse" if f else "full_attention" for f in sparse_cfg["sparse_attention_freq"]
            ]
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


class MiniMaxM3VLSparseCacheLayer(DynamicLayer):
    """Cache layer for M3 sparse-attention layers: a standard ``DynamicLayer``
    for the main attention plus an ``idx_keys`` slot holding the lightning
    indexer's keys (one head, ``index_head_dim`` per token).

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


class MiniMaxM3VLSparseStaticCacheLayer(StaticLayer):
    """Static counterpart of [`MiniMaxM3VLSparseCacheLayer`] for ``torch.compile`` / ``StaticCache``:
    a standard ``StaticLayer`` for the main GQA attention plus a pre-allocated ``idx_keys`` buffer
    holding the lightning indexer's keys (one head, ``index_head_dim`` per token), written in place.

    It shares ``layer_type = "minimax_m3_sparse"`` with the dynamic layer but, being a ``StaticLayer``
    subclass, auto-registers in ``LAYER_TYPE_STATIC_CACHE_MAPPING`` instead, so ``StaticCache`` picks
    it (and ``DynamicCache`` keeps picking the dynamic one) for the sparse layers.

    The indexer keeps its own ``idx_cumulative_length`` rather than reusing the main attention's
    ``cumulative_length``: the indexer runs *before* the main attention writes its KV in each forward,
    so the two counters are bumped at different points and must be tracked independently.
    """

    layer_type = "minimax_m3_sparse"

    def __init__(self, max_cache_len: int):
        super().__init__(max_cache_len)
        self.idx_keys: torch.Tensor | None = None
        # Tensor (not int) so it can be marked as a static address for cudagraphs, like ``cumulative_length``.
        self.idx_cumulative_length = torch.tensor([0], dtype=int)

    def update_index(self, idx_k: torch.Tensor) -> torch.Tensor:
        """Write the new token's ``idx_k`` into the static buffer in place and return the whole buffer.

        The buffer's unfilled tail holds zeros, but those slots sit at key positions ahead of every
        current query, so the indexer's block- and token-level causal masking discards them — the
        returned ``[B, 1, max_cache_len, D]`` history is therefore safe to score against directly.
        """
        if self.idx_keys is None:
            self.idx_keys = torch.zeros(
                (idx_k.shape[0], idx_k.shape[1], self.max_cache_len, idx_k.shape[-1]),
                dtype=idx_k.dtype,
                device=idx_k.device,
            )
            self.idx_cumulative_length = self.idx_cumulative_length.to(idx_k.device)
            if not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.idx_keys)
                torch._dynamo.mark_static_address(self.idx_cumulative_length)

        kv_len = idx_k.shape[-2]
        cache_position = torch.arange(kv_len, device=self.idx_keys.device) + self.idx_cumulative_length
        self.idx_cumulative_length.add_(kv_len)
        try:
            self.idx_keys.index_copy_(2, cache_position, idx_k)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.idx_keys[:, :, cache_position] = idx_k
        return self.idx_keys

    def reset(self) -> None:
        super().reset()
        if self.idx_keys is not None:
            self.idx_keys.zero_()
        self.idx_cumulative_length.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.idx_keys is not None:
            self.idx_keys = self.idx_keys.index_select(0, beam_idx.to(self.idx_keys.device))


class MiniMaxM3VLRMSNorm(Gemma3RMSNorm):
    """Gemma-style RMSNorm: normalizes in fp32 and scales by `weight + 1`."""


class MiniMaxM3VLDenseMLP(nn.Module):
    def __init__(self, config: MiniMaxM3VLTextConfig, intermediate_size: int | None = None):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else config.dense_intermediate_size
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return self.down_proj((up + 1.0) * glu)


class MiniMaxM3VLExperts(DeepseekV4Experts):
    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.swiglu_alpha)
        return (up + 1.0) * glu


class MiniMaxM3VLTopKRouter(MiniMaxM2TopKRouter):
    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(config.num_local_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.to(self.weight.dtype), self.weight)
        # Sigmoid scoring (not softmax), as in M2.
        routing_weights = F.sigmoid(router_logits.float())
        scores_for_choice = routing_weights + self.e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        return router_logits, top_k_weights, top_k_index


class MiniMaxM3VLSparseMoeBlock(LagunaSparseMoeBlock):
    def __init__(self, config: MiniMaxM3VLTextConfig):
        nn.Module.__init__(self)
        self.gate = MiniMaxM3VLTopKRouter(config)
        self.experts = MiniMaxM3VLExperts(config)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.shared_experts = MiniMaxM3VLDenseMLP(
            config, intermediate_size=config.shared_intermediate_size * config.n_shared_experts
        )


class MiniMaxM3VLRotaryEmbedding(MiniMaxM2RotaryEmbedding):
    pass


class MiniMaxM3VLAttention(MiniMaxM2Attention):
    """
    M3 attention: per-head Gemma QK-norm + partial RoPE, optionally sparse indexer selection which require position IDs.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = (
            MiniMaxM3VLIndexer(config, layer_idx) if config.layer_types[layer_idx] == "minimax_m3_sparse" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.indexer is not None:
            attention_mask = self.indexer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
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
        return self.o_proj(attn_output), attn_weights


class MiniMaxM3VLIndexer(nn.Module):
    r"""Lightning Indexer for MiniMax M3 sparse attention.

    Scores each query against every key with a small `index_n_heads`-head
    dot-product branch, then max-pools those per-key scores into *blocks* of
    `index_block_size` keys and keeps, per query, the top-`index_topk_blocks`
    key blocks plus the `index_local_blocks` blocks immediately preceding the
    query (always visible). Selection therefore happens at the granularity of a
    *block of keys* rather than individual keys: the expensive main attention
    only has to attend the handful of selected key blocks, which is what makes
    it block-sparse (and cheaper) on long sequences. It returns a
    `[B, 1, S_q, S_k]` attention mask that is `0` at every allowed
    (query, key) pair and `-inf` elsewhere.

    Like DeepSeek-V4's indexer this is purely a *selection* branch: it has no
    value projection and produces no residual output of its own (the upstream
    checkpoint disables the index-value path on every sparse layer).

    TODO: blocks are anchored to absolute key *slots* (the contiguous reshape in
    `forward` and `q_block = slot // block_size`), so left-padding shifts the block
    boundaries and the selection diverges from an unpadded run -- only right-padding
    is equivalent (same limitation as DeepSeek-V4; see `test_right_padding_does_not_leak`
    / the skipped `test_left_padding_compatibility`). For *true* left-padding equivalence
    we'd make blocking content-relative instead of slot-relative:
      1. derive block ids from `position_ids` (content positions, 0 at each row's first
         real token) rather than from absolute slots, and
      2. replace the contiguous `view(..., num_key_blocks, block_size).amax(-1)` key pool
         with a per-row position-binned pool (e.g. `scatter_reduce` over `key_position //
         block_size`), so pad never shifts the boundaries, and
      3. mask padded keys' scores to `-inf` before the pool so a pad key can't win a block
         a top-k slot.
    """

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.block_size = config.index_block_size
        self.topk_blocks = config.index_topk_blocks
        self.local_blocks = config.index_local_blocks
        self.q_proj = nn.Linear(config.hidden_size, config.index_n_heads * config.index_head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.q_norm = MiniMaxM3VLRMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxM3VLRMSNorm(config.index_head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch, q_len, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(batch, q_len, self.num_heads, self.head_dim)
        idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)  # [B, H_idx, Sq, D]
        idx_k = self.k_norm(idx_k).transpose(1, 2)  # [B, 1, Sq, D]
        cos, sin = position_embeddings
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

        if past_key_values is not None:
            idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)

        k_len = idx_k.shape[2]
        q_positions = torch.arange(k_len - q_len, k_len, device=idx_q.device)
        num_key_blocks = -(-k_len // self.block_size)  # ceil-div
        pad = num_key_blocks * self.block_size - k_len

        # we compute a single score per block of keys
        scores = torch.matmul(idx_q.float(), idx_k.expand(-1, self.num_heads, -1, -1).float().transpose(-1, -2))
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        scores = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)
        block_scores = scores.amax(dim=-1).amax(dim=1)  # -> [B, S_q, num_key_blocks]

        # Block-level causality on slot indices (so top-k never spends a slot on a future block).
        q_block = q_positions // self.block_size  # [S_q]
        future = torch.arange(num_key_blocks, device=idx_q.device).view(1, 1, -1) > q_block.view(1, -1, 1)
        block_scores = block_scores.masked_fill(future, float("-inf"))

        bias = block_scores.new_full((batch, q_len, num_key_blocks), float("-inf"))
        bias.scatter_(-1, block_scores.topk(min(self.topk_blocks, num_key_blocks), dim=-1).indices, 0.0)
        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=idx_q.device)
            local = (q_block.view(-1, 1) - local.view(1, -1)).clamp(min=0)  # [S_q, local]
            bias.scatter_(-1, local.unsqueeze(0).expand(batch, -1, -1), 0.0)
        bias = bias.masked_fill(future, float("-inf"))

        # Broadcast the per-block keep/drop verdict back onto every key (block granularity), add head axis.
        block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :k_len].unsqueeze(1)

        # even if a block is selected, we need to make sure that inside the mask is causal.
        min_dtype = torch.finfo(idx_q.dtype).min
        if attention_mask is not None:
            return attention_mask.masked_fill(~block_keep, min_dtype)

        # if no attention mask is provided we build it
        k_positions = torch.arange(k_len, device=idx_q.device)
        token_future = k_positions.view(1, 1, 1, -1) > q_positions.view(1, 1, -1, 1)
        keep = block_keep & ~token_future
        return keep.new_zeros(keep.shape, dtype=idx_q.dtype).masked_fill(~keep, min_dtype)


class MiniMaxM3VLDecoderLayer(MixtralDecoderLayer):
    """M3 decoder layer: per-layer dense/MoE MLP and dense/sparse attention."""

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiniMaxM3VLAttention(config, layer_idx)
        self.mlp = (
            MiniMaxM3VLSparseMoeBlock(config)
            if config.moe_layer_freq[layer_idx]
            else MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)
        )
        self.input_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxM3VLPreTrainedModel(MiniMaxM2PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    input_modalities = ("image", "video", "text")
    # MTP modules ship in the upstream checkpoint but aren't part of this port.
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, MiniMaxM3VLExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, MiniMaxM3VLTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
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


class MiniMaxM3VLVisionEmbeddings(Qwen2_5_VisionPatchEmbed):
    """Conv3d patch embedding over a flat ``[N_patches, C * T * P * P]`` input.

    Identical to [`Qwen2_5_VisionPatchEmbed`]; only the constructor differs (it
    reads the dims from the vision config). The upstream checkpoint stores the
    conv as ``patch_embedding``, renamed to the inherited ``proj`` in the
    conversion mapping.
    """

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        nn.Module.__init__(self)
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.num_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )


class MiniMaxM3VL3DRotaryEmbedding(nn.Module):
    r"""3D RoPE for the vision tower: each patch is rotated by its ``(T, H, W)`` grid position.

    ``head_dim`` is split into three contiguous bands, one per axis. A band only sees
    its own coordinate, so the patch embedding carries independent temporal, row and
    column phases::

        |<------------------------- head_dim ------------------------>|
        +--------------------+--------------------+--------------------+
        |     T  (frames)    |      H  (rows)     |      W  (cols)     |
        |        band        |        band        |  head_dim - 2*band |
        +--------------------+--------------------+--------------------+

    ``band = head_dim // 3`` rounded down to an even number, so each band holds
    ``band // 2`` frequencies. ``inv_freq`` lays those frequencies out as ``T|H|W``;
    ``axis_of_freq`` tags each one with the grid axis (``0=T, 1=H, 2=W``) whose
    coordinate rotates it. ``forward`` scales every frequency by its axis' coordinate,
    then ``repeat_interleave(2)`` duplicates each side by side into the interleaved
    ``[f0, f0, f1, f1, ...]`` layout that pairs with ``rotate_half_llm``, returning
    ``(cos, sin)`` of shape ``[num_patches, head_dim]``.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        band = (head_dim // 3) // 2 * 2
        self.dims = (band, band, head_dim - 2 * band)
        self.theta = theta

    def forward(
        self, grid_thw: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = []
        for t, h, w in grid_thw.tolist():
            ti = torch.arange(t).repeat_interleave(h * w)
            hi = torch.arange(h).repeat_interleave(w).repeat(t)
            wi = torch.arange(w).repeat(t * h)
            coords.append(torch.stack([ti, hi, wi], dim=-1))
        coords = torch.cat(coords).to(device=device, dtype=torch.float32)

        # Per-axis inverse frequencies laid out as T|H|W, then broadcast each patch's
        # T/H/W coordinate across its own band so every frequency rotates by its axis.
        inv_freq = torch.cat(
            [1.0 / (self.theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)) for d in self.dims]
        )
        band_freqs = torch.tensor([d // 2 for d in self.dims], device=device)
        freqs = coords.repeat_interleave(band_freqs, dim=1) * inv_freq
        emb = freqs.repeat_interleave(2, dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input (interleaved layout)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    q_embed = q * cos + rotate_half_llm(q) * sin
    k_embed = k * cos + rotate_half_llm(k) * sin
    return q_embed, k_embed


class MiniMaxM3VLVisionAttention(CLIPAttention):
    """CLIP-style vision attention; the only difference from [`CLIPAttention`] is
    that queries and keys are rotated by the tower's 3D RoPE before the
    (interface-dispatched) scaled dot-product attention."""

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        # The vision tower has no grouped-query attention; the shared eager kernel
        # still expects this attribute to drive its (no-op) ``repeat_kv``.
        self.num_key_value_groups = 1

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
        cos, sin = position_embeddings
        queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)
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


# 3D-RoPE `position_embeddings` pass via `**kwargs` for simplicity
class MiniMaxM3VLVisionEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        self.self_attn = MiniMaxM3VLVisionAttention(config)
        self.mlp = MiniMaxM3VLVisionMLP(config)


class MiniMaxM3VLVisionTransformer(nn.Module):
    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__()
        self.embeddings = MiniMaxM3VLVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([MiniMaxM3VLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MiniMaxM3VL3DRotaryEmbedding(head_dim, theta=config.rope_theta)

    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        embeds = self.embeddings(pixel_values)
        cos, sin = self.rotary_emb(image_grid_thw, device=embeds.device, dtype=embeds.dtype)
        hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None, position_embeddings=(cos, sin), **kwargs)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states[:, 0])


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
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image.
        """
        return self.vision_model(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)


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


class MiniMaxM3VLMultiModalProjector(LlavaMultiModalProjector):
    """2-layer GELU projector from vision hidden_size to text hidden_size.

    Same forward as [`LlavaMultiModalProjector`]; only the constructor differs —
    M3 projects through ``projector_hidden_size`` rather than reusing
    ``text_config.hidden_size`` for the inner dimension.
    """

    def __init__(self, config: MiniMaxM3VLConfig):
        nn.Module.__init__(self)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.projector_hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size, bias=True)


class MiniMaxM3VLModelOutputWithPast(LlavaModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_patches, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_patches, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: torch.FloatTensor | None = None


class MiniMaxM3VLCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_patches, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_patches, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: torch.FloatTensor | None = None


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

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains video last hidden states from the vision tower and apply multimodal projection."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor`):
            The tensors corresponding to the input video frames.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of each video's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        # Video frames flow through the same vision pipeline as images (the tower is
        # grid-agnostic); only the placeholder token they scatter into differs.
        vision_outputs = self.vision_tower(pixel_values=pixel_values_videos, image_grid_thw=video_grid_thw, **kwargs)
        hidden_states = self.multi_modal_projector(vision_outputs.last_hidden_state.squeeze(0))
        vision_outputs.pooler_output = self.patch_merge(hidden_states)
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains the image/video placeholder masks from `input_ids` or `inputs_embeds`, and checks that the
        placeholder token count matches the multimodal feature length. Raises if they differ.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
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
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of each video's feature grid, used to build the vision 3D RoPE
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

        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw
            ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

        image_mask, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds, image_features=image_features, video_features=video_features
        )
        if image_features is not None:
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if video_features is not None:
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

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
            video_hidden_states=video_features,
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

    def get_video_features(self, pixel_values_videos, video_grid_thw, **kwargs):
        r"""
        pixel_values_videos (`torch.FloatTensor`):
            The tensors corresponding to the input video frames.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of each video's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
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
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of each video's feature grid, used to build the vision 3D RoPE
            and to merge patch features.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
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
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- pixel inputs are merged into the cache on the first step, so we
        # only forward them once (image and video alike).
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        or_mask_function: Callable | None = None,
        and_mask_function: Callable | None = None,
        block_sequence_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        # ``generate`` calls this to pre-build the per-layer-type mask dict when running under a
        # compilable (static) cache. M3's "minimax_m3_sparse" layer type is model-specific, so it is
        # absent from the global `LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING`; rather than mutate that
        # global from here (a module-level write the modular converter would prune), we override the
        # hook locally. All M3 layers share a plain causal base mask (the block-sparse selection is
        # the indexer's separate additive bias), so every layer type maps to the same causal mask.
        text_config = config.get_text_config()
        causal_mask = create_causal_mask(
            config=text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
            block_sequence_ids=block_sequence_ids,
        )
        return dict.fromkeys(set(text_config.layer_types), causal_mask)


class MiniMaxM3VLProcessorKwargs(Qwen2VLProcessorKwargs):
    _defaults = {
        "videos_kwargs": {"do_resize": False, "return_metadata": True},
    }


class MiniMaxM3VLProcessor(Qwen2VLProcessor):
    """Combines tokenizer + image_processor + video_processor for MiniMax M3 VL.

    Expands ``IMAGE_TOKEN`` / ``VIDEO_TOKEN`` markers in the prompt into the matching
    number of placeholder tokens (one per merged patch), wrapped in ``VISION_START_TOKEN``
    / ``VISION_END_TOKEN`` brackets. Video chunks are additionally prefixed with a
    ``]<]{seconds} seconds[>[`` timestamp marker per frame when metadata is available.
    """

    valid_processor_kwargs = MiniMaxM3VLProcessorKwargs

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = self.IMAGE_TOKEN
        self.video_token = self.VIDEO_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) if tokenizer else None
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN) if tokenizer else None
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.VISION_START_TOKEN) if tokenizer else None
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.VISION_END_TOKEN) if tokenizer else None

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = int(image_inputs["image_grid_thw"][image_idx].prod() // merge_length)
        return self.VISION_START_TOKEN + self.IMAGE_TOKEN * num_image_tokens + self.VISION_END_TOKEN

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        merge_length = self.video_processor.merge_size**2
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        grid_t = int(grid_thw[0])
        frame_seqlen = int(grid_thw[1:].prod() // merge_length)
        metadata = video_inputs.get("video_metadata", [None] * (video_idx + 1))[video_idx]
        temporal_patch_size = self.video_processor.temporal_patch_size
        chunk = ""
        for frame in range(grid_t):
            if (
                metadata is not None
                and getattr(metadata, "fps", None) is not None
                and getattr(metadata, "frames_indices", None) is not None
            ):
                ts = (
                    metadata.frames_indices[min(frame * temporal_patch_size, len(metadata.frames_indices) - 1)]
                    / metadata.fps
                )
                chunk += f"]<]{ts:.1f} seconds[>["
            chunk += self.VISION_START_TOKEN + self.VIDEO_TOKEN * frame_seqlen + self.VISION_END_TOKEN
        return chunk


__all__ = [
    "MiniMaxM3VLConfig",
    "MiniMaxM3VLTextConfig",
    "MiniMaxM3VLVisionConfig",
    "MiniMaxM3VLForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLPreTrainedModel",
    "MiniMaxM3VLProcessor",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLVisionModel",
]
