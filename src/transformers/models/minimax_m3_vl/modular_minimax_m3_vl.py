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
from ...cache_utils import Cache, DynamicCache, DynamicLayer, StaticLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPooling, MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging, torch_compilable_check
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


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLTextConfig(MiniMaxM2Config):
    r"""
    dense_intermediate_size (`int`, *optional*, defaults to 12288):
        Intermediate size of the dense MLP used on layers whose `mlp_layer_types` entry is `"dense"`.
    shared_intermediate_size (`int`, *optional*, defaults to 3072):
        Intermediate size of a single shared expert in the MoE layers.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of head channels rotated by RoPE; the remaining channels are passed through unchanged.
    swiglu_alpha (`float`, *optional*, defaults to 1.702):
        Sigmoid gain of the SwiGLU-OAI activation.
    swiglu_limit (`float`, *optional*, defaults to 7.0):
        Clamp bound applied to the gate and up projections of the SwiGLU-OAI activation.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP selector: `"sparse"` for a MoE block, `"dense"` for a dense MLP.
    index_n_heads (`int`, *optional*, defaults to 4):
        Number of heads in the lightning indexer's dot-product scoring branch.
    index_head_dim (`int`, *optional*, defaults to 128):
        Per-head channel dimension of the lightning indexer.
    index_block_size (`int`, *optional*, defaults to 128):
        Number of key tokens pooled into a single scored block.
    index_topk_blocks (`int`, *optional*, defaults to 16):
        Number of top-scoring key blocks each query may attend to.
    index_local_blocks (`int`, *optional*, defaults to 1):
        Number of key blocks immediately preceding the query always kept visible / attended to.
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
    routed_scaling_factor: float = 2.0
    rotary_dim: int = 64
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    mlp_layer_types: list[str] | None = None
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    layer_types: list[str] | None = None
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        sparse_cfg = kwargs.pop("sparse_attention_config", None) or {}
        moe_layer_freq = kwargs.pop("moe_layer_freq", None)
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

        # `layer_types` is the canonical per-layer attention dispatch: it tells
        # `DynamicCache(config=...)` which layers want the sparse cache and tells
        # `MiniMaxM3VLAttention` which layers build a sparse Lightning Indexer.
        if self.layer_types is None and "sparse_attention_freq" in sparse_cfg:
            self.layer_types = [
                "minimax_m3_sparse" if f else "full_attention" for f in sparse_cfg["sparse_attention_freq"]
            ]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        # `mlp_layer_types` is the per-layer MLP dispatch read by `MiniMaxM3VLDecoderLayer`:
        if self.mlp_layer_types is None and moe_layer_freq is not None:
            self.mlp_layer_types = ["sparse" if f else "dense" for f in moe_layer_freq]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-M3-preview")
@strict
class MiniMaxM3VLVisionConfig(PreTrainedConfig):
    r"""
    rope_parameters (`RopeParameters`, *optional*):
        Standard RoPE configuration for the vision tower's 3D rotary position embedding.
    """

    model_type = "minimax_m3_vl_vision"
    base_config_key = "vision_config"
    default_theta = 10000.0

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
    rope_parameters: RopeParameters | dict | None = None
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

        # Channel dim after grouping `spatial_merge_size**2` projected patches, consumed by the
        # patch-merge MLP inside `MiniMaxM3VLMultiModalProjector`.
        self.merged_hidden_size = self.text_config.hidden_size * (self.vision_config.spatial_merge_size**2)

        super().__post_init__(**kwargs)


class MiniMaxM3VLSparseCacheLayer(DynamicLayer):
    layer_type = "minimax_m3_sparse"

    def __init__(self, config: PreTrainedConfig | None = None):
        super().__init__(config)
        self.idx_keys: torch.Tensor | None = None

    def update_index(self, idx_k: torch.Tensor) -> torch.Tensor:
        """Append the new token's `idx_k` to the cache and return the full history."""
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
    layer_type = "minimax_m3_sparse"

    def __init__(self, max_cache_len: int):
        super().__init__(max_cache_len)
        self.idx_keys: torch.Tensor | None = None
        # Tensor (not int) so it can be marked as a static address for cudagraphs, like `cumulative_length`.
        self.idx_cumulative_length = torch.tensor([0], dtype=int)

    def update_index(self, idx_k: torch.Tensor) -> torch.Tensor:
        """Write the new token's `idx_k` into the static buffer in place and return the whole buffer.

        The buffer's unfilled tail holds zeros, but those slots sit at key positions ahead of every
        current query, so the indexer's block- and token-level causal masking discards them — the
        returned `[B, 1, max_cache_len, D]` history is therefore safe to score against directly.
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
        del self.act_fn

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # same as GPT OSS, but the weights are not interleaved
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
        self.shared_experts = MiniMaxM3VLDenseMLP(config, intermediate_size=config.shared_intermediate_size)


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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

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
        block_indices = None
        if self.indexer is not None:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(
                    key_states.shape[2] - query_states.shape[2], key_states.shape[2], device=query_states.device
                )
            position_ids = (position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)).expand(
                query_states.shape[0], -1
            )
            block_indices = self.indexer(hidden_states, position_embeddings, past_key_values, position_ids)
            if self.config._attn_implementation in ("eager", "sdpa"):
                attention_mask = self.indexer.build_block_mask(
                    block_indices,
                    attention_mask,
                    key_states.shape[2],
                    query_states.dtype,
                    query_states.device,
                    position_ids,
                )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            block_indices=block_indices,
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
    it block-sparse (and cheaper) on long sequences.

    The `index_local_blocks` boosting their score so they always win key slots, the
    same way the deployment block-sparse kernel (MiniMax `topk_sparse`) does it.

    `forward` returns the per-query selected key-block indices
    `[B, S_q, index_topk_blocks]`. Valid indices are left-packed and `-1`
    right-pads the unused slots (future/empty blocks), and the local boost makes
    selections deduplicated -- the exact contract the block-sparse attention
    kernel consumes (it counts the valid entries, then reads them sequentially
    and would double-count a repeated block). The eager/SDPA path instead calls
    `build_block_mask`, which expands the indices into the dense
    `[B, 1, S_q, S_k]` additive mask the standard attention interface expects
    (`0` at every allowed (query, key) pair, `-inf` elsewhere).

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
        past_key_values: Cache | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, q_len, _ = hidden_states.shape
        idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
        idx_q = self.q_norm(idx_q).transpose(1, 2)  # [B, H_idx, Sq, D]
        idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
        idx_k = self.k_norm(idx_k).transpose(1, 2)  # [B, 1, Sq, D]
        cos, sin = position_embeddings
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., : self.head_dim], sin[..., : self.head_dim])

        if past_key_values is not None:
            idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)

        k_len = idx_k.shape[2]
        num_key_blocks = -(-k_len // self.block_size)  # ceil-div
        pad = num_key_blocks * self.block_size - k_len

        scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))
        k_positions = torch.arange(k_len, device=idx_q.device)
        token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]  # [B, 1, S_q, S_k]
        scores = scores.masked_fill(token_future, float("-inf"))
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        scores = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)
        block_scores = scores.amax(dim=-1).amax(dim=1)  # -> [B, S_q, num_key_blocks]

        q_block = position_ids // self.block_size  # [B, S_q]

        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=idx_q.device)
            local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)  # [B, S_q, local]
            block_scores.scatter_(-1, local_idx, float("inf"))

        # Slots that fall on a future/empty block keep their `-inf`
        # score, which top-k sorts to the end, so tagging them `-1` yields left-packed block indices
        # with `-1` right-padding which is the format expect by block-sparse attention kernel.
        topk = min(self.topk_blocks, num_key_blocks)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)  # [B, S_q, topk]
        return topk_indices.masked_fill(topk_scores == float("-inf"), -1)

    def build_block_mask(
        self,
        block_indices: torch.Tensor,
        attention_mask: torch.Tensor | None,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        We build the full 4D attention mask (Batch, query, key, head)
        """
        batch, q_len, _ = block_indices.shape
        num_key_blocks = -(-key_length // self.block_size)

        # Scatter the kept blocks to `0`; `-1` slots land in a throwaway column we drop afterwards.
        safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
        bias = block_indices.new_full((batch, q_len, num_key_blocks + 1), float("-inf"), dtype=dtype)
        bias.scatter_(-1, safe, 0.0)
        bias = bias[..., :num_key_blocks]

        # Broadcast the per-block keep/drop verdict back onto every key (block granularity), add head axis.
        block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[..., :key_length].unsqueeze(1)

        # Compose block-selection with the existing mask, then emit a single additive float mask.
        if attention_mask is not None:
            padding_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            keep = block_keep & padding_mask
        else:
            k_positions = torch.arange(key_length, device=device)
            token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]  # [B, 1, S_q, S_k]
            keep = block_keep & ~token_future
        min_dtype = torch.finfo(dtype).min
        return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, min_dtype)


class MiniMaxM3VLDecoderLayer(MixtralDecoderLayer):
    """M3 decoder layer: per-layer dense/MoE MLP and dense/sparse attention."""

    def __init__(self, config: MiniMaxM3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiniMaxM3VLAttention(config, layer_idx)
        self.mlp = (
            MiniMaxM3VLSparseMoeBlock(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)
        )
        self.input_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxM3VLPreTrainedModel(MiniMaxM2PreTrainedModel):
    config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniMaxM3VLDecoderLayer", "MiniMaxM3VLVisionEncoderLayer"]
    input_modalities = ("image", "video", "text")
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _compatible_flash_implementations = ["kernels-staging/msa@v0"]

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
    config: MiniMaxM3VLTextConfig

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MiniMaxM3VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # `position_ids` is threaded to every layer so the sparse layers' lightning indexer can anchor
        # block selection to each query's content position (see `MiniMaxM3VLIndexer`).
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MiniMaxM3VLForCausalLM(MiniMaxM2ForCausalLM):
    config: MiniMaxM3VLTextConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMaxM3VLTextConfig):
        super().__init__(config)
        self.model = MiniMaxM3VLTextModel(config)
        self.post_init()


class MiniMaxM3VLVisionEmbeddings(Qwen2_5_VisionPatchEmbed):
    """Patch embedding, identical to [`Qwen2_5_VisionPatchEmbed`] (reads its dims from the vision
    config). The upstream checkpoint stores the conv as `patch_embedding`, renamed to the
    inherited `proj` in the conversion mapping."""

    def __init__(self, config) -> None:
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
    r"""3D RoPE for the vision tower: each patch is rotated by its `(T, H, W)` grid position.

    `2 * (head_dim // 2)` rotary dims are split evenly across the three axes (each rounded
    down to a multiple of 2), giving `axis_dim` dims per axis and `axis_dim // 2` frequencies::

        |<------------------ rotated (3 * axis_dim) ------------------>|<- pass ->|
        +--------------------+--------------------+--------------------+----------+
        |     T  (frames)    |      H  (rows)     |      W  (cols)     |          |
        |      axis_dim      |      axis_dim      |      axis_dim      |          |
        +--------------------+--------------------+--------------------+----------+

    Each axis' coordinate scales its own band of frequencies; the bands are concatenated as
    `T|H|W` and duplicated via `cat([f, f])` to pair with the half-rotation in
    `apply_rotary_pos_emb_vision`. Any head dims past `3 * axis_dim` are left unrotated.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0, spatial_merge_size: int = 1):
        super().__init__()
        # `2 * (head_dim // 2)` rotary dims are split evenly across T/H/W, each axis rounded
        # down to a multiple of 2. With head_dim=80 that is 26 dims/axis (39 freqs total); the
        # remaining `head_dim - 3 * axis_dim` dims are never rotated (they pass through).
        rope_dims = 2 * (head_dim // 2)
        self.axis_dim = 2 * ((rope_dims // 3) // 2)
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta

    def forward(
        self, grid_thw: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.spatial_merge_size
        coords = []
        for t, h, w in grid_thw.tolist():
            hi = torch.arange(h).unsqueeze(1).expand(-1, w)
            hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            wi = torch.arange(w).unsqueeze(0).expand(h, -1)
            wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            ti = torch.arange(t).repeat_interleave(h * w)
            coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
        coords = torch.cat(coords).to(device=device, dtype=torch.float32)

        # meta device init was having trouble when it was registered. TODO standardize?
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32, device=device) / self.axis_dim)
        )
        freqs = torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Only the first `rot_dim` head dims carry 3D RoPE; the tail passes through untouched.
    rot_dim = cos.shape[-1]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class MiniMaxM3VLVisionAttention(CLIPAttention):
    """CLIP-style vision attention; the only difference from [`CLIPAttention`] is
    that queries and keys are rotated by the tower's 3D RoPE before the
    (interface-dispatched) scaled dot-product attention."""

    def __init__(self, config: MiniMaxM3VLVisionConfig):
        super().__init__(config)
        # The vision tower has no grouped-query attention; the shared eager kernel
        # still expects this attribute to drive its (no-op) `repeat_kv`.
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
        self.embeddings = MiniMaxM3VLVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([MiniMaxM3VLVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MiniMaxM3VL3DRotaryEmbedding(
            head_dim, theta=config.rope_parameters["rope_theta"], spatial_merge_size=config.spatial_merge_size
        )
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image.
        """
        embeds = self.embeddings(pixel_values).to(self.pre_layrnorm.weight.dtype)
        cos, sin = self.rotary_emb(image_grid_thw, device=embeds.device, dtype=embeds.dtype)
        hidden_states = self.pre_layrnorm(embeds).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None, position_embeddings=(cos, sin), **kwargs)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states[:, 0])


class MiniMaxM3VLMultiModalProjector(nn.Module):
    """Projects each vision patch from `vision_config.hidden_size` to `text_config.hidden_size`
    (GELU MLP), then groups `spatial_merge_size**2` neighbouring patches into the channel dim and
    fuses them back to a single `text_config.hidden_size` token with a second GELU MLP."""

    def __init__(self, config: MiniMaxM3VLConfig):
        super().__init__()
        text_hidden = config.text_config.hidden_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.projector_hidden_size, bias=True)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.projector_hidden_size, text_hidden, bias=True)
        self.merge_linear_1 = nn.Linear(config.merged_hidden_size, config.projector_hidden_size, bias=True)
        self.merge_act = ACT2FN["gelu"]
        self.merge_linear_2 = nn.Linear(config.projector_hidden_size, text_hidden, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_2(self.act(self.linear_1(image_features)))
        hidden_states = hidden_states.reshape(hidden_states.shape[0] // (self.spatial_merge_size**2), -1)
        return self.merge_linear_2(self.merge_act(self.merge_linear_1(hidden_states)))


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
        # ready to scatter into the text embeddings — in `pooler_output`.
        vision_outputs = self.vision_tower(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        vision_outputs.pooler_output = self.multi_modal_projector(vision_outputs.last_hidden_state.squeeze(0))
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
        vision_outputs.pooler_output = self.multi_modal_projector(vision_outputs.last_hidden_state.squeeze(0))
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
class MiniMaxM3SparseForConditionalGeneration(LlavaForConditionalGeneration):
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


class MiniMaxM3VLProcessorKwargs(Qwen2VLProcessorKwargs):
    _defaults = {
        "videos_kwargs": {"do_resize": False, "return_metadata": True},
    }


class MiniMaxM3VLProcessor(Qwen2VLProcessor):
    """Combines tokenizer + image_processor + video_processor for MiniMax M3 VL.

    Expands `IMAGE_TOKEN` / `VIDEO_TOKEN` markers in the prompt into the matching
    number of placeholder tokens (one per merged patch), wrapped in `VISION_START_TOKEN`
    / `VISION_END_TOKEN` brackets. Video chunks are additionally prefixed with a
    `]<]{seconds} seconds[>[` timestamp marker per frame when metadata is available.
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
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLPreTrainedModel",
    "MiniMaxM3VLProcessor",
    "MiniMaxM3VLTextModel",
    "MiniMaxM3VLVisionModel",
]
