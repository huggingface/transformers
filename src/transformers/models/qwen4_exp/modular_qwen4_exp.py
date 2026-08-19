# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen4-Exp model."""

import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionRotaryEmbedding,
    causal_conv1d_fn,
    causal_conv1d_update,
    rotate_half,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from ..qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeModelOutputWithPast,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextExperts,
    Qwen3NextRMSNormGated,
    Qwen3NextSparseMoeBlock,
    Qwen3NextTopKRouter,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen4-Exp")
@strict
class Qwen4ExpTextConfig(Qwen3_5MoeTextConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers.
    partial_rotary_factor (`float`, *optional*, defaults to 0.25):
        Fraction of head_dim that gets RoPE.
    hc_count (`int`, *optional*, defaults to 4):
        Number of residual streams used by the hyper-connections.
    hc_lowrank (`int`, *optional*, defaults to 320):
        Rank of the learned hyper-connection input mixer.
    ple_layer_ids (`list[int]`, *optional*):
        One-indexed decoder layer ids that use Per-Layer Embedding (PLE).
    ple_embed_dim (`int`, *optional*):
        Total dimension of the embeddings concatenated from all n-gram heads in each PLE module. Defaults to
        `hidden_size`.
    ple_conv_kernel_size (`int`, *optional*, defaults to 4):
        Kernel size of the dilated depthwise convolution in each PLE module.
    ngram_size (`int`, *optional*, defaults to 3):
        Largest token n-gram represented by PLE.
    heads_per_ngram (`int`, *optional*, defaults to 8):
        Number of independently hashed embedding heads for every n-gram order.
    ngram_vocab_size_base (`int`, *optional*, defaults to 20000000):
        Lower bound used to derive a distinct prime vocabulary size for each hashed n-gram head.
    make_ngram_vocab_size_divisible_by (`int`, *optional*, defaults to 128):
        Divisor used to pad the combined n-gram embedding vocabulary.
    seed (`int`, *optional*, defaults to 1234):
        Seed used to deterministically derive the per-layer n-gram hash multipliers.
    split_ngram_parts (`int`, *optional*, defaults to 512):
        Number of checkpoint shards used for each PLE n-gram embedding table. Loading concatenates the shards into a
        single runtime embedding, while `save_pretrained` restores the configured sharded layout.
    indexer_n_heads (`int`, *optional*):
        Number of query heads used by the QSA token indexer. Setting this enables QSA on full-attention layers.
    indexer_kv_heads (`int`, *optional*):
        Number of indexer key heads. Qwen4-Exp QSA requires one key head.
    indexer_head_dim (`int`, *optional*):
        Dimension of every QSA indexer query and key head.
    indexer_budget (`int`, *optional*):
        Maximum number of tokens selected from complete compressed blocks for each query.
    indexer_compress_ratio (`int`, *optional*):
        Number of consecutive token keys averaged into one QSA index block.
    output_gate_type (`str`, *optional*):
        Activation used by the output gate of linear attention. If unset, `hidden_act` is used.
    """

    model_type = "qwen4_exp_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
        "layers.*.self_attn.indexer.index_qk_proj": "colwise_gather_output",
        "layers.*.attn_hyper_connection.input_mix_weight_down": "rowwise_split_input",
        "layers.*.mlp_hyper_connection.input_mix_weight_down": "rowwise_split_input",
        "hyper_connection_mixer.input_mix_weight_down": "rowwise_split_input",
        "layers.*.ple.ple_embedding.ngram_embedding": "embedding_rowwise",
    }
    base_model_fsdp_plan = {
        "embed_tokens": "free_full_weight",
        "layers.*": "free_full_weight",
        "hyper_connection_mixer": "keep_full_weight",
    }
    base_model_pp_plan = None

    partial_rotary_factor: float = 0.25
    hc_count: int = 4
    hc_lowrank: int = 320
    ple_layer_ids: list[int] | None = None
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    output_gate_type: str | None = None

    def __post_init__(self, **kwargs):
        self.ple_layer_ids = [] if self.ple_layer_ids is None else sorted(set(self.ple_layer_ids))
        self.ple_embed_dim = self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim

        # Qwen4-Exp keeps the GatedDeltaNet convolution, PLE convolution and n-gram context in separate cache states.
        # Without PLE, only the GatedDeltaNet state is needed.
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        # Full-attention layers use an indexed cache when QSA is enabled. If PLE is also attached to that layer, the hybrid
        # indexed cache additionally carries its convolution and n-gram context states
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            layer_types = []
            for i in range(self.num_hidden_layers):
                if bool((i + 1) % interval_pattern):
                    layer_types.append("linear_attention")
                elif i + 1 in self.ple_layer_ids:
                    layer_types.append("hybrid_indexed")
                else:
                    layer_types.append("deepseek_sparse_attention")
            self.layer_types = layer_types

        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates Qwen4-Exp architecture invariants."""
        unsupported_layer_types = sorted(
            set(self.layer_types) - {"linear_attention", "hybrid_indexed", "deepseek_sparse_attention"}
        )
        if unsupported_layer_types:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported_layer_types}.")
        output_gate_type = self.output_gate_type or self.hidden_act
        if output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError(f"Unsupported Qwen4-Exp output gate activation: {output_gate_type}.")
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}.")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, num_experts], "
                f"got {self.num_experts_per_tok} and {self.num_experts}."
            )
        if self.moe_intermediate_size <= 0 or self.shared_expert_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size and shared_expert_intermediate_size must be > 0.")
        qsa_fields = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        qsa_values = {name: getattr(self, name) for name in qsa_fields}
        if any(value is not None for value in qsa_values.values()):
            missing = [name for name, value in qsa_values.items() if value is None]
            if missing:
                raise ValueError(f"QSA config is missing required fields: {missing}.")
            if any(value <= 0 for value in qsa_values.values()):
                raise ValueError(f"QSA config values must be positive: {qsa_values}.")
            if self.indexer_kv_heads != 1:
                raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
            if self.indexer_budget % self.indexer_compress_ratio != 0:
                raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")

        if self.ple_layer_ids:
            ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            if ngram_heads <= 0 or self.ple_embed_dim <= 0 or self.ple_embed_dim % ngram_heads != 0:
                raise ValueError(
                    "ple_embed_dim and the total number of n-gram heads must be positive, and ple_embed_dim must be "
                    f"divisible by the number of heads: {self.ple_embed_dim} % {ngram_heads} != 0."
                )
            invalid_ple_layers = [
                layer_id for layer_id in self.ple_layer_ids if layer_id < 1 or layer_id > self.num_hidden_layers
            ]
            if invalid_ple_layers:
                raise ValueError(
                    f"ple_layer_ids must contain one-indexed ids in [1, {self.num_hidden_layers}], "
                    f"got {invalid_ple_layers}."
                )
            if self.eos_token_id is None or isinstance(self.eos_token_id, list) and not self.eos_token_id:
                raise ValueError("eos_token_id must be set when Qwen4-Exp PLE layers are enabled.")

        partial_rotary_factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0)
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        if rotary_dim > self.indexer_head_dim:
            raise ValueError(
                f"Qwen4-Exp attention RoPE dimensions must fit the QSA index head: rotary_dim={rotary_dim}, "
                f"indexer_head_dim={self.indexer_head_dim}."
            )


@auto_docstring(checkpoint="Qwen/Qwen4-Exp")
@strict
class Qwen4ExpVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "qwen4_exp_vision"
    base_model_fsdp_plan = None


@auto_docstring(checkpoint="Qwen/Qwen4-Exp")
@strict
class Qwen4ExpConfig(Qwen3_5MoeConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    >>> configuration = Qwen4ExpConfig()
    >>> model = Qwen4ExpForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "qwen4_exp"
    base_model_fsdp_plan = None


class Qwen4ExpVisionRotaryEmbedding(Qwen3_5VisionRotaryEmbedding):
    pass


class Qwen4ExpTextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    pass


class Qwen4ExpRMSNorm(Qwen3_5RMSNorm):
    pass


class Qwen4ExpRMSNormGated(Qwen3NextRMSNormGated):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__(hidden_size, eps)
        self.activation = activation


@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type or config.hidden_act
        )


def apply_rotary_pos_emb(hidden_states, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the `hidden_states` tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        hidden_states (`torch.Tensor`): The input tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    rope, nope = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    rope = (rope * cos) + (rotate_half(rope) * sin)

    # Concatenate back to full shape
    return torch.cat([rope, nope], dim=-1)


class Qwen4ExpQSAIndexer(nn.Module):
    """Select QSA token indices from compressed key blocks."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def build_sparse_attention_mask(
        selected_token_indices: torch.Tensor,
        target_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        additive_attention_mask = torch.full(
            (*selected_token_indices.shape[:-1], target_length + 1),
            torch.finfo(dtype).min,
            device=selected_token_indices.device,
            dtype=dtype,
        )
        scatter_indices = torch.where(
            selected_token_indices >= 0,
            selected_token_indices.long(),
            target_length,
        )
        additive_attention_mask.scatter_(-1, scatter_indices, 0)
        return additive_attention_mask[..., :target_length].unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        key_length: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_shape = (batch_size, sequence_length, -1, self.index_head_dim)
        cos, sin = position_embeddings

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, token_k = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)

        indexer_states = torch.cat([token_k, cos.to(token_k.dtype), sin.to(token_k.dtype)], dim=-1)
        if past_key_values is not None:
            indexer_states = past_key_values.update_indexer(indexer_states, self.layer_idx)
        indexer_states = indexer_states[:, :key_length]

        rotary_dim = cos.shape[-1]
        raw_keys, key_cos, key_sin = torch.split(indexer_states, [self.index_head_dim, rotary_dim, rotary_dim], dim=-1)

        # If we don't have the mask, we will need those values to recreate valid indices
        if attention_mask is None:
            q_length = sequence_length
            if past_key_values is not None:
                q_offset = past_key_values.get_query_offset(self.layer_idx)
                kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, self.layer_idx)
            else:
                q_offset = 0
                kv_length, kv_offset = q_length, 0

        selected_token_indices = torch.full(
            (batch_size, sequence_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(sequence_length):
                # Only eager and sdpa are accepted, so the mask is always either None or 4D here
                # If it's None, it means that we rely on the lengths and offsets to get valid indices
                if attention_mask is None:
                    query_position = q_offset + query_idx
                    key_positions = torch.arange(kv_length, device=hidden_states.device) + kv_offset
                    visible_token_indices = key_positions <= query_position
                # Always 4D with either bool (sdpa) or float (eager)
                else:
                    mask_slice = attention_mask[batch_idx, 0, query_idx, :]
                    visible_token_indices = mask_slice if mask_slice.dtype == torch.bool else mask_slice == 0

                num_complete_blocks = visible_token_indices.numel() // self.compress_ratio
                selected_tokens = visible_token_indices.new_empty((0,))
                if num_complete_blocks > 0:
                    block_token_indices = visible_token_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        key_cos[batch_idx].index_select(0, group_starts),
                        key_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                    selected_tokens = selected_tokens[: self.token_budget]
                tail = visible_token_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        return selected_token_indices


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = Qwen4ExpQSAIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_length = hidden_states.shape[1]
        past_length = 0
        target_length = query_length
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length(self.layer_idx)
            past_length = int(past_length.item()) if isinstance(past_length, torch.Tensor) else past_length
            target_length, _ = past_key_values.get_mask_sizes(query_length, self.layer_idx)
        selected_token_indices = self.indexer(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            key_length=past_length + query_length,
        )
        attention_mask = self.indexer.build_sparse_attention_mask(
            selected_token_indices,
            target_length=target_length,
            dtype=hidden_states.dtype,
        )

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )


class Qwen4ExpExperts(Qwen3NextExperts):
    pass


class Qwen4ExpTopKRouter(Qwen3NextTopKRouter):
    pass


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    pass


class Qwen4ExpGroupedRMSNorm(Qwen4ExpRMSNorm):
    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps=eps)
        if hidden_size % group_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")
        self.group_size = group_size

    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grouped_shape = (*hidden_states.shape[:-1], -1, self.group_size)
        return super()._norm(hidden_states.reshape(grouped_shape)).flatten(-2)


class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            hc_hidden_size,
            eps=config.rms_norm_eps,
            group_size=self.hidden_size,
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, got {hyper_input.shape[-1]}."
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, (hyper_input, injection_weights)

    @staticmethod
    def combine(
        block_output: torch.Tensor,
        residual_context: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hyper_input, injection_weights = residual_context
        injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int) -> torch.Tensor:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        self.eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id

        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size

        self.layer_multipliers = nn.Buffer(
            _build_layer_multipliers(self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed)
        )
        self.ngram_heads_vocab_sizes = nn.Buffer(torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.ngram_heads_offsets = nn.Buffer(torch.tensor(self.head_offsets, dtype=torch.long))
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.ngram_embedding = nn.Embedding(padded_vocab_size, head_dim_per_ngram)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def _get_previous_context(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        context_len = self.ngram_size - 1
        previous_context = input_ids.new_full((input_ids.shape[0], context_len), self.eos_token_id)
        if past_key_values is None:
            return previous_context

        if past_key_values.has_previous_state(layer_idx, state_idx=2):
            context_update = input_ids
        else:
            # LinearAttentionLayer pads newly initialized states with zeros, while Qwen4-Exp n-grams must be padded
            # with EOS. Seed the state explicitly on its first update to preserve prefill/decode parity.
            context_update = torch.cat([previous_context, input_ids], dim=-1)
        context_states = past_key_values.update_conv_state(
            context_update,
            layer_idx,
            state_idx=2,
            conv_kernel_size=context_len,
        )
        context_end = context_states.shape[-1] - input_ids.shape[-1]
        return context_states[..., context_end - context_len : context_end].to(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        input_ids = input_ids.long()
        previous_context = self._get_previous_context(input_ids, past_key_values, layer_idx)
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]
        return self.ngram_embedding(ngram_ids).flatten(-2)


class Qwen4ExpPLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream.

    PLE projects each token's concatenated n-gram embedding to a shared value and one key per residual stream. The
    normalized stream activations gate those values, then a dilated depthwise convolution adds local lexical context.
    The returned tensor has shape `(batch_size, sequence_length, hc_count * hidden_size)`.
    """

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        ple_embed_dim = config.ple_embed_dim
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(config, ple_embed_dim, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpGroupedRMSNorm(hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size)
        self.norm_query = Qwen4ExpGroupedRMSNorm(hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size)
        self.norm_conv = Qwen4ExpGroupedRMSNorm(hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )

    def _short_conv(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        sequence_length = hidden_states.shape[-1]
        conv_input = hidden_states
        if self.short_conv_state_len:
            if past_key_values is not None:
                conv_input = past_key_values.update_conv_state(
                    hidden_states,
                    self.layer_idx,
                    state_idx=1,
                    conv_kernel_size=self.short_conv_state_len,
                )
            conv_input = F.pad(conv_input, (self.short_conv_state_len, 0))
            conv_input = conv_input[..., -(self.short_conv_state_len + sequence_length) :]
        return F.silu(self.conv1d(conv_input)).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        ple_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values, self.layer_idx)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if ple_padding_mask is not None:
            current_ple_padding_mask = ple_padding_mask[:, -input_ids.shape[1] :].unsqueeze(-1).to(gated_value.dtype)
            gated_value = gated_value * current_ple_padding_mask
            gated_value_normed = gated_value_normed * current_ple_padding_mask
        output = gated_value + self._short_conv(gated_value_normed, past_key_values)
        return output if ple_padding_mask is None else output * current_ple_padding_mask


class Qwen4ExpDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_idx)
        self.mlp = Qwen4ExpSparseMoeBlock(config)
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        ple_padding_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        if self.ple is not None:
            if ple_input_ids is None:
                raise ValueError("input_ids are required when Qwen4-Exp PLE layers are enabled.")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                ple_input_ids,
                past_key_values,
                ple_padding_mask=ple_padding_mask,
            )

        hidden_states, residual = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.attn_hyper_connection.combine(hidden_states, residual)

        hidden_states, residual = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.mlp_hyper_connection.combine(hidden_states, residual)


class Qwen4ExpPreTrainedModel(Qwen3_5MoePreTrainedModel):
    config: Qwen4ExpConfig
    _no_split_modules = ["Qwen4ExpDecoderLayer", "Qwen4ExpVisionBlock"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen4ExpTopKRouter, index=0),
        "attentions": Qwen4ExpAttention,
    }
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_flex_attn = False
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen4ExpNGramEmbedding):
            init.copy_(
                module.layer_multipliers,
                _build_layer_multipliers(
                    module.unigram_vocab_size, module.ngram_size, module.ple_layer_index, module.seed
                ),
            )
            init.copy_(module.ngram_heads_vocab_sizes, torch.tensor(module.head_vocab_sizes, dtype=torch.long))
            init.copy_(module.ngram_heads_offsets, torch.tensor(module.head_offsets, dtype=torch.long))
        elif isinstance(module, Qwen4ExpGroupedRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen4ExpPLELayer):
            init.zeros_(module.conv1d.weight)


class Qwen4ExpVisionModel(Qwen3_5MoeVisionModel):
    config: Qwen4ExpVisionConfig


class Qwen4ExpModelOutputWithPast(Qwen3_5MoeModelOutputWithPast):
    pass


class Qwen4ExpTextModel(Qwen3_5MoeTextModel):
    config: Qwen4ExpTextConfig

    def __init__(self, config: Qwen4ExpTextConfig):
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen4ExpDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=config)
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self.gradient_checkpointing = False
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
        use_cache: bool | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used by Per-Layer Embedding (PLE). This is only needed when PLE is enabled and
            `inputs_embeds` are passed instead of `input_ids`.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("input_ids and inputs_embeds cannot both be used as model inputs.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if input_ids is not None:
            ple_input_ids = input_ids
        if self.config.ple_layer_ids and ple_input_ids is None:
            raise ValueError(
                "ple_input_ids must be provided when Qwen4-Exp PLE layers are enabled and inputs_embeds are used."
            )
        if ple_input_ids is not None and ple_input_ids.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                "ple_input_ids must have the same batch size and sequence length as inputs_embeds, but got "
                f"{tuple(ple_input_ids.shape)} and {tuple(inputs_embeds.shape[:2])}."
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        ple_padding_mask = causal_mask_mapping.get("linear_attention")
        if self.config.ple_layer_ids and ple_input_ids is not None and ple_padding_mask is not None:
            eos_token_id = self.config.eos_token_id
            eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            ple_input_ids = torch.where(ple_padding_mask.bool(), ple_input_ids, eos_token_id)

        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[layer_idx]
            mask_key = "linear_attention" if layer_type == "linear_attention" else "full_attention"
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[mask_key],
                past_key_values=past_key_values,
                ple_input_ids=ple_input_ids,
                ple_padding_mask=ple_padding_mask,
                use_cache=use_cache,
                **kwargs,
            )
            if output_hidden_states:
                layer_hidden_states = self.hyper_connection_mixer(hidden_states)
                all_hidden_states += (layer_hidden_states,)

        hidden_states = self.hyper_connection_mixer(hidden_states)
        return Qwen4ExpModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class Qwen4ExpModel(Qwen3_5MoeModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen4ExpModelOutputWithPast:
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used by Per-Layer Embedding (PLE) when inputs_embeds are passed instead of input_ids.
            When input_ids are provided, they are always used for PLE.
        """
        if input_ids is not None:
            ple_input_ids = input_ids
        kwargs["ple_input_ids"] = ple_input_ids
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )


class Qwen4ExpForCausalLM(Qwen3_5MoeForCausalLM):
    config: Qwen4ExpTextConfig

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
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used by Per-Layer Embedding (PLE) when inputs_embeds are passed instead of input_ids.
        """
        kwargs["ple_input_ids"] = ple_input_ids
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


class Qwen4ExpForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used by Per-Layer Embedding (PLE) when inputs_embeds are passed instead of input_ids.
        """
        kwargs["ple_input_ids"] = ple_input_ids
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = [
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpVisionConfig",
    "Qwen4ExpVisionModel",
    "Qwen4ExpTextModel",
    "Qwen4ExpModel",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpPreTrainedModel",
]
