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
from ...cache_utils import Cache, DynamicCache, DynamicLayer, StaticLayer
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_kernelized_func
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
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
    Qwen3NextMLP,
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

        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if (i + 1) % interval_pattern else "qwen_sparse_attention"
                for i in range(self.num_hidden_layers)
            ]
        # The real checkpoint contains "full_attention" entries for layers that are actually using an indexer
        elif "full_attention" in self.layer_types:
            self.layer_types = [
                "qwen_sparse_attention" if layer == "full_attention" else layer for layer in self.layer_types
            ]

        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates Qwen4-Exp architecture invariants."""
        unsupported_layer_types = sorted(set(self.layer_types) - {"linear_attention", "qwen_sparse_attention"})
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
            partial_rotary_factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial_rotary_factor)
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    f"Qwen4-Exp attention RoPE dimensions must fit the QSA index head: rotary_dim={rotary_dim}, "
                    f"indexer_head_dim={self.indexer_head_dim}."
                )

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
            non_linear_ple_layers = [
                layer_id for layer_id in self.ple_layer_ids if self.layer_types[layer_id - 1] != "linear_attention"
            ]
            if non_linear_ple_layers:
                raise ValueError(
                    "Qwen4-Exp PLE is only supported on linear_attention layers, "
                    f"got PLE on layers {non_linear_ple_layers}."
                )
            if self.eos_token_id is None or isinstance(self.eos_token_id, list) and not self.eos_token_id:
                raise ValueError("eos_token_id must be set when Qwen4-Exp PLE layers are enabled.")


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


class Qwen4ExpTextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    pass


class Qwen4ExpTextRMSNorm(Qwen3_5RMSNorm):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__(dim, eps=eps)
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out


class Qwen4ExpTextRMSNormGated(Qwen3NextRMSNormGated):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__(hidden_size, eps)
        self.activation = activation


@use_kernelized_func(
    [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, causal_conv1d_fn, causal_conv1d_update]
)
class Qwen4ExpTextGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.norm = Qwen4ExpTextRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon, activation=config.output_gate_type or config.hidden_act
        )


class Qwen4ExpSparseAttentionCacheLayerMixin:
    """Cache completed QSA block keys and the raw states of the incomplete block."""

    def __init__(self, config: Qwen4ExpTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.indexer_compress_ratio = config.indexer_compress_ratio
        self.indexer_keys: torch.Tensor | None = None
        self.indexer_buffer: torch.Tensor | None = None

    def _ensure_indexer_key_capacity(self, block_keys: torch.Tensor, block_capacity: int) -> None:
        current_capacity = 0 if self.indexer_keys is None else self.indexer_keys.shape[1] - 1
        if (
            self.indexer_keys is not None
            and current_capacity == block_capacity
            and self.indexer_keys.shape[0] == block_keys.shape[0]
        ):
            return
        resized_keys = block_keys.new_zeros((block_keys.shape[0], block_capacity + 1, block_keys.shape[-1]))
        if self.indexer_keys is not None:
            copy_length = min(current_capacity, block_capacity)
            resized_keys[:, :copy_length] = self.indexer_keys[:, :copy_length]
        self.indexer_keys = resized_keys

    def get_indexer_buffer(self, indexer_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, state_dim = indexer_states.shape
        if self.indexer_buffer is None:
            return indexer_states.new_zeros((batch_size, self.indexer_compress_ratio - 1, state_dim))
        return self.indexer_buffer

    def update_indexer(
        self,
        block_keys: torch.Tensor,
        block_offsets: torch.LongTensor,
        new_block_counts: torch.LongTensor,
        buffer: torch.Tensor,
        block_capacity: int,
    ) -> torch.Tensor:
        max_cache_len = self.get_max_length()
        if max_cache_len >= 0:
            block_capacity = max_cache_len // self.indexer_compress_ratio
        self._ensure_indexer_key_capacity(block_keys, block_capacity)
        if self.indexer_buffer is None:
            self.indexer_buffer = torch.zeros_like(buffer)
            if self.is_compileable and not is_torchdynamo_compiling():
                torch._dynamo.mark_static_address(self.indexer_keys)
                torch._dynamo.mark_static_address(self.indexer_buffer)

        num_new_slots = block_keys.shape[1]
        new_slot_indices = torch.arange(num_new_slots, device=block_keys.device)
        block_positions = block_offsets.unsqueeze(-1) + new_slot_indices
        valid_blocks = new_slot_indices < new_block_counts.unsqueeze(-1)
        # The extra final cache slot receives padded block entries and is never returned.
        block_positions = block_positions.masked_fill(~valid_blocks, block_capacity)
        self.indexer_keys.scatter_(1, block_positions.unsqueeze(-1).expand_as(block_keys), block_keys)
        self.indexer_keys[:, block_capacity].zero_()
        self.indexer_buffer.copy_(buffer)
        return self.indexer_keys[:, :block_capacity]

    def reset(self) -> None:
        super().reset()
        if self.indexer_keys is None:
            return
        if self.is_compileable:
            self.indexer_keys.zero_()
            self.indexer_buffer.zero_()
        else:
            self.indexer_keys = None
            self.indexer_buffer = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.indexer_keys is not None:
            beam_idx = beam_idx.to(self.indexer_keys.device)
            self.indexer_keys = self.indexer_keys.index_select(0, beam_idx)
            self.indexer_buffer = self.indexer_buffer.index_select(0, beam_idx)


class Qwen4ExpDynamicSparseAttentionLayer(Qwen4ExpSparseAttentionCacheLayerMixin, DynamicLayer):
    _layer_type = "qwen_sparse_attention"

    def crop(self, tokens_to_remove: int) -> None:
        current_length = self.get_seq_length()
        is_crop = tokens_to_remove < 0 or 0 < tokens_to_remove < current_length
        if is_crop and self.indexer_keys is not None:
            raise ValueError("Qwen4-Exp block-compressed QSA caches do not support cropping.")
        super().crop(tokens_to_remove)

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.indexer_keys is not None:
            self.indexer_keys = self.indexer_keys.repeat_interleave(repeats, dim=0)
            self.indexer_buffer = self.indexer_buffer.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.indexer_keys is not None:
            self.indexer_keys = self.indexer_keys[indices, ...]
            self.indexer_buffer = self.indexer_buffer[indices, ...]


class Qwen4ExpStaticSparseAttentionLayer(Qwen4ExpSparseAttentionCacheLayerMixin, StaticLayer):
    _layer_type = "qwen_sparse_attention"


def apply_rotary_pos_emb_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply partial RoPE to one tensor with broadcast-ready cosine and sine tensors."""
    rotary_dim = cos.shape[-1]
    x_rope, x_nope = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rope = (x_rope * cos) + (rotate_half(x_rope) * sin)
    return torch.cat([x_rope, x_nope], dim=-1)


class Qwen4ExpTextQSAIndexer(nn.Module):
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
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    def _group_indexer_states(
        self,
        indexer_states: torch.Tensor,
        current_valid_mask: torch.BoolTensor,
        buffer: torch.Tensor,
        buffer_counts: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        batch_size, _, state_dim = indexer_states.shape
        buffer_length = self.compress_ratio - 1
        buffer_positions = torch.arange(buffer_length, device=indexer_states.device)
        buffer_valid = buffer_positions < buffer_counts.unsqueeze(-1)
        combined_states = torch.cat([buffer, indexer_states], dim=1)
        combined_valid = torch.cat([buffer_valid, current_valid_mask], dim=1)

        # Compact valid states so left/right padding does not change content-relative block boundaries.
        combined_length = combined_states.shape[1]
        packed_positions = combined_valid.long().cumsum(dim=-1) - 1
        packed_positions = packed_positions.masked_fill(~combined_valid, combined_length)
        packed_states = combined_states.new_zeros((batch_size, combined_length + 1, state_dim))
        packed_states.scatter_(
            1,
            packed_positions.unsqueeze(-1).expand_as(combined_states),
            combined_states.masked_fill(~combined_valid.unsqueeze(-1), 0),
        )

        total_counts = combined_valid.sum(dim=-1)
        new_block_counts = total_counts // self.compress_ratio
        num_new_slots = combined_length // self.compress_ratio
        block_states = packed_states[:, : num_new_slots * self.compress_ratio].view(
            batch_size, num_new_slots, self.compress_ratio, state_dim
        )

        buffer_offsets = new_block_counts.unsqueeze(-1) * self.compress_ratio + buffer_positions
        new_buffer = packed_states.gather(
            1, buffer_offsets.clamp(max=combined_length).unsqueeze(-1).expand(-1, -1, state_dim)
        )
        return block_states, new_block_counts, new_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None,
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
        q, token_k = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb_single(q, cos.unsqueeze(2), sin.unsqueeze(2))

        indexer_states = torch.cat([token_k, cos.to(token_k.dtype), sin.to(token_k.dtype)], dim=-1)

        # Eager/SDPA provide a 4D bool/additive mask. QSA follows the model's contiguous left/right-padding contract
        # (not packed sequences), so every query sees a prefix of one valid-key sequence and block keys can be reused.
        visible_token_mask = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        visible_token_mask = visible_token_mask[:, 0]
        key_length = visible_token_mask.shape[-1]
        valid_key_mask = visible_token_mask.any(dim=1)
        valid_key_counts = valid_key_mask.sum(dim=-1)
        first_valid_key = valid_key_mask.int().argmax(dim=-1)
        key_offsets = torch.arange(key_length + 1, device=hidden_states.device)
        ordered_key_indices = first_valid_key.unsqueeze(-1) + key_offsets
        ordered_key_indices = torch.where(
            key_offsets < valid_key_counts.unsqueeze(-1), ordered_key_indices, key_length
        )

        past_length = 0
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            if not isinstance(cache_layer, Qwen4ExpSparseAttentionCacheLayerMixin):
                raise ValueError(
                    "Qwen4-Exp QSA requires a cache initialized from a config containing qwen_sparse_attention layers."
                )
            past_length = cache_layer.get_seq_length()
            buffer = cache_layer.get_indexer_buffer(indexer_states)
        else:
            buffer = indexer_states.new_zeros((batch_size, self.compress_ratio - 1, indexer_states.shape[-1]))

        current_key_positions = past_length + torch.arange(sequence_length, device=hidden_states.device)
        current_valid_mask = valid_key_mask.gather(1, current_key_positions.expand(batch_size, -1))
        past_valid_counts = valid_key_counts - current_valid_mask.sum(dim=-1)
        buffer_counts = past_valid_counts % self.compress_ratio
        block_states, new_block_counts, buffer = self._group_indexer_states(
            indexer_states, current_valid_mask, buffer, buffer_counts
        )

        rotary_dim = cos.shape[-1]
        raw_keys, key_cos, key_sin = torch.split(block_states, [self.index_head_dim, rotary_dim, rotary_dim], dim=-1)
        pooled_keys = self.k_layernorm(raw_keys.float().mean(dim=2).to(raw_keys.dtype))
        block_key_states = apply_rotary_pos_emb_single(pooled_keys, key_cos[:, :, 0], key_sin[:, :, 0])

        block_capacity = key_length // self.compress_ratio
        if past_key_values is not None:
            block_key_states = cache_layer.update_indexer(
                block_key_states,
                past_valid_counts // self.compress_ratio,
                new_block_counts,
                buffer,
                block_capacity,
            )
        else:
            block_key_states = block_key_states[:, :block_capacity]

        num_blocks = block_key_states.shape[1]
        block_token_indices = ordered_key_indices[:, : num_blocks * self.compress_ratio].view(
            batch_size, num_blocks, self.compress_ratio
        )

        visible_token_counts = visible_token_mask.sum(dim=-1)
        num_visible_blocks = visible_token_counts // self.compress_ratio
        selected_token_mask = torch.zeros(
            (batch_size, sequence_length, key_length + 1), device=attention_mask.device, dtype=torch.bool
        )
        num_topk_blocks = min(self.block_topk, num_blocks)
        block_indices = torch.arange(num_blocks, device=hidden_states.device)
        tail_offsets = torch.arange(self.compress_ratio - 1, device=hidden_states.device)
        transposed_block_keys = block_key_states.float().transpose(-1, -2).unsqueeze(1)
        # Bound the temporary `[batch, queries, indexer_heads, blocks]` score tensor for long prefills.
        max_score_elements = 1 << 22  # 16 MiB in float32
        score_elements_per_query = batch_size * self.index_n_heads * max(num_blocks, 1)
        query_chunk_size = min(sequence_length, max(1, max_score_elements // score_elements_per_query))
        for query_start in range(0, sequence_length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, sequence_length)
            query = q[:, query_start:query_end]
            query_num_visible_blocks = num_visible_blocks[:, query_start:query_end]

            scores = torch.matmul(query.float(), transposed_block_keys)
            scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)
            scores = scores.masked_fill(
                block_indices >= query_num_visible_blocks.unsqueeze(-1), torch.finfo(scores.dtype).min
            )
            selected_blocks = scores.topk(num_topk_blocks, dim=-1).indices

            # Expand selected blocks back to token indices, then append the current incomplete block as the tail.
            selected_block_tokens = (
                block_token_indices[:, None]
                .expand(-1, query.shape[1], -1, -1)
                .gather(2, selected_blocks.unsqueeze(-1).expand(-1, -1, -1, self.compress_ratio))
            )
            valid_selected_blocks = selected_blocks < query_num_visible_blocks.unsqueeze(-1)
            selected_block_tokens = selected_block_tokens.masked_fill(~valid_selected_blocks.unsqueeze(-1), -1)

            tail_positions = query_num_visible_blocks.unsqueeze(-1) * self.compress_ratio + tail_offsets
            valid_tail = tail_positions < visible_token_counts[:, query_start:query_end].unsqueeze(-1)
            tail_tokens = (
                ordered_key_indices[:, None]
                .expand(-1, query.shape[1], -1)
                .gather(2, tail_positions.clamp(max=key_length))
            )
            selected_tokens = torch.cat(
                [selected_block_tokens.flatten(2), tail_tokens.masked_fill(~valid_tail, -1)], dim=-1
            )
            # Invalid selections land in the extra final column, which is removed below.
            scatter_indices = selected_tokens.masked_fill(selected_tokens < 0, key_length)
            selected_token_mask[:, query_start:query_end].scatter_(-1, scatter_indices, True)

        selected_token_mask = selected_token_mask[..., :key_length].unsqueeze(1)
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

        return selected_token_mask


class Qwen4ExpTextAttention(Qwen3_5Attention):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        # Combine both masks (they are never None, and are always 4D with either bool for sdpa, or float for eager)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )


class Qwen4ExpTextMLP(Qwen3NextMLP):
    pass


class Qwen4ExpTextExperts(Qwen3NextExperts):
    pass


class Qwen4ExpTextTopKRouter(Qwen3NextTopKRouter):
    pass


class Qwen4ExpTextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(config, intermediate_size=config.shared_expert_intermediate_size)


class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
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
        return mixed_input, hyper_input, injection_weights


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


class Qwen4ExpTextNGramEmbedding(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, layer_idx: int, ple_layer_index: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
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

    def forward(self, input_ids: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        input_ids = input_ids.long()
        # This is a trick to store the previous N=self.context_len `input_ids` - indeed the manipulations are identical to storing
        # a past conv_state, so we can use an additional conv_states inside the Cache for it
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx, state_idx=2):
            previous_context = past_key_values.layers[self.layer_idx].conv_states[2].clone()
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        # Store the current input_ids for the next forward
        if past_key_values is not None:
            input_ids_to_cache = input_ids
            # In the case where `input_ids` would be smaller than `self.context_len`, the `update_conv_state` will pad with zeros, whereas
            # here we want to pad with eos, so we do it explicitly
            if (
                not past_key_values.has_previous_state(self.layer_idx, state_idx=2)
                and input_ids.shape[1] < self.context_len
            ):
                input_ids_to_cache = torch.nn.functional.pad(
                    input_ids_to_cache, (self.context_len - input_ids.shape[1], 0), value=self.eos_token_id
                )
            _ = past_key_values.update_conv_state(
                input_ids_to_cache, self.layer_idx, state_idx=2, conv_kernel_size=self.context_len
            )

        # Get full token history
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


class Qwen4ExpTextPLELayer(nn.Module):
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
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
        )

    def _short_conv(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)

        if past_key_values is not None:
            hidden_states = past_key_values.update_conv_state(
                hidden_states, self.layer_idx, state_idx=1, conv_kernel_size=self.short_conv_state_len
            )

        # We always pad and slice due to the dilation in the conv, to make sure we have enough states
        hidden_states = F.pad(hidden_states, (self.short_conv_state_len, 0))
        hidden_states = hidden_states[..., -(self.short_conv_state_len + seq_len) :]

        # We cannot use the usual functions/kernels here for the short conv as the conv1d has dilation
        hidden_states = F.silu(self.conv1d(hidden_states))

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
        conv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = apply_mask_to_padding_states(gated_value_normed, conv_mask)
        output = gated_value + self._short_conv(gated_value_normed, past_key_values)
        return output


class Qwen4ExpTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen4ExpTextAttention(config, layer_idx)
        self.mlp = Qwen4ExpTextSparseMoeBlock(config)
        ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
        self.ple = Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpTextGatedResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        conv_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask
            )

        hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)

        hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        hidden_states = self.mlp(hidden_states)

        injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        hidden_states = hyper_input + injection.flatten(-2)
        return hidden_states


@auto_docstring
class Qwen4ExpPreTrainedModel(Qwen3_5MoePreTrainedModel):
    config: Qwen4ExpConfig
    _no_split_modules = None  # will be set on text and vision separately
    _can_record_outputs = None  # will be set on text and vision separately
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_flex_attn = False
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Qwen4ExpTextGatedDeltaNet):
            init.ones_(module.dt_bias)
            # Lower bound kept away from 0 so log(A) never becomes -inf
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen4ExpTextRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen4ExpTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen4ExpTextSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen4ExpVisionRotaryEmbedding):  # noqa
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)
        if isinstance(module, Qwen4ExpTextNGramEmbedding):
            init.copy_(
                module.layer_multipliers,
                _build_layer_multipliers(
                    module.unigram_vocab_size, module.ngram_size, module.ple_layer_index, module.seed
                ),
            )
            init.copy_(module.ngram_heads_vocab_sizes, torch.tensor(module.head_vocab_sizes, dtype=torch.long))
            init.copy_(module.ngram_heads_offsets, torch.tensor(module.head_offsets, dtype=torch.long))
        elif isinstance(module, Qwen4ExpTextPLELayer):
            init.zeros_(module.conv1d.weight)


class Qwen4ExpModelOutputWithPast(Qwen3_5MoeModelOutputWithPast):
    pass


@auto_docstring
class Qwen4ExpTextModel(Qwen3_5MoeTextModel):
    config: Qwen4ExpTextConfig
    _no_split_modules = ["Qwen4ExpTextDecoderLayer"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen4ExpTextTopKRouter, index=0),
        "attentions": Qwen4ExpTextAttention,
        "hidden_states": Qwen4ExpTextDecoderLayer,
    }

    def __init__(self, config: Qwen4ExpTextConfig):
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen4ExpTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=config)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)
        self.gradient_checkpointing = False
        self.post_init()

    def reverse_embedding(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Recreate `input_ids` from `inputs_embeds` by reverting the embedding matrix. This is needed when the user only provides
        `inputs_embeds` and ple is active.
        """
        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        with torch.no_grad():
            input_ids = (
                (inputs_embeds[:, :, None, :] == self.embed_tokens.weight[None, None, :, :]).all(dim=3).nonzero()[:, 2]
            )
            try:
                input_ids = input_ids.view(inputs_embeds.shape[:2])
            except RuntimeError:
                raise RuntimeError(
                    "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that "
                    "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse "
                    "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                )
        return input_ids

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
        ple_input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used by Per-Layer Embedding (PLE). This is only needed when PLE is enabled and
            `inputs_embeds` are passed instead of `input_ids`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.ple_layer_ids and ple_input_ids is None:
            # If we do not have input_ids but have ple, we need to revert the embeddings to find back the ids
            ple_input_ids = input_ids if input_ids is not None else self.reverse_embedding(inputs_embeds)

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
                # Due to the indexer, we always want to create a mask to then simply overlay the indexer mask in each layer - otherwise
                # we may have to recreate it in each layer if it gets skipped
                "allow_is_causal_skip": False,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        conv_mask = causal_mask_mapping.get("linear_attention")
        if self.config.ple_layer_ids and conv_mask is not None:
            eos_token_id = self.config.eos_token_id
            eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos_token_id)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping["full_attention"],
                conv_mask=conv_mask,
                past_key_values=past_key_values,
                ple_input_ids=ple_input_ids,
                **kwargs,
            )

        hidden_states = self.hyper_connection_mixer(hidden_states)

        return Qwen4ExpModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen4ExpForCausalLM(Qwen3_5MoeForCausalLM):
    config: Qwen4ExpTextConfig

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        **kwargs,
    ) -> dict:
        # We need to overwrite to add the `allow_is_causal_skip=False` condition
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            # Due to the indexer, we always want to create a mask to then simply overlay the indexer mask in each layer - otherwise
            # we may have to recreate it in each layer if it gets skipped
            "allow_is_causal_skip": False,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        return causal_mask_mapping


@auto_docstring
class Qwen4ExpVisionModel(Qwen3_5MoeVisionModel):
    _no_split_modules = ["Qwen4ExpVisionBlock"]
    config: Qwen4ExpVisionConfig


class Qwen4ExpModel(Qwen3_5MoeModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen4ExpModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        ple_input_ids = None
        if self.config.text_config.ple_layer_ids:
            # If we do not have input_ids but have ple, we need to revert the embeddings to find back the ids
            ple_input_ids = (
                input_ids if input_ids is not None else self.language_model.reverse_embedding(inputs_embeds)
            )

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )

        return Qwen4ExpModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


@auto_docstring
class Qwen4ExpForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        **kwargs,
    ) -> dict:
        # We need to overwrite to add the `allow_is_causal_skip=False` condition
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            # Due to the indexer, we always want to create a mask to then simply overlay the indexer mask in each layer - otherwise
            # we may have to recreate it in each layer if it gets skipped
            "allow_is_causal_skip": False,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        return causal_mask_mapping


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
