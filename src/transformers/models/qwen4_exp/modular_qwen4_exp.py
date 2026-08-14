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
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
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
    apply_rotary_pos_emb,
    eager_attention_forward,
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
    Qwen3NextSparseMoeBlock,
    Qwen3NextTopKRouter,
    load_balancing_loss_func,
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
    output_gate_type (`str`, *optional*):
        Activation used by the output gate of linear attention. If unset, `hidden_act` is used.
    hc_count (`int`, *optional*, defaults to 4):
        Number of residual streams used by the hyper-connections.
    hc_lowrank (`int`, *optional*, defaults to 320):
        Rank of the learned hyper-connection input mixer.
    ple_layer_ids (`list[int]`, *optional*):
        One-indexed decoder layer ids that use the positional lexical embedding (PLE) module.
    ple_embed_dim (`int`, *optional*):
        Size of the concatenated n-gram embeddings. Defaults to `hidden_size`.
    ple_conv_kernel_size (`int`, *optional*, defaults to 4):
        Kernel size of the dilated depthwise convolution in each PLE module.
    ngram_size (`int`, *optional*, defaults to 3):
        Largest token n-gram represented by PLE.
    heads_per_ngram (`int`, *optional*, defaults to 8):
        Number of independently hashed embedding heads for every n-gram order.
    ngram_vocab_size_base (`int`, *optional*, defaults to 20000000):
        Base prime vocabulary size used by the hashed n-gram heads.
    make_ngram_vocab_size_divisible_by (`int`, *optional*, defaults to 128):
        Divisor used to pad the combined n-gram embedding vocabulary.
    seed (`int`, *optional*, defaults to 1234):
        Seed used to deterministically derive the per-layer n-gram hash multipliers.
    moe_intermediate_size (`int`, *optional*, defaults to 512):
        Intermediate size of each routed expert.
    shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
        Intermediate size of the shared expert.
    num_experts_per_tok (`int`, *optional*, defaults to 10):
        Number of routed experts selected for every token.
    num_experts (`int`, *optional*, defaults to 512):
        Number of routed experts.
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
    split_ngram_parts (`int`, *optional*, defaults to 512):
        Number of checkpoint shards used for each PLE n-gram embedding table. The shards are concatenated into a
        single embedding weight when loading.
    norm_topk_prob (`bool`, *optional*, defaults to `True`):
        Whether to normalize the selected experts' routing probabilities.
    """

    model_type = "qwen4_exp_text"
    base_config_key = "text_config"
    base_model_tp_plan = None
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
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    output_gate_type: str | None = None
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self, **kwargs):
        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}.")
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}.")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}.")
        if self.ple_conv_kernel_size <= 0:
            raise ValueError(f"ple_conv_kernel_size must be > 0, got {self.ple_conv_kernel_size}.")
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0.")
        if self.make_ngram_vocab_size_divisible_by <= 0:
            raise ValueError("make_ngram_vocab_size_divisible_by must be > 0.")
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be > 0.")
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

        self.ple_layer_ids = [] if self.ple_layer_ids is None else sorted(set(self.ple_layer_ids))
        self.ple_embed_dim = self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_embed_dim % ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by the total number of n-gram heads: "
                f"{self.ple_embed_dim} % {ngram_heads} != 0."
            )
        invalid_ple_layers = [
            layer_id for layer_id in self.ple_layer_ids if layer_id < 1 or layer_id > self.num_hidden_layers
        ]
        if invalid_ple_layers:
            raise ValueError(
                f"ple_layer_ids must contain one-indexed ids in [1, {self.num_hidden_layers}], "
                f"got {invalid_ple_layers}."
            )

        # Qwen4-Exp keeps the GatedDeltaNet convolution, PLE convolution and n-gram context in separate cache states.
        # Without PLE, only the GatedDeltaNet state is needed.
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1
        super().__post_init__(**kwargs)
        if self.indexer_n_heads is not None:
            partial_rotary_factor = (self.rope_parameters or {}).get("partial_rotary_factor", 1.0)
            rotary_dim = int(self.head_dim * partial_rotary_factor)
            if rotary_dim > self.indexer_head_dim:
                raise ValueError(
                    "Qwen4-Exp attention RoPE dimensions must fit the QSA index head: "
                    f"rotary_dim={rotary_dim}, indexer_head_dim={self.indexer_head_dim}."
                )

        # Full-attention layers use an indexed cache when QSA is enabled. If PLE is also attached to that layer,
        # the hybrid indexed cache additionally carries its convolution and n-gram context states.
        block_types = self.layers_block_type
        self.layer_types = [
            (
                "hybrid_indexed"
                if self.indexer_n_heads is not None and layer_idx + 1 in self.ple_layer_ids
                else "deepseek_sparse_attention"
                if self.indexer_n_heads is not None
                else "hybrid"
                if layer_idx + 1 in self.ple_layer_ids
                else "full_attention"
            )
            if block_type == "full_attention"
            else block_type
            for layer_idx, block_type in enumerate(block_types)
        ]

    @property
    def layers_block_type(self) -> list[str]:
        full_attention_cache_types = {"deepseek_sparse_attention", "full_attention", "hybrid", "hybrid_indexed"}
        return [
            "full_attention" if layer_type in full_attention_cache_types else layer_type
            for layer_type in self.layer_types
        ]

    @property
    def short_conv_layer_ids(self) -> list[int]:
        return [layer_id - 1 for layer_id in self.ple_layer_ids]

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        if not self.ple_layer_ids:
            return None
        state_len = (self.ple_conv_kernel_size - 1) * self.ngram_size
        return self.hidden_size * self.hc_count, state_len

    @property
    def ngram_context_len(self) -> int:
        return self.ngram_size - 1 if self.ple_layer_ids else 0


@auto_docstring(checkpoint="Qwen/Qwen4-Exp")
@strict
class Qwen4ExpVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "qwen4_exp_vision"


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


class Qwen4ExpVisionRotaryEmbedding(Qwen3_5VisionRotaryEmbedding):
    pass


class Qwen4ExpTextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    pass


class Qwen4ExpRMSNorm(Qwen3_5RMSNorm):
    pass


class Qwen4ExpRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__()
        if activation not in {"sigmoid", "silu"}:
            raise ValueError(f"Unsupported Qwen4-Exp output gate activation: {activation}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        gate = torch.sigmoid(gate.float()) if self.activation == "sigmoid" else F.silu(gate.float())
        return (hidden_states * gate).to(input_dtype)


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=config.output_gate_type or config.hidden_act,
        )


class Qwen4ExpQSAIndexer(nn.Module):
    """Reference QSA indexer matching the Qwen4-Exp SGLang block-selection semantics."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_topk = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _apply_rope(
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int,
    ) -> torch.Tensor:
        hidden_states, _ = apply_rotary_pos_emb(
            hidden_states,
            hidden_states,
            cos,
            sin,
            unsqueeze_dim=unsqueeze_dim,
        )
        return hidden_states

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        qk = self.index_qk_proj(hidden_states)
        q_raw, token_k = torch.split(
            qk,
            [
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ],
            dim=-1,
        )
        q = self.q_layernorm(q_raw.view(batch_size, sequence_length, self.index_n_heads, self.index_head_dim))
        q = self._apply_rope(q, cos, sin, unsqueeze_dim=2)
        token_k = token_k.view(batch_size, sequence_length, self.index_kv_heads, self.index_head_dim)
        return q, token_k.squeeze(2)

    def _compress_keys(
        self,
        raw_keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        block_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        key_groups = raw_keys.index_select(0, block_token_indices.flatten())
        key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
        pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys)
        group_starts = block_token_indices[:, 0]
        return self._apply_rope(
            pooled_keys.unsqueeze(1),
            key_cos.index_select(0, group_starts),
            key_sin.index_select(0, group_starts),
            unsqueeze_dim=1,
        ).squeeze(1)

    def _score_blocks(self, q: torch.Tensor, compressed_keys: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("...hd,nd->...nh", q.float(), compressed_keys.float())
        return torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

    def _select_visible_row(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        key_cos: torch.Tensor,
        key_sin: torch.Tensor,
        visible_indices: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.full(
            (self.token_topk + self.compress_ratio - 1,),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        num_blocks = visible_indices.numel() // self.compress_ratio
        selected_tokens = visible_indices.new_empty((0,))
        if num_blocks:
            block_token_indices = visible_indices[: num_blocks * self.compress_ratio].view(
                num_blocks,
                self.compress_ratio,
            )
            compressed_keys = self._compress_keys(
                raw_keys,
                key_cos,
                key_sin,
                block_token_indices,
            )
            scores = self._score_blocks(q, compressed_keys)
            selected_blocks = scores.topk(min(self.block_topk, num_blocks), dim=0).indices
            selected_tokens = block_token_indices.index_select(0, selected_blocks).flatten()
            selected_tokens = selected_tokens[: self.token_topk]
        tail = visible_indices[num_blocks * self.compress_ratio :]
        selected_tokens = torch.cat([selected_tokens, tail])
        output[: selected_tokens.numel()] = selected_tokens.to(torch.int32)
        return output

    def _visible_indices(
        self,
        attention_mask: torch.Tensor | None,
        batch_idx: int,
        query_idx: int,
        query_position: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        key_positions = torch.arange(key_length, device=device)
        visible = key_positions <= query_position
        if isinstance(attention_mask, torch.Tensor):
            mask_batch_idx = min(batch_idx, attention_mask.shape[0] - 1)
            if attention_mask.ndim == 2:
                visible &= attention_mask[mask_batch_idx, :key_length].bool()
            elif attention_mask.ndim == 3:
                mask_query_idx = min(query_idx, attention_mask.shape[-2] - 1)
                mask_row = attention_mask[mask_batch_idx, mask_query_idx, :key_length]
                visible &= mask_row if mask_row.dtype == torch.bool else mask_row == 0
            elif attention_mask.ndim == 4:
                mask_query_idx = min(query_idx, attention_mask.shape[-2] - 1)
                mask_row = attention_mask[mask_batch_idx, 0, mask_query_idx, :key_length]
                visible &= mask_row if mask_row.dtype == torch.bool else mask_row == 0
        return torch.nonzero(visible, as_tuple=False).flatten()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        cos, sin = position_embeddings

        q, token_k = self.project_qk(hidden_states, cos, sin)

        rotary_dim = cos.shape[-1]
        indexer_states = torch.cat(
            [token_k, cos.to(token_k.dtype), sin.to(token_k.dtype)],
            dim=-1,
        )
        if past_key_values is not None:
            indexer_states = past_key_values.update_indexer(indexer_states, self.layer_idx)
            cache_length = past_key_values.get_seq_length(self.layer_idx)
            actual_key_length = int(cache_length.item()) if isinstance(cache_length, torch.Tensor) else cache_length
        else:
            actual_key_length = sequence_length
        indexer_states = indexer_states[:, :actual_key_length]

        raw_keys, key_cos, key_sin = torch.split(
            indexer_states,
            [self.index_head_dim, rotary_dim, rotary_dim],
            dim=-1,
        )

        output = torch.full(
            (batch_size, sequence_length, self.token_topk + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        query_start = actual_key_length - sequence_length
        for batch_idx in range(batch_size):
            for query_idx in range(sequence_length):
                query_position = query_start + query_idx
                visible_indices = self._visible_indices(
                    attention_mask,
                    batch_idx,
                    query_idx,
                    query_position,
                    actual_key_length,
                    hidden_states.device,
                )
                output[batch_idx, query_idx] = self._select_visible_row(
                    q[batch_idx, query_idx],
                    raw_keys[batch_idx],
                    key_cos[batch_idx],
                    key_sin[batch_idx],
                    visible_indices,
                )
        return output


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = Qwen4ExpQSAIndexer(config, layer_idx) if config.indexer_n_heads is not None else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
            )
            selected = torch.zeros(
                (*topk_indices.shape[:-1], key_states.shape[-2]),
                device=topk_indices.device,
                dtype=torch.int32,
            )
            selected.scatter_add_(
                -1,
                topk_indices.clamp_min(0).long(),
                (topk_indices >= 0).to(selected.dtype),
            )
            attention_mask = torch.zeros_like(selected, dtype=query_states.dtype)
            attention_mask = attention_mask.masked_fill(
                selected == 0,
                torch.finfo(query_states.dtype).min,
            ).unsqueeze(1)

        attention_implementation = self.config._attn_implementation
        if self.indexer is not None and attention_implementation not in ("eager", "sdpa"):
            logger.warning_once(
                "Qwen4-Exp QSA currently uses the eager attention reference path when %s is requested.",
                attention_implementation,
            )
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                attention_implementation,
                eager_attention_forward,
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
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), attn_weights


class Qwen4ExpExperts(Qwen3NextExperts):
    pass


class Qwen4ExpTopKRouter(Qwen3NextTopKRouter):
    pass


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    pass


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, group_size: int | None = None):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size}).")
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped_shape = (*hidden_states.shape[:-1], -1, self.group_size)
            hidden_states = hidden_states.reshape(grouped_shape)
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            hidden_states = (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (hidden_states * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpGatedResidualSimple(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpPLEGroupedNorm(
            hc_hidden_size,
            eps=config.rms_norm_eps,
            group_size=self.hidden_size,
        )
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False)

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        return mixed_input, (hyper_input, hyper_input_normed)

    def combine(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hyper_input, hyper_input_normed = residuals
        block_inject_weight = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        injection = block_output.unsqueeze(-2) * block_inject_weight.unsqueeze(-1)
        return (hyper_input.unflatten(-1, (self.hc_count, self.hidden_size)) + injection).flatten(-2)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PRIME_1 = 10007
    _CONTEXT_STATE_IDX = 2

    def __init__(self, config: Qwen4ExpTextConfig, embedding_dim: int, ple_layer_index: int = 0):
        super().__init__()
        self.ngram_embed_dim = embedding_dim
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.split_ngram_parts = config.split_ngram_parts
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = config.make_ngram_vocab_size_divisible_by
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        if eos_token_id is None:
            raise ValueError("eos_token_id must be set when Qwen4-Exp PLE layers are enabled.")
        self.eos_token_id = eos_token_id

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(config.seed),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = self._build_head_vocab_and_offsets()
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = math.ceil(total_vocab_size / self.make_ngram_vocab_size_divisible_by)
        padded_vocab_size *= self.make_ngram_vocab_size_divisible_by
        self.ngram_embedding = nn.Embedding(padded_vocab_size, self.head_dim_per_ngram)

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    def _build_layer_multipliers(self, seed: int) -> torch.Tensor:
        max_long = (1 << 63) - 1
        multiplier_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, multiplier_max // 2)
        base_seed = seed + self._PRIME_1 * self.ple_layer_index
        multipliers = []
        for index in range(self.ngram_size):
            value = (base_seed + self._SPLITMIX_GAMMA * (index + 1)) & self._MASK64
            multipliers.append(2 * (self._splitmix64(value) % half_bound) + 1)
        return torch.tensor(multipliers, dtype=torch.long)

    @staticmethod
    def _is_prime(value: int) -> bool:
        if value < 2:
            return False
        if value % 2 == 0:
            return value == 2
        for divisor in range(3, math.isqrt(value) + 1, 2):
            if value % divisor == 0:
                return False
        return True

    @classmethod
    def _find_nth_prime_after(cls, start: int, count: int) -> int:
        prime = start
        for _ in range(count):
            prime += 1
            while not cls._is_prime(prime):
                prime += 1
        return prime

    def _build_head_vocab_and_offsets(self) -> tuple[list[int], list[int], int]:
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = self._find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

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

        layer_cache = past_key_values.layers[layer_idx]
        if layer_cache.is_conv_states_initialized[self._CONTEXT_STATE_IDX]:
            cached_context = layer_cache.conv_states[self._CONTEXT_STATE_IDX]
            previous_context = cached_context[..., -context_len:].to(input_ids).clone()
            context_update = input_ids
        else:
            # LinearAttentionLayer pads newly initialized states with zeros, while Qwen4-Exp n-grams must be padded
            # with EOS. Seed the state explicitly on its first update to preserve prefill/decode parity.
            context_update = torch.cat([previous_context, input_ids], dim=-1)
        past_key_values.update_conv_state(
            context_update,
            layer_idx,
            state_idx=self._CONTEXT_STATE_IDX,
            conv_kernel_size=context_len,
        )
        return previous_context

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
    _CONV_STATE_IDX = 1

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(config, self.ple_embed_dim, ple_layer_index)
        self.short_conv_dilation = config.ngram_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.key_proj = nn.Linear(self.ple_embed_dim, self.hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(self.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            self.hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            self.hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            self.hc_hidden_size, eps=config.rms_norm_eps, group_size=self.hidden_size
        )
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            kernel_size=self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.short_conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
        )

    def _apply_norm(self, norm: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return norm(hidden_states.flatten(-2)).unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(self, hidden_states: torch.Tensor, past_key_values: Cache | None) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        if self.short_conv_state_len == 0:
            conv_input = hidden_states
        else:
            previous_state = hidden_states.new_zeros(
                hidden_states.shape[0], hidden_states.shape[1], self.short_conv_state_len
            )
            if past_key_values is not None:
                layer_cache = past_key_values.layers[self.layer_idx]
                if layer_cache.is_conv_states_initialized[self._CONV_STATE_IDX]:
                    previous_state = (
                        layer_cache.conv_states[self._CONV_STATE_IDX][..., -self.short_conv_state_len :]
                        .to(hidden_states)
                        .clone()
                    )
                past_key_values.update_conv_state(
                    hidden_states,
                    self.layer_idx,
                    state_idx=self._CONV_STATE_IDX,
                    conv_kernel_size=self.short_conv_state_len,
                )
            conv_input = torch.cat([previous_state, hidden_states], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(hidden_states.dtype),
            dilation=self.short_conv_dilation,
            groups=self.hc_hidden_size,
        )
        return F.silu(conv_output).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids, past_key_values, self.layer_idx)
        key = self.key_proj(embeddings).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query = hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        key_normed = self._apply_norm(self.norm_key, key)
        query_normed = self._apply_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        gated_value = gated_value.flatten(-2)
        return gated_value + self._short_conv(gated_value_normed, past_key_values)


class Qwen4ExpDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen4ExpGatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen4ExpAttention(config, layer_idx)
        self.mlp = Qwen4ExpSparseMoeBlock(config)
        self.ple = None
        if layer_idx + 1 in config.ple_layer_ids:
            ple_layer_index = config.ple_layer_ids.index(layer_idx + 1)
            self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_layer_index)
        self.attn_hyper_connection = Qwen4ExpGatedResidualSimple(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidualSimple(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        ple_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        if hidden_states.shape[-1] == self.hidden_size:
            hidden_states = hidden_states.repeat(1, 1, self.hc_count)

        if self.ple is not None:
            if ple_input_ids is None:
                raise ValueError("input_ids are required when Qwen4-Exp PLE layers are enabled.")
            hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values)

        hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )
        hidden_states = self.attn_hyper_connection.combine(hidden_states, residual)

        hidden_states, residual = self.mlp_hyper_connection.mix(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return self.mlp_hyper_connection.combine(hidden_states, residual)


class Qwen4ExpPreTrainedModel(Qwen3_5MoePreTrainedModel):
    config: Qwen4ExpConfig
    _no_split_modules = ["Qwen4ExpDecoderLayer", "Qwen4ExpVisionBlock"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen4ExpTopKRouter, index=0),
        "attentions": Qwen4ExpAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen4ExpPLEGroupedNorm):
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
        self.hyper_connection_mixer = Qwen4ExpGatedResidualSimple(config, use_combine=False)
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
            Original token ids used to construct positional lexical embeddings. This is only needed when PLE is
            enabled and `inputs_embeds` are passed instead of `input_ids`.
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

        raw_attention_mask = attention_mask
        if (
            self.config.ple_layer_ids
            and ple_input_ids is not None
            and isinstance(raw_attention_mask, torch.Tensor)
            and raw_attention_mask.ndim == 2
        ):
            current_attention_mask = raw_attention_mask[:, -ple_input_ids.shape[1] :]
            eos_token_id = self.config.eos_token_id
            eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            ple_input_ids = torch.where(current_attention_mask.bool(), ple_input_ids, eos_token_id)

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

        output_hidden_states = kwargs.pop("output_hidden_states", self.config.output_hidden_states)
        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            block_type = self.config.layers_block_type[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[block_type],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                ple_input_ids=ple_input_ids,
                use_cache=use_cache,
                **kwargs,
            )
            if output_hidden_states:
                layer_hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
                all_hidden_states += (layer_hidden_states,)

        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return Qwen4ExpModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class Qwen4ExpModel(Qwen3_5MoeModel):
    _no_split_modules = ["Qwen4ExpDecoderLayer", "Qwen4ExpVisionBlock"]

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
            Original token ids used to construct positional lexical embeddings when inputs_embeds are passed
            instead of input_ids. When input_ids are provided, they are always used for PLE.
        """
        if input_ids is not None:
            ple_input_ids = input_ids
        elif self.config.text_config.ple_layer_ids and ple_input_ids is None:
            raise ValueError(
                "ple_input_ids must be provided when Qwen4-Exp PLE layers are enabled and inputs_embeds are used."
            )
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
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

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
    ) -> MoeCausalLMOutputWithPast:
        r"""
        ple_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Original token ids used to construct positional lexical embeddings when inputs_embeds are passed
            instead of input_ids.
        """
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
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
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Qwen4ExpForTokenClassification(GenericForTokenClassification, Qwen4ExpPreTrainedModel):
    config: Qwen4ExpConfig


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
            Original token ids used to construct positional lexical embeddings when inputs_embeds are passed
            instead of input_ids.
        """
        super().forward(
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
            ple_input_ids=ple_input_ids,
            **kwargs,
        )
        outputs = self.model(  # noqa: F841
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
            **kwargs,
        )


class Qwen4ExpTextForSequenceClassification(GenericForSequenceClassification, Qwen4ExpPreTrainedModel):
    config: Qwen4ExpTextConfig
    input_modalities = ("text",)


class Qwen4ExpForSequenceClassification(GenericForSequenceClassification, Qwen4ExpPreTrainedModel):
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
    ) -> SequenceClassifierOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=ple_input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
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
    "Qwen4ExpTextForSequenceClassification",
    "Qwen4ExpForSequenceClassification",
    "Qwen4ExpForTokenClassification",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpPreTrainedModel",
]
