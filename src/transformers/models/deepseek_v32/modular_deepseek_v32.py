# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""DeepSeek-V3.2-Exp: DeepSeek-V3 plus DeepSeek Sparse Attention (DSA).

This is DeepSeek-V3 with a lightning indexer added to each attention layer: the indexer scores every
query against the cached keys and keeps the top-`index_topk` tokens, which become an additive sparse
mask folded into the MLA attention mask. Everything else (MoE, MLA projections, RoPE, the decoder /
model / causal-LM scaffolding) is inherited unchanged from DeepSeek-V3.

The cross-layer top-k *sharing* variant is a GLM-MoE-DSA innovation and lives in that model, which
inherits from this one (see `models/glm_moe_dsa/modular_glm_moe_dsa.py`).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.output_capturing import OutputRecorder
from ..axk1.modeling_axk1 import AXK1Attention
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V3.2-Exp")
@strict
class DeepseekV32Config(Glm4MoeLiteConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    output_indexer_scores (`bool`, *optional*, defaults to `False`):
        Whether or not to return the DSA indexer scores of every layer. This is required to compute the indexer
        distillation loss (see `indexer_kl_loss_func`) and is the only case in which the indexer builds an autograd
        graph: top-k selection has no gradient, so the indexer is never trained through the attention output.
    indexer_loss_coef (`float`, *optional*, defaults to 1.0):
        Coefficient of the indexer distillation loss added to the language modeling loss when
        `output_indexer_scores=True`. The indexer parameters receive gradients only from this loss and the rest of the
        model only from the language modeling loss, so it acts as a learning-rate multiplier for the indexer.
    dense_indexer (`bool`, *optional*, defaults to `False`):
        Whether to ignore the indexer's top-k selection and run dense attention, as in the dense warm-up stage of DSA
        training.

    ```python
    >>> from transformers import DeepseekV32Config, DeepseekV32Model

    >>> # Initializing a DeepSeek-V3.2 configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False
    head_dim: int = 64
    first_k_dense_replace: int = 3
    output_indexer_scores: bool = False
    indexer_loss_coef: float = 1.0
    dense_indexer: bool = False
    pretraining_tp = AttributeError()
    rope_interleave = AttributeError()
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the inherited (Llama) rotary
        # embedding reads `config.head_dim` and then computes the right frequencies with no override needed.
        self.head_dim = self.qk_rope_head_dim
        # MLP layer types: the first `first_k_dense_replace` layers are dense, the rest are MoE.
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense)
        # Every layer is DSA — drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        # BC: re-route `num_experts` to `n_routed_experts`
        if (num_experts := kwargs.get("num_experts")) is not None:
            self.n_routed_experts = num_experts

        super().__post_init__(**kwargs)


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32Indexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the main MLA attention, and returns
    the top-k token indices, which the attention scatters into a sparse mask. Since top-k selection has no gradient,
    the indexer is not trained through the attention output but with its own distillation loss on its scores
    (`indexer_kl_loss_func`). On request, it returns the scores along with their target, the attention distribution it
    is distilled towards, recomputed from the attention's queries and keys so that it is never kept for backward.

    **Cache strategy**: the indexer key cache lives on the per-layer `DynamicIndexedLayer` (or the
    `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config: "DeepseekV32Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,  # Kept for BC
        past_key_values: Cache | None = None,
        output_scores: bool = False,
        query_states: torch.Tensor | None = None,
        key_states: torch.Tensor | None = None,
        scaling: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform)
        and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal (dot products
        are preserved: Hq·Hk = q·k), and FP8 quantization is a precision optimization, we skip both and compute
        scores directly in bf16/fp32.

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.
            output_scores: Whether to also return the indexer scores and their distillation target, used for the
                indexer loss (see `indexer_kl_loss_func`). This is the only case in which the indexer builds an
                autograd graph.
            query_states: Queries of the main attention, `[B, num_heads, S, qk_head_dim]`, only needed with
                `output_scores` to compute the target.
            key_states: Keys of the main attention, `[B, num_heads, T, qk_head_dim]`, only needed with `output_scores`.
            scaling: Softmax scale of the main attention, only needed with `output_scores`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]`: the `int32` top-k token indices of shape
                `[B, S, topk]` (the eager / SDPA paths turn these into an additive sparse mask; the `flash-mla` kernel
                consumes them directly), then, if `output_scores=True`, the `float32` indexer scores and the detached
                `float32` attention distribution they are distilled towards (see `attention_target`). Both cover the
                candidate keys of every query: the `topk` selected keys, in the order of the indices, with shape
                `[B, S, topk]`, or every key with shape `[B, S, T]` in the dense warm-up stage
                (`config.dense_indexer`). `None, None` otherwise.
        """
        # The indexer is only trained through `indexer_kl_loss_func` on its scores, never through the attention
        # output, so an autograd graph is only built when the scores are requested.
        with torch.set_grad_enabled(torch.is_grad_enabled() and output_scores):
            # The inputs come from the main model, which is trained by the language modeling loss only: detach them so
            # the indexer loss reaches no parameter but the indexer's.
            hidden_states, q_resid = hidden_states.detach(), q_resid.detach()
            batch_size, seq_len, _ = hidden_states.shape
            cos, sin = position_embeddings
            q = self.wq_b(q_resid)  # [B, S, H*D]
            q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
            q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

            k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
            k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

            # The indexer uses NON-interleaved (half-split) RoPE — unlike the main MLA attention
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
            q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
            k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

            if past_key_values is not None:
                k = past_key_values.update_indexer(k, self.layer_idx)

            scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
            scores = F.relu(scores)

            # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
            weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (
                self.n_heads**-0.5
            )
            index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

            # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
            # (masked with the finite dtype minimum rather than `-inf` so the scores can go through a log-softmax)
            if attention_mask.dtype == torch.bool:
                index_scores = index_scores.masked_fill(~attention_mask, torch.finfo(index_scores.dtype).min)
            else:
                index_scores = index_scores + attention_mask

            topk = min(self.index_topk, index_scores.shape[-1])
            topk_indices = index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]

            if not output_scores:
                return topk_indices, None, None
            # Sparse training stage: attention only sees the selected keys, so both distributions are restricted to
            # them (the KL is computed over S_t). In the dense warm-up stage they cover every visible key.
            candidates = None if self.config.dense_indexer else topk_indices.long()
            if candidates is not None:
                index_scores = index_scores.gather(-1, candidates)
            target = self.attention_target(query_states, key_states, attention_mask, scaling, candidates)
            return topk_indices, index_scores, target

    @torch.no_grad()
    def attention_target(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        candidates: torch.Tensor | None = None,
        heads_per_chunk: int = 16,
    ) -> torch.Tensor:
        """
        Computes the distillation target of the indexer: the distribution of the main attention over the candidate
        keys, summed over heads and normalized (see `indexer_kl_loss_func`). It is recomputed from the queries and keys,
        `heads_per_chunk` heads at a time, so the attention probabilities are never kept for backward nor all
        materialized at once, and the main attention can use any implementation.

        Args:
            query_states: Queries of the main attention, `[B, num_heads, S, qk_head_dim]`.
            key_states: Keys of the main attention, `[B, num_heads, T, qk_head_dim]`.
            attention_mask: Causal mask `[B, S, T]`, boolean or additive.
            scaling: Softmax scale of the main attention.
            candidates: The selected keys `[B, S, topk]`, or `None` to attend to every visible key (dense warm-up).

        Returns:
            `torch.Tensor`: the `float32` target of shape `[B, S, topk]`, or `[B, S, T]` if `candidates` is `None`.
        """
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=query_states.dtype).masked_fill(
                ~attention_mask, torch.finfo(query_states.dtype).min
            )
        attention_mask = attention_mask.unsqueeze(1)  # [B, 1, S, T]
        if candidates is not None:
            candidates = candidates.unsqueeze(1)  # [B, 1, S, topk]

        target = None
        for query_chunk, key_chunk in zip(
            query_states.split(heads_per_chunk, dim=1), key_states.split(heads_per_chunk, dim=1)
        ):
            logits = torch.matmul(query_chunk, key_chunk.transpose(-1, -2)) * scaling + attention_mask
            if candidates is not None:
                logits = logits.gather(-1, candidates.expand(-1, logits.shape[1], -1, -1))
            probs = F.softmax(logits, dim=-1, dtype=torch.float32).sum(dim=1)
            target = probs if target is None else target + probs
        return target / query_states.shape[1]


class DeepseekV32Attention(AXK1Attention):
    """
    DeepSeek-V3 MLA, with a DSA indexer whose top-k sparse mask is folded into the attention mask.
    Qlora rank formulation is dropped as it is never used in released models.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # Both latents are viewed as single-head, 4D tensors, as expected by `expand_kv`
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # Sparse-attention models cache the expanded K/V, not the compressed latents. TODO (remi-or): fix this with topk
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`. Its scores and
        # their distillation target are only returned (and recorded, see `output_indexer_scores`) for the indexer loss.
        topk_indices, _, _ = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask[:, 0, :, :],
            position_ids,  # Kept for BC
            past_key_values=past_key_values,
            output_scores=kwargs.get("output_indexer_scores", self.config.output_indexer_scores),
            query_states=query_states,
            key_states=key_states,
            scaling=self.scaling,
        )  # [B, S, topk], then the scores and their target (`[B, S, topk]` each) or `None, None`

        sparse_indices = None
        if self.config.dense_indexer:
            pass  # Dense attention: every visible key is attended to, the indexer only runs to be distilled from it
        elif self.config._attn_implementation in ("eager", "sdpa"):
            # Boolean mask: `True` at keys *not* selected by the indexer (to be masked out).
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )  # [B, 1, S, T]; True = masked

            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & ~index_mask
            else:
                attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

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
            indices=sparse_indices,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": DeepseekV32DecoderLayer,
        "attentions": DeepseekV32Attention,
        "indexer_scores": OutputRecorder(DeepseekV32Indexer, index=1),
        "indexer_targets": OutputRecorder(DeepseekV32Indexer, index=2),
    }
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False


@dataclass
class DeepseekV32ModelOutputWithPast(BaseModelOutputWithPast):
    """
    indexer_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_scores=True` is passed or when `config.output_indexer_scores=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`, or
        `(batch_size, sequence_length, kv_sequence_length)` in the dense warm-up stage (`config.dense_indexer`).

        Scores of the DSA indexer over the candidate keys of every query: the `index_topk` selected keys, in the order
        of the top-k indices, or every key in the dense warm-up stage. Used to compute the indexer loss.
    indexer_targets (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_targets=True` is passed):
        Tuple of `torch.FloatTensor` (one for each layer) of the same shape as `indexer_scores`.

        Distillation targets of the indexer scores: the distribution of the main attention over the same candidate
        keys, summed over heads, normalized and detached. Used to compute the indexer loss.
    """

    indexer_scores: tuple[torch.FloatTensor, ...] | None = None
    indexer_targets: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class DeepseekV32CausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    indexer_loss (`torch.FloatTensor`, *optional*, returned when `output_indexer_scores=True` is passed or when `config.output_indexer_scores=True`):
        Distillation loss of the DSA indexer: KL divergence between the attention distribution (summed over heads,
        normalized and detached) and the softmax of the indexer scores, see `indexer_kl_loss_func`. Only the indexer
        parameters receive gradients from it.
    indexer_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_scores=True` is passed or when `config.output_indexer_scores=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`, or
        `(batch_size, sequence_length, kv_sequence_length)` in the dense warm-up stage (`config.dense_indexer`).

        Scores of the DSA indexer over the candidate keys of every query: the `index_topk` selected keys, in the order
        of the top-k indices, or every key in the dense warm-up stage.
    indexer_targets (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_scores=True` is passed or when `config.output_indexer_scores=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of the same shape as `indexer_scores`.

        Distillation targets of the indexer scores: the distribution of the main attention over the same candidate
        keys, summed over heads, normalized and detached.
    """

    indexer_loss: torch.FloatTensor | None = None
    indexer_scores: tuple[torch.FloatTensor, ...] | None = None
    indexer_targets: tuple[torch.FloatTensor, ...] | None = None


class DeepseekV32Model(DeepseekV3Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DeepseekV32ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for causality in the indexer
            }
            causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return DeepseekV32ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def indexer_kl_loss_func(
    indexer_scores: tuple[torch.Tensor, ...] | None,
    indexer_targets: tuple[torch.Tensor, ...] | None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    Computes the DeepSeek Sparse Attention (DSA) indexer loss, see the DeepSeek-V3.2 technical report
    (https://huggingface.co/papers/2512.02556), section 2.2.

    Top-k selection has no gradient, so the indexer cannot learn from the language modeling loss. It is instead
    distilled from the main attention: for every query, the target is the attention distribution summed over heads and
    normalized, and the loss is its KL divergence to the softmax of the indexer scores. The target and the inputs of
    the indexer are detached, so only the indexer receives gradients from this loss, and the indexer parameters are not
    on the language modeling path, so the two objectives train disjoint parameters. In the sparse training stage both
    distributions only cover the selected keys (the KL is computed over S_t); in the dense warm-up stage
    (`config.dense_indexer=True`) they cover every visible key.

    Args:
        indexer_scores (`tuple[torch.Tensor, ...]`):
            One tensor per layer of shape `[batch_size, sequence_length, num_candidates]`, the indexer scores over the
            candidate keys, as returned when `output_indexer_scores=True`.
        indexer_targets (`tuple[torch.Tensor, ...]`):
            One tensor per layer of the same shape, the attention distribution over the same keys, as returned when
            `output_indexer_targets=True`.
        attention_mask (`torch.Tensor`, *optional*):
            The 2D padding mask of shape `[batch_size, kv_sequence_length]`; padded query positions are excluded.

    Returns:
        `torch.Tensor`: the scalar `float32` loss, averaged over layers and (non-padded) query positions.
    """
    if not indexer_scores or not indexer_targets or not indexer_scores[0].is_floating_point():
        raise ValueError(
            "The indexer loss needs the indexer scores and their targets. Make sure `output_indexer_scores=True` and "
            "that the indexer of this model returns its scores."
        )

    loss = None
    for scores, target in zip(indexer_scores, indexer_targets):
        # The scores are finite (non-candidate keys are excluded, not masked to -inf), so `p * log q` is 0 wherever p is 0
        kl = (torch.xlogy(target, target) - target * F.log_softmax(scores, dim=-1)).sum(dim=-1)  # [B, S]
        if attention_mask is not None:
            # The queries are the last `sequence_length` positions of the (possibly longer) key axis
            valid_queries = attention_mask[:, -kl.shape[1] :].to(kl.dtype)
            layer_loss = (kl * valid_queries).sum() / valid_queries.sum().clamp_min(1)
        else:
            layer_loss = kl.mean()
        loss = layer_loss if loss is None else loss + layer_loss
    return loss / len(indexer_scores)


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DeepseekV32CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV32ForCausalLM

        >>> model = DeepseekV32ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
        >>> tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```

        To train the DSA indexer, pass `output_indexer_scores=True` (or set it in the config): the indexer
        distillation loss (`indexer_kl_loss_func`) is returned as `indexer_loss` and, when `labels` are given, added to
        `loss` scaled by `config.indexer_loss_coef`."""
        output_indexer_scores = kwargs.get("output_indexer_scores", self.config.output_indexer_scores)
        if output_indexer_scores:
            # The loss needs the distillation targets of the scores, which are recorded alongside them
            kwargs["output_indexer_targets"] = True

        outputs: DeepseekV32ModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        indexer_loss = None
        if output_indexer_scores:
            indexer_loss = indexer_kl_loss_func(outputs.indexer_scores, outputs.indexer_targets, attention_mask)
            if labels is not None:
                loss = loss + self.config.indexer_loss_coef * indexer_loss.to(loss.device)

        return DeepseekV32CausalLMOutputWithPast(
            loss=loss,
            indexer_loss=indexer_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            indexer_scores=outputs.indexer_scores,
            indexer_targets=outputs.indexer_targets,
        )


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
