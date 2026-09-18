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
model / causal-LM scaffolding) is based on DeepSeek-V3.

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
from ...utils.generic import maybe_autocast
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
    output_indexer_loss (`bool`, *optional*, defaults to `False`):
        Whether [`DeepseekV32ForCausalLM`] computes the indexer's distillation loss from the indexer scores and
        attention targets recorded in every layer, and adds it to `loss`. Only the indexer receives gradients from it.

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
    keys_to_ignore_at_inference = ["past_key_values", "indexer_loss", "indexer_scores", "indexer_targets"]

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
    output_indexer_loss: bool = False
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

    The Indexer has its own lightweight projections (wq_b, wk) separate from the main MLA attention, and scores
    every query against the cached keys to select the top-k tokens the attention may attend to.

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the `int32` top-k token indices of shape `[B, S, topk]` and their
                `float32` index scores. The eager / SDPA paths turn the indices into an additive sparse mask; the
                `flash-mla` kernel consumes them directly. The scores are the input of the indexer's distillation
                loss; they are `-inf` at masked keys, which early queries select when fewer than `topk` are visible.
        """
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

        # Score in FP32 like the reference kernel, also under autocast. Flatten queries and heads to avoid broadcasting a
        # copy of the keys for every query position.
        with maybe_autocast(device_type=hidden_states.device.type, enabled=False):
            scores = torch.matmul(q.flatten(1, 2).float(), k.transpose(-1, -2).float()) * self.softmax_scale
            scores = F.relu(scores).view(batch_size, seq_len, self.n_heads, k.shape[1])
            # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
            weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (
                self.n_heads**-0.5
            )
            index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        topk_scores, topk_indices = index_scores.topk(topk, dim=-1)  # [B, S, topk]
        return topk_indices.to(torch.int32), topk_scores


@torch.no_grad()
def indexer_attention_target(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    topk_indices: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Attention distribution over the keys the indexer selected, averaged over heads: the target its scores are
    distilled towards. It is recomputed from the attention inputs, in head chunks, so the attention itself can run
    on any backend. Queries without a visible key get an all-zero row and thus add nothing to the loss.

    Returns:
        `torch.Tensor` of shape `[B, S, topk]` in `float32`.
    """
    if attention_mask.dtype == torch.bool:
        attention_mask = torch.zeros_like(attention_mask, dtype=query_states.dtype).masked_fill(
            ~attention_mask, torch.finfo(query_states.dtype).min
        )
    candidates = topk_indices.long().unsqueeze(1)  # [B, 1, S, topk]
    target = query_states.new_zeros(topk_indices.shape, dtype=torch.float32)
    for query_chunk, key_chunk in zip(query_states.split(16, dim=1), key_states.split(16, dim=1)):
        logits = torch.matmul(query_chunk, key_chunk.transpose(-1, -2)) * scaling + attention_mask
        logits = logits.gather(-1, candidates.expand(-1, logits.shape[1], -1, -1))
        # Fully masked queries must stay finite too; their rows are zeroed below.
        logits = logits.clamp_min(torch.finfo(logits.dtype).min)
        target += F.softmax(logits, dim=-1, dtype=torch.float32).sum(dim=1)
    target /= query_states.shape[1]
    visible = (attention_mask > torch.finfo(attention_mask.dtype).min).any(dim=-1).any(dim=1)  # [B, S]
    return target.masked_fill(~visible.unsqueeze(-1), 0.0)


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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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

        # Cache read / write is performed while latent KV is still compressed
        if past_key_values is not None:
            k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            attention_mask[:, 0, :, :],
            position_ids,  # Kept for BC
            past_key_values=past_key_values,
        )[0]  # [B, S, topk]

        # Recorded, with the indexer scores, as the target of the indexer's distillation loss
        indexer_target = None
        if kwargs.get("output_indexer_targets", False):
            indexer_target = indexer_attention_target(
                query_states, key_states, attention_mask, topk_indices, self.scaling
            )

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
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
        return attn_output, attn_weights, indexer_target


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_record_outputs = {
        "hidden_states": DeepseekV32DecoderLayer,
        "attentions": DeepseekV32Attention,
        "indexer_scores": OutputRecorder(DeepseekV32Indexer, index=1),
        "indexer_targets": OutputRecorder(DeepseekV32Attention, index=2),
    }


@auto_docstring(
    custom_intro="""
    Base class for DeepSeek-V3.2 model outputs, with the recorded inputs of the indexer's distillation loss.
    """
)
@dataclass
class DeepseekV32ModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    indexer_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_scores=True` is passed):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`.

        The indexer's scores of the keys it selected, the input of its distillation loss.
    indexer_targets (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_targets=True` is passed):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`.

        The attention distribution over the selected keys, averaged over heads, the target of the indexer's
        distillation loss.
    """

    indexer_scores: tuple[torch.FloatTensor, ...] | None = None
    indexer_targets: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Base class for DeepSeek-V3.2 causal language model outputs, with the indexer's distillation loss.
    """
)
@dataclass
class DeepseekV32CausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    indexer_loss (`torch.FloatTensor`, *optional*, returned when `output_indexer_loss=True` is passed):
        Distillation loss of the indexer, averaged over layers and queries. It is added to `loss` when `labels` are
        provided.
    indexer_scores (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_loss=True` or
        `output_indexer_scores=True` is passed):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`.

        The indexer's scores of the keys it selected, the input of its distillation loss.
    indexer_targets (`tuple(torch.FloatTensor)`, *optional*, returned when `output_indexer_loss=True` or
        `output_indexer_targets=True` is passed):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, index_topk)`.

        The attention distribution over the selected keys, averaged over heads, the target of the indexer's
        distillation loss.
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


def indexer_kl_loss(
    indexer_scores: tuple[torch.Tensor, ...],
    indexer_targets: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    r"""
    Distillation loss of the DSA indexer, from section 2.1.1 of the [DeepSeek-V3.2 report](https://arxiv.org/abs/2512.02556):
    the KL divergence from each layer's attention distribution over the selected keys (`indexer_targets`) to the
    indexer's distribution over them (`softmax(indexer_scores)`), summed over layers and queries.

    The sum is divided by the number of layers and by `num_items_in_batch` when given, as the language modeling loss
    is under gradient accumulation, or else by the number of queries that count: those of non-padding tokens when
    `attention_mask` is 2D, otherwise those with a visible key. With `index_topk` at least the sequence length,
    every visible key is selected: the dense warm-up stage.
    """
    query_mask = None
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        query_mask = attention_mask[:, -indexer_targets[0].shape[1] :].bool()

    loss = None
    for scores, target in zip(indexer_scores, indexer_targets):
        # Early queries select masked keys when fewer than topk are visible. Keep log-softmax finite.
        log_probs = F.log_softmax(scores.clamp_min(torch.finfo(scores.dtype).min), dim=-1, dtype=torch.float32)
        kl = F.kl_div(log_probs, target, reduction="none").sum(dim=-1)  # [B, S]
        if query_mask is not None:
            kl = kl.masked_fill(~query_mask.to(kl.device), 0.0)
        loss = kl.sum() if loss is None else loss + kl.sum().to(loss.device)

    if num_items_in_batch is None:
        if query_mask is not None:
            num_items_in_batch = query_mask.sum()
        else:
            num_items_in_batch = (indexer_targets[0].sum(dim=-1) > 0).sum()
    normalizer = torch.as_tensor(num_items_in_batch, device=loss.device).clamp_min(1)
    return loss / (len(indexer_scores) * normalizer)


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
        output_indexer_loss: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DeepseekV32CausalLMOutputWithPast:
        r"""
        output_indexer_loss (`bool`, *optional*):
            Whether to compute the indexer's distillation loss from the indexer scores and attention targets recorded
            in every layer, and add it to `loss`. Defaults to `config.output_indexer_loss`. Only the indexer receives
            gradients from this loss.

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
        ```"""
        output_indexer_loss = (
            output_indexer_loss if output_indexer_loss is not None else self.config.output_indexer_loss
        )
        if output_indexer_loss:
            kwargs["output_indexer_scores"] = kwargs["output_indexer_targets"] = True

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
        if output_indexer_loss:
            indexer_loss = indexer_kl_loss(
                outputs.indexer_scores, outputs.indexer_targets, attention_mask, kwargs.get("num_items_in_batch")
            )
            if self.training and torch.is_grad_enabled() and not indexer_loss.requires_grad:
                logger.warning_once(
                    "The indexer loss carries no gradient: either the indexer's parameters are frozen, or gradient "
                    "checkpointing runs with `use_reentrant=True`, which keeps the recorded indexer scores out of the "
                    "autograd graph. Use `use_reentrant=False` (the default) to train the indexer."
                )
            if loss is not None:
                loss = loss + indexer_loss.to(loss.device)

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
