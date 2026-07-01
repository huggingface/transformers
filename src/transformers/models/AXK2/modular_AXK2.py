# Copyright 2025 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch A.X-K2 model (modular)."""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GenericForSequenceClassification, GenericForTokenClassification
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.generic import is_flash_attention_requested
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    rotate_half,
)
from ..mixtral.modeling_mixtral import MixtralExperts
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from .configuration_AXK2 import AXK2Config


logger = logging.get_logger(__name__)


class AXK2RMSNorm(LlamaRMSNorm):
    pass


class AXK2GatedRMSNorm(nn.Module):
    """RMSNorm wrapped with a low-rank input-dependent gate (Megatron GatedNormWrapper).

    forward(x):
        y = RMSNorm(x)
        gate = W_up(silu(W_down(y)))
        return y * sigmoid(gate.float()).to(y.dtype)
    """

    def __init__(self, hidden_size: int, rank: int, eps: float):
        super().__init__()
        self.norm = AXK2RMSNorm(hidden_size, eps=eps)
        self.W_down = nn.Linear(hidden_size, rank, bias=False)
        self.W_up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        raw_gate = self.W_up(F.silu(self.W_down(y)))
        return (y * torch.sigmoid(raw_gate.float())).to(y.dtype)


class AXK2RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class AXK2MLP(Qwen2MoeMLP):
    pass


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies interleaved Rotary Position Embedding to the query and key tensors.

    Unlike the standard RoPE which splits the hidden dim into two halves, this variant interleaves
    even/odd dimensions before applying the rotation, matching the weight layout used during pretraining.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Unused. Kept for API compatibility.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            Dimension along which to unsqueeze cos/sin for broadcasting.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_noninterleaved(x, cos, sin, unsqueeze_dim=1):
    """Applies standard (non-interleaved) Rotary Position Embedding to a single tensor.

    Used by the SGA (Sparse Gated Attention) indexer, whose weights are stored in non-interleaved
    (NeoX/Llama) layout — unlike the main MLA attention, which uses the interleaved variant. `cos`/`sin`
    are already gathered for the current positions (shape `[batch, seq, dim]`), matching the convention
    of the model-level rotary embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class AXK2Indexer(nn.Module):
    """SGA (Sparse Gated Attention) lightning indexer.

    Computes a cheap relevance score between each query and every key, then returns the indices of the
    `index_topk` highest-scoring positions per query. Those indices drive a sparse attention mask so the
    main MLA attention only attends to the selected key/value positions.

    Score per (query s, key t):
        index_score[b, s, t] = sum_h ReLU(q[b, h, s, :] · k[b, t, :] * scale) * weights[b, s, h]

    The indexer uses non-interleaved (NeoX/Llama) RoPE on the rotary slice of its head dim, separate from
    the interleaved RoPE used by the main attention. It keeps its own key cache for autoregressive decode,
    independent of the model's `Cache` (which is sized for the main attention layers only).

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py
    """

    def __init__(self, config: AXK2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        self.softmax_scale = self.head_dim ** (-0.5)

        # Query bottleneck expansion: [q_lora_rank] -> [n_heads * head_dim]
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        # Key projection: [hidden_size] -> [head_dim]
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        # Key normalization (LayerNorm with bias, not RMSNorm)
        self.k_norm = nn.LayerNorm(self.head_dim)
        # Per-head weight projection: [hidden_size] -> [n_heads]
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)

        # Indexer keeps its own key cache (the model `Cache` only holds the main attention layers).
        self._cached_keys: torch.Tensor | None = None

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.LongTensor:
        bsz, seq_len, _ = hidden_states.size()

        # Queries: [B, H, S, D], rotary applied to the PE slice only
        q = self.wq_b(q_compressed).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_pe, q_nope = q[..., : self.qk_rope_head_dim], q[..., self.qk_rope_head_dim :]
        q_pe = apply_rotary_pos_emb_noninterleaved(q_pe, cos, sin)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Keys: [B, 1, S, D] (shared across heads), rotary applied to the PE slice only
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(1)
        k_pe, k_nope = k[..., : self.qk_rope_head_dim], k[..., self.qk_rope_head_dim :]
        k_pe = apply_rotary_pos_emb_noninterleaved(k_pe, cos, sin)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # Key cache for autoregressive decode; reset on prefill (new prompt) to drop stale keys.
        if seq_len > 1:
            self._cached_keys = None
        if use_cache:
            if self._cached_keys is not None:
                k = torch.cat([self._cached_keys, k], dim=2)
            self._cached_keys = k

        # Per-head weights [B, S, H], scaled by n_heads^-0.5
        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** (-0.5))
        # Per-head QK scores [B, H, S, T] with ReLU, reduced over heads -> [B, S, T]
        scores = F.relu(torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.softmax_scale)
        index_scores = torch.einsum("bhst,bsh->bst", scores, weights)

        # Causal masking so the indexer cannot select future tokens.
        if attention_mask is not None:
            indexer_mask = attention_mask[:, 0, :, :] if attention_mask.dim() == 4 else attention_mask
            index_scores = index_scores + indexer_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices


class AXK2TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class AXK2NaiveMoe(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_local_experts
        self.intermediate_dim = config.moe_intermediate_size


class AXK2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = AXK2NaiveMoe(config)
        self.gate = AXK2TopkRouter(config)
        self.shared_experts = AXK2MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        if self.n_group is None or self.topk_group is None:
            topk_indices = torch.topk(router_logits_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        else:
            group_scores = (
                router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(-1, self.n_routed_experts)
            )
            scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
            topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class AXK2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AXK2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        self.use_output_gate = getattr(config, "attention_output_gate", False)
        # When the output gate is fused into q_b_proj (vLLM-compatible), the input is doubled
        # ([q_a post-norm | q_a pre-norm]) and each head outputs [q (qk_head_dim) | gate (v_head_dim)].
        self.attn_gate_fused = self.use_output_gate and getattr(config, "attn_gate_fused", False)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = AXK2RMSNorm(config.q_lora_rank)
            if self.attn_gate_fused:
                self.q_b_proj = nn.Linear(
                    2 * config.q_lora_rank,
                    self.num_heads * (self.qk_head_dim + self.v_head_dim),
                    bias=False,
                )
            else:
                self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = AXK2RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Separate output-gate projection for non-fused checkpoints; the fused variant
        # (attn_gate_fused) carries the gate inside q_b_proj instead.
        if self.use_output_gate and not self.attn_gate_fused:
            gate_in = self.q_lora_rank if self.q_lora_rank is not None else config.hidden_size
            self.linear_gate = nn.Linear(gate_in, self.num_heads * self.v_head_dim, bias=False)
        else:
            self.linear_gate = None

        # SGA (Sparse Gated Attention) indexer: present only when the config enables it.
        if getattr(config, "index_topk", None):
            self.indexer = AXK2Indexer(config, layer_idx=layer_idx)
        else:
            self.indexer = None

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        gate = None
        if self.q_lora_rank is None:
            q_compressed = hidden_states
            q_states = self.q_proj(hidden_states)
        else:
            q_compressed = self.q_a_proj(hidden_states)
            q_post = self.q_a_layernorm(q_compressed)
            if self.attn_gate_fused:
                # Fused output gate: doubled input [q_post | q_pre] -> per-head [q | gate].
                qg = self.q_b_proj(torch.cat([q_post, q_compressed], dim=-1))
                qg = qg.view(batch_size, seq_length, self.num_heads, self.qk_head_dim + self.v_head_dim)
                q_states, gate = torch.split(qg, [self.qk_head_dim, self.v_head_dim], dim=-1)
                q_states = q_states.reshape(batch_size, seq_length, self.num_heads * self.qk_head_dim)
                gate = gate.reshape(batch_size, seq_length, self.num_heads * self.v_head_dim)
            else:
                q_states = self.q_b_proj(q_post)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # SGA: keep only the indexer-selected key/value positions by adding a sparse (-inf) mask.
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_compressed,
                cos,
                sin,
                attention_mask=attention_mask,
                use_cache=past_key_values is not None,
            )
            total_kv_len = key_states.shape[-2]
            index_mask = torch.full(
                (batch_size, seq_length, total_kv_len),
                float("-inf"),
                device=hidden_states.device,
                dtype=query_states.dtype,
            )
            index_mask.scatter_(-1, topk_indices.clamp(0, total_kv_len - 1), 0.0)
            index_mask = index_mask.unsqueeze(1)
            attention_mask = index_mask if attention_mask is None else index_mask + attention_mask

        use_flash = self.indexer is None and is_flash_attention_requested(self.config)
        if use_flash and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # SGA relies on an explicit additive mask, so it always runs through eager attention.
        if self.indexer is not None:
            attention_interface: Callable = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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

        if use_flash and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.linear_gate is not None:
            gate = self.linear_gate(q_compressed)
        if gate is not None:
            attn_output = (attn_output * torch.sigmoid(gate.float())).to(attn_output.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AXK2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: AXK2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = AXK2Attention(config=config, layer_idx=layer_idx)

        self.is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if self.is_moe_layer:
            self.mlp = AXK2MoE(config)
        else:
            self.mlp = AXK2MLP(config)

        use_gated = getattr(config, "gated_norm", False)
        gate_rank = getattr(config, "gated_norm_rank", 16)

        if use_gated:
            self.input_layernorm = AXK2GatedRMSNorm(config.hidden_size, rank=gate_rank, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = AXK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if use_gated and self.is_moe_layer:
            self.post_attention_layernorm = AXK2GatedRMSNorm(
                config.hidden_size, rank=gate_rank, eps=config.rms_norm_eps
            )
        else:
            self.post_attention_layernorm = AXK2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class AXK2PreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, AXK2TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, AXK2NaiveMoe):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class AXK2Model(LlamaModel):
    pass


class AXK2ForCausalLM(LlamaForCausalLM):
    pass


class AXK2ForSequenceClassification(GenericForSequenceClassification, AXK2PreTrainedModel):
    pass


class AXK2ForTokenClassification(GenericForTokenClassification, AXK2PreTrainedModel):
    pass


__all__ = [
    "AXK2PreTrainedModel",
    "AXK2Model",
    "AXK2ForCausalLM",
    "AXK2ForSequenceClassification",
    "AXK2ForTokenClassification",
]
