# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HYV4 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteModel,
    Glm4MoeLitePreTrainedModel,
    Glm4MoeLiteRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding


@auto_docstring(custom_intro="HYV4")
@strict
class HYV4Config(PreTrainedConfig):
    r"""
    HYV4 is a mixture-of-experts causal language model using Multi-head Latent Attention (MLA),
    DeepSeek-style sparse attention (DSA), gated MLA, learnable attention sinks, and independent
    Hyper-Connections (iHC).

    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by
        sparse MoE layers.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer type, either `"full"` or `"shared"`. A shared layer reuses the
        most recent full indexer in the same forward request.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each DSA query.
    index_head_dim (`int`, *optional*, defaults to 128):
        Hidden dimension of each DSA indexer head.
    index_n_heads (`int`, *optional*, defaults to 16):
        Number of DSA indexer heads.
    enable_lm_head_fp32 (`bool`, *optional*, defaults to `True`):
        Whether the language-model head is evaluated in float32.
    enable_ihc (`bool`, *optional*, defaults to `True`):
        Whether independent Hyper-Connections are enabled.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of hidden-state channels maintained by iHC.
    hc_magnitude (`float`, *optional*, defaults to 2.0):
        Scale applied to the iHC post-gating branch.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical epsilon added to iHC sigmoid gates.
    gated_mla (`bool`, *optional*, defaults to `True`):
        Whether to gate the MLA output.
    gating_type (`str`, *optional*, defaults to `"elementwise"`):
        MLA gate granularity, either `"elementwise"` or `"headwise"`.
    learnable_sink (`bool`, *optional*, defaults to `True`):
        Whether to add a learned per-head attention sink.
    learnable_sink_init (`float`, *optional*, defaults to 0.0):
        Initial value of each learned attention-sink logit.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.
    """

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.linear_gate": "colwise",
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
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 120832
    hidden_size: int = 2816
    intermediate_size: int = 6912
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.006
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 120002
    bos_token_id: int | None = 120000
    eos_token_id: int | list[int] | None = 120025
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    mlp_layer_types: list[str] | None = None
    layer_types: list[str] | None = None
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    indexer_types: list[str] | None = None
    enable_lm_head_fp32: bool = True
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    gated_mla: bool = True
    gating_type: str = "elementwise"
    learnable_sink: bool = True
    learnable_sink_init: float = 0.0
    swiglu_limit: float = 10.0
    rope_parameters: RopeParameters | dict | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so point `head_dim` at it: the inherited rotary
        # embedding reads `config.head_dim` and then computes the right frequencies with no override.
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + ["sparse"] * max(
                self.num_hidden_layers - 1, 0
            )
        if self.layer_types is None:
            # All HYV4 attention layers are DeepSeek-style sparse attention; this canonical layer
            # type makes the cache provision a `DynamicIndexedLayer` (with the DSA indexer-key cache)
            # for every layer.
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


class HYV4RMSNorm(Glm4MoeLiteRMSNorm):
    pass


class HYV4RotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    pass


class HYV4Indexer(nn.Module):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.scaling = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        query_residual: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.LongTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.wq_b(query_residual).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key_states = F.layer_norm(
            self.wk(hidden_states).float(),
            (self.head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.k_norm.eps,
        ).to(hidden_states.dtype)

        query_pass, query_rotary = torch.split(
            query_states, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        key_pass, key_rotary = torch.split(
            key_states, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        query_rotary = query_rotary.transpose(1, 2)
        key_rotary = key_rotary.unsqueeze(1)
        cos, sin = position_embeddings
        query_rotary, key_rotary = apply_rotary_pos_emb(query_rotary, key_rotary, cos, sin)
        query_states = torch.cat([query_pass, query_rotary.transpose(1, 2)], dim=-1)
        key_states = torch.cat([key_pass, key_rotary.squeeze(1)], dim=-1)

        if past_key_values is not None:
            key_states = past_key_values.update_indexer(key_states, self.layer_idx)

        head_weights = F.linear(hidden_states.float(), self.weights_proj.weight.float())
        head_weights = head_weights * (self.num_heads**-0.5) * self.scaling
        scores = torch.einsum("bshd,btd->bsht", query_states.float(), key_states.float())
        scores = F.relu(scores)
        scores = torch.einsum("bsht,bsh->bst", scores, head_weights)
        if attention_mask is not None:
            indexer_mask = attention_mask[:, 0] if attention_mask.ndim == 4 else attention_mask
            if indexer_mask.dtype == torch.bool:
                scores = scores.masked_fill(~indexer_mask, float("-inf"))
            else:
                scores = scores + indexer_mask.float()
        topk = min(self.index_topk, scores.shape[-1])
        topk_scores, topk_indices = scores.topk(topk, dim=-1)
        return torch.where(topk_scores.isneginf(), torch.full_like(topk_indices, -1), topk_indices)


class HYV4Attention(Glm4MoeLiteAttention):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer_type = config.indexer_types[layer_idx]
        if self.indexer_type == "full":
            self.indexer = HYV4Indexer(config, layer_idx)
        self.gated_mla = config.gated_mla
        self.gate_projection_size = 1 if config.gating_type == "headwise" else self.v_head_dim
        if self.gated_mla:
            self.linear_gate = nn.Linear(config.hidden_size, self.num_heads * self.gate_projection_size, bias=False)
        self.learnable_sink = config.learnable_sink
        if self.learnable_sink:
            self.learnable_sink_param = nn.Parameter(
                torch.full((self.num_heads,), config.learnable_sink_init, dtype=torch.float32)
            )

    def _add_sink_to_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, _, query_length, _ = query_states.shape
        key_length = key_states.shape[-2]
        if attention_mask is None:
            query_positions = torch.arange(query_length, device=query_states.device) + key_length - query_length
            key_positions = torch.arange(key_length, device=query_states.device)
            allowed_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            additive_mask = torch.zeros(
                (batch_size, 1, query_length, key_length),
                dtype=query_states.dtype,
                device=query_states.device,
            ).masked_fill(~allowed_mask.unsqueeze(0).unsqueeze(0), torch.finfo(query_states.dtype).min)
        elif attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(attention_mask, dtype=query_states.dtype).masked_fill(
                ~attention_mask, torch.finfo(query_states.dtype).min
            )
        else:
            additive_mask = attention_mask.to(dtype=query_states.dtype)
        additive_mask = additive_mask[..., :key_length]
        if additive_mask.shape[1] == 1 and self.num_heads > 1:
            additive_mask = additive_mask.expand(-1, self.num_heads, -1, -1)

        zero_key = torch.zeros_like(key_states[..., :1, :])
        zero_value = torch.zeros_like(value_states[..., :1, :])
        key_states = torch.cat([key_states, zero_key], dim=-2)
        value_states = torch.cat([value_states, zero_value], dim=-2)
        sink_bias = self.learnable_sink_param.to(query_states.dtype).view(1, -1, 1, 1)
        sink_bias = sink_bias.expand(batch_size, -1, query_length, -1)
        additive_mask = torch.cat([additive_mask, sink_bias], dim=-1)

        if self.config._attn_implementation == "sdpa":
            output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=additive_mask,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                scale=self.scaling,
            )
            return output.transpose(1, 2).contiguous(), None

        logits = torch.matmul(query_states.float(), key_states.float().transpose(-2, -1)) * self.scaling
        logits = logits + additive_mask.float()
        attention_weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        output = torch.matmul(attention_weights, value_states)
        return output.transpose(1, 2).contiguous(), attention_weights[..., :-1]

    def _build_sparse_mask(
        self,
        topk_indices: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        key_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if topk_indices is None:
            return attention_mask
        batch_size, query_length, _ = topk_indices.shape
        safe_indices = topk_indices.clamp_min(0)
        valid_indices = topk_indices.ge(0)
        selected_counts = torch.zeros(
            (batch_size, query_length, key_length),
            dtype=torch.int32,
            device=topk_indices.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, valid_indices.to(torch.int32))
        selected_mask = selected_counts.gt(0).unsqueeze(1)
        sparse_mask = torch.zeros(
            (batch_size, 1, query_length, key_length), dtype=dtype, device=topk_indices.device
        ).masked_fill(~selected_mask, torch.finfo(dtype).min)
        if attention_mask is None:
            return sparse_mask
        attention_mask = attention_mask[..., :key_length]
        if attention_mask.dtype == torch.bool:
            return sparse_mask.masked_fill(~attention_mask, torch.finfo(dtype).min)
        return sparse_mask + attention_mask.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.LongTensor | None]:
        batch_size, sequence_length = hidden_states.shape[:-1]
        query_shape = (batch_size, sequence_length, -1, self.qk_head_dim)
        key_shape = (batch_size, sequence_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        gate_score = None
        if self.gated_mla:
            gate_score = self.linear_gate(hidden_states).view(
                batch_size, sequence_length, self.num_heads, self.gate_projection_size
            )

        if self.q_lora_rank is None:
            query_residual = None
            query_states = self.q_proj(hidden_states)
        else:
            query_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query_states = self.q_b_proj(query_residual)
        query_states = query_states.view(query_shape).transpose(1, 2)
        query_pass, query_rotary = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        key_pass, key_rotary = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_pass = self.kv_b_proj(self.kv_a_layernorm(key_pass)).view(key_shape).transpose(1, 2)
        key_pass, value_states = torch.split(key_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_rotary = key_rotary.view(batch_size, 1, sequence_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        query_rotary, key_rotary = apply_rotary_pos_emb(query_rotary, key_rotary, cos, sin)
        key_rotary = key_rotary.expand(*key_pass.shape[:-1], -1)
        query_states = torch.cat((query_pass, query_rotary), dim=-1)
        key_states = torch.cat((key_pass, key_rotary), dim=-1)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.indexer_type == "full":
            topk_indices = self.indexer(
                hidden_states,
                query_residual,
                position_embeddings,
                attention_mask,
                past_key_values,
            )
        else:
            topk_indices = prev_topk_indices
        attention_mask = self._build_sparse_mask(
            topk_indices, attention_mask, key_states.shape[-2], query_states.dtype
        )

        if self.learnable_sink:
            attention_output, attention_weights = self._add_sink_to_attention(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attention_output, attention_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        if self.gated_mla:
            attention_output = attention_output * torch.sigmoid(gate_score)
        attention_output = attention_output.reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(attention_output), attention_weights, topk_indices


class HYV4MLP(Glm4MoeLiteMLP):
    pass


class HYV4TopKRouter(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        routing_weights = torch.sigmoid(router_logits)
        scores_for_choice = routing_weights + self.e_score_correction_bias
        top_k_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False).indices
        top_k_weights = routing_weights.gather(1, top_k_indices)
        if self.norm_topk_prob and self.top_k > 1:
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        top_k_weights = top_k_weights * self.router_scaling_factor
        return router_logits, top_k_weights, top_k_indices


class HYV4Experts(nn.Module):
    """Trainable stacked routed experts using the release checkpoint parameter layout."""

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = expert_idx_tensor[0]
            top_k_position, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            if self.swiglu_limit > 0:
                gate = gate.float().clamp(max=self.swiglu_limit).to(current_state.dtype)
                up = up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit).to(current_state.dtype)
            current_state = F.silu(gate) * up
            current_state = F.linear(current_state, self.down_proj[expert_idx])
            current_state = current_state * top_k_weights[token_idx, top_k_position, None]
            final_hidden_states.index_add_(0, token_idx, current_state.to(final_hidden_states.dtype))
        return final_hidden_states


class HYV4MoE(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.gate = HYV4TopKRouter(config)
        self.experts = HYV4Experts(config)
        self.shared_experts = HYV4MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)
        _, top_k_weights, top_k_indices = self.gate(flat_states)
        routed_output = self.experts(flat_states, top_k_indices, top_k_weights)
        output = routed_output.float() + self.shared_experts(flat_states).float()
        return output.to(hidden_states.dtype).reshape(batch_size, sequence_length, hidden_dim)


def _hc_rms_gated_logits(hidden_states: torch.Tensor, mix_weight: torch.Tensor, rms_eps: float) -> torch.Tensor:
    """Shared iHC projection: flatten the multi-channel hidden state, RMS-scale it in float32, and
    project it through ``mix_weight`` to produce per-channel gate logits.

    Returns logits of shape ``[batch, sequence, mix_weight.shape[0]]`` in float32. Both the pre/post
    gating layer and the head layer build their sigmoid gates on top of these logits; only the
    number of output gate groups and the presence of a post branch differ between them.
    """
    flat_states = hidden_states.flatten(2).float()
    inverse_rms = torch.rsqrt(flat_states.square().mean(-1, keepdim=True) + rms_eps)
    return F.linear(flat_states, mix_weight.float()) * inverse_rms


class HYV4HCPreLayer(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.layernorm_epsilon = config.rms_norm_eps
        self.hc_fn = nn.Parameter(torch.empty(2 * self.hc_mult, self.hc_mult * self.hidden_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(2 * self.hc_mult, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        input_dtype = hidden_states.dtype
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            mixes = _hc_rms_gated_logits(hidden_states, self.hc_fn, self.layernorm_epsilon)
            pre_logits, post_logits = mixes.split(self.hc_mult, dim=-1)
            pre_gates = (
                torch.sigmoid(pre_logits * self.hc_scale[0].float() + self.hc_base[: self.hc_mult].float())
                + self.hc_eps
            )
            post_gates = (
                self.magnitude
                * torch.sigmoid(post_logits * self.hc_scale[1].float() + self.hc_base[self.hc_mult :].float())
                + self.hc_eps
            )
            reduced_states = torch.sum(pre_gates.unsqueeze(-1) * hidden_states.reshape(original_shape), dim=2)
        return reduced_states.to(input_dtype), post_gates


class HYV4HCPostLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, post_gates: torch.Tensor) -> torch.Tensor:
        output = post_gates.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2)
        return (output + residual.float()).to(hidden_states.dtype)


class HYV4HCHeadLayer(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * self.hidden_size, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        input_dtype = hidden_states.dtype
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            mixes = _hc_rms_gated_logits(hidden_states, self.hc_head_fn, self.rms_norm_eps)
            pre_gates = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float()) + self.hc_eps
            output = torch.sum(pre_gates.unsqueeze(-1) * hidden_states.reshape(original_shape), dim=2)
        return output.to(input_dtype)


class HYV4HCLayer(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.enable_ihc = config.enable_ihc
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(config)
            self.hc_post = HYV4HCPostLayer()

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enable_ihc or hidden_states.ndim == 4:
            return hidden_states
        return hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def pre(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced_states, post_gates = self.hc_pre(hidden_states)
        return reduced_states, post_gates, hidden_states

    def post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.enable_ihc:
            return hidden_states + residual
        return self.hc_post(hidden_states, residual, post_gates)


class HYV4DecoderLayer(Glm4MoeLiteDecoderLayer, nn.Module):
    def __init__(self, config: HYV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mlp = HYV4MoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else HYV4MLP(config)
        self.input_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HCLayer(config)
        self.hc_mlp_layer = HYV4HCLayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.LongTensor | None]:
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)
        return hidden_states, topk_indices


class HYV4PreTrainedModel(Glm4MoeLitePreTrainedModel):
    config: HYV4Config
    _no_split_modules = ["HYV4DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp_layers\.0\..*"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "hc_fn",
        "hc_scale",
        "hc_base",
        "hc_head_fn",
        "hc_head_scale",
        "hc_head_base",
        "learnable_sink_param",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, HYV4TopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, HYV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, HYV4HCPreLayer):
            init.normal_(module.hc_fn, mean=0.0, std=6e-3)
            init.constant_(module.hc_scale, 0.01)
            if not getattr(module.hc_base, "_is_hf_initialized", False):
                base_value = -float(torch.log(torch.tensor(max(module.hc_mult - 1, 1), dtype=torch.float32)))
                module.hc_base[: module.hc_mult].fill_(base_value)
                module.hc_base[module.hc_mult :].zero_()
        elif isinstance(module, HYV4HCHeadLayer):
            init.normal_(module.hc_head_fn, mean=0.0, std=6e-3)
            init.constant_(module.hc_head_scale, 0.01)
            if not getattr(module.hc_head_base, "_is_hf_initialized", False):
                base_value = -float(torch.log(torch.tensor(max(module.hc_mult - 1, 1), dtype=torch.float32)))
                module.hc_head_base.fill_(base_value)
        elif isinstance(module, HYV4Attention) and module.learnable_sink:
            if not getattr(module.learnable_sink_param, "_is_hf_initialized", False):
                init.constant_(module.learnable_sink_param, self.config.learnable_sink_init)


class HYV4Model(Glm4MoeLiteModel):
    def __init__(self, config: HYV4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HYV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HYV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV4RotaryEmbedding(config=config)
        self.enable_ihc = config.enable_ihc
        if self.enable_ihc:
            self.hc_head = HYV4HCHeadLayer(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if causal_mask is None:
            query_length = inputs_embeds.shape[1]
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            key_length = past_length + query_length
            query_positions = torch.arange(query_length, device=inputs_embeds.device) + past_length
            key_positions = torch.arange(key_length, device=inputs_embeds.device)
            allowed_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            causal_mask = torch.zeros(
                (inputs_embeds.shape[0], 1, query_length, key_length),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).masked_fill(~allowed_mask.unsqueeze(0).unsqueeze(0), torch.finfo(inputs_embeds.dtype).min)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HYV4ForCausalLM(Glm4MoeLiteForCausalLM):
    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def __init__(self, config: HYV4Config):
        super().__init__(config)
        self.model = HYV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        head_input = hidden_states[:, slice_indices, :]
        if self.config.enable_lm_head_fp32:
            logits = F.linear(head_input.float(), self.lm_head.weight.float())
        else:
            logits = self.lm_head(head_input)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["HYV4Config", "HYV4PreTrainedModel", "HYV4Model", "HYV4ForCausalLM"]
