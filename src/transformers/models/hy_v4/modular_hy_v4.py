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
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4UnweightedRMSNorm
from ..glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteExperts,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteMLP,
    Glm4MoeLiteModel,
    Glm4MoeLiteMoE,
    Glm4MoeLitePreTrainedModel,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteTopkRouter,
    apply_rotary_pos_emb,
)
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, GlmMoeDsaRotaryEmbedding
from ..gpt_oss.modeling_gpt_oss import eager_attention_forward


@auto_docstring(custom_intro="HYV4")
@strict
class HYV4Config(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for routing. HYV4 selects experts globally, so this is 1 (one group holding every expert).
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups kept during routing. With `n_group=1` this reuses `Glm4MoeLiteTopkRouter` as a plain global top-k.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by
        sparse MoE layers.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each DSA query.
    index_head_dim (`int`, *optional*, defaults to 128):
        Hidden dimension of each DSA indexer head.
    index_n_heads (`int`, *optional*, defaults to 16):
        Number of DSA indexer heads.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer type, either `"full"` or `"shared"`. A shared layer reuses the
        most recent full indexer in the same forward request.
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
        "layers.*.self_attn.sinks": "colwise",
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
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
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
    n_group: int = 1
    topk_group: int = 1
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
        # MLA expands the latent to one key/value per query head, so keys are never grouped.
        self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so `head_dim` points at it.
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + ["sparse"] * max(
                self.num_hidden_layers - 1, 0
            )
        if self.layer_types is None:
            self.layer_types = ["deepseek_sparse_attention"] * self.num_hidden_layers
        if self.indexer_types is None:
            self.indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


class HYV4RMSNorm(Glm4MoeLiteRMSNorm):
    pass


class HYV4UnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    def inverse_rms(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + self.eps)


class HYV4RotaryEmbedding(GlmMoeDsaRotaryEmbedding):
    pass


class HYV4Indexer(GlmMoeDsaIndexer):
    """GLM-MoE-DSA indexer with HYV4's split-half RoPE and FP32 key normalization."""

    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        query_residual: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.LongTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.wq_b(query_residual).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        key_states = self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype)

        query_pass, query_rotary = torch.split(
            query_states, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        key_pass, key_rotary = torch.split(
            key_states, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        query_rotary = query_rotary.transpose(1, 2)
        key_rotary = key_rotary.unsqueeze(1)
        cos, sin = position_embeddings
        query_rotary, key_rotary = apply_rotary_pos_emb(query_rotary, key_rotary, cos, sin)
        query_states = torch.cat([query_pass, query_rotary.transpose(1, 2)], dim=-1)
        key_states = torch.cat([key_pass, key_rotary.squeeze(1)], dim=-1)

        if past_key_values is not None:
            key_states = past_key_values.update_indexer(key_states, self.layer_idx)

        head_weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        head_weights = head_weights * (self.n_heads**-0.5) * self.softmax_scale
        scores = torch.matmul(query_states.float(), key_states.float().unsqueeze(1).transpose(-1, -2))
        scores = F.relu(scores)
        scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)
        if attention_mask is not None:
            scores = scores + attention_mask
        else:
            key_positions = torch.arange(scores.shape[-1], device=scores.device)
            causal = key_positions[None, None, :] > position_ids[:, :, None]
            scores = scores.masked_fill(causal, float("-inf"))
        topk = min(self.index_topk, scores.shape[-1])
        return scores.topk(topk, dim=-1).indices.to(torch.int32)


class HYV4Attention(Glm4MoeLiteAttention):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer_type = config.indexer_types[layer_idx]
        self.indexer = HYV4Indexer(config, layer_idx) if self.indexer_type == "full" else None
        self.gate_projection_size = self.v_head_dim
        self.linear_gate = nn.Linear(config.hidden_size, self.num_heads * self.gate_projection_size, bias=False)
        self.sinks = nn.Parameter(torch.full((self.num_heads,), config.learnable_sink_init, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.LongTensor | None]:
        batch_size, sequence_length = hidden_states.shape[:-1]
        query_shape = (batch_size, sequence_length, -1, self.qk_head_dim)
        key_shape = (batch_size, sequence_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        gate_score = self.linear_gate(hidden_states).view(batch_size, sequence_length, -1, self.gate_projection_size)

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

        if self.indexer is not None:
            indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
            topk_indices = self.indexer(
                hidden_states,
                query_residual,
                position_embeddings,
                indexer_mask,
                position_ids,
                past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        index_mask = (
            topk_indices.new_ones((batch_size, sequence_length, key_states.shape[2]), dtype=torch.bool)
            .scatter(-1, topk_indices.long(), False)
            .unsqueeze(1)
        )
        if attention_mask is None:
            key_positions = torch.arange(key_states.shape[2], device=hidden_states.device)
            index_mask = index_mask | (key_positions[None, None, None, :] > position_ids[:, None, :, None])
            attention_mask = hidden_states.new_zeros((batch_size, 1, sequence_length, key_states.shape[2]))
        attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)

        attention_output, attention_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )
        attention_output = attention_output * torch.sigmoid(gate_score)
        attention_output = attention_output.reshape(batch_size, sequence_length, -1).contiguous()
        return self.o_proj(attention_output), attention_weights, topk_indices


class HYV4MLP(Glm4MoeLiteMLP):
    pass


class HYV4TopKRouter(Glm4MoeLiteTopkRouter):
    pass


@use_experts_implementation
class HYV4Experts(Glm4MoeLiteExperts):
    """GLM4-MoE-Lite experts with HYV4's bounded SwiGLU and EP sentinel handling."""

    def __init__(self, config: HYV4Config):
        if config._experts_implementation == "sonicmoe":
            raise ValueError("HYV4 does not support SonicMoE because its fused SwiGLU omits `swiglu_limit`.")
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        if self.swiglu_limit > 0:
            gate = gate.float().clamp(max=self.swiglu_limit).to(gate.dtype)
            up = up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit).to(up.dtype)
        return self.act_fn(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_position, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx].to(self.gate_up_proj.dtype)
            current_state = self._apply_gate(F.linear(current_state, self.gate_up_proj[expert_idx]))
            current_state = F.linear(current_state, self.down_proj[expert_idx])
            current_state = current_state * top_k_weights[token_idx, top_k_position, None]
            final_hidden_states.index_add_(0, token_idx, current_state.to(final_hidden_states.dtype))
        return final_hidden_states


class HYV4MoE(Glm4MoeLiteMoE):
    def __init__(self, config: HYV4Config):
        nn.Module.__init__(self)
        self.config = config
        self.gate = HYV4TopKRouter(config)
        self.experts = HYV4Experts(config)
        self.shared_experts = HYV4MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)


class HYV4HyperConnection(nn.Module):
    """Independent Hyper-Connection following the DeepSeek-V4 form (parameters on the module)."""

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.magnitude = config.hc_magnitude
        self.hc_eps = config.hc_eps
        self.input_norm = HYV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.hc_fn = nn.Parameter(torch.empty(2 * self.hc_mult, self.hc_mult * self.hidden_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(2 * self.hc_mult, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 4:
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        original_shape = hidden_states.shape
        input_dtype = hidden_states.dtype
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            flat = hidden_states.flatten(2).float()
            inverse_rms = self.input_norm.inverse_rms(flat)
            mixes = F.linear(flat.to(self.hc_fn.dtype), self.hc_fn) * inverse_rms
            pre_logits, post_logits = mixes.split(self.hc_mult, dim=-1)
            pre_gates = torch.sigmoid(pre_logits * self.hc_scale[0] + self.hc_base[: self.hc_mult]) + self.hc_eps
            post_gates = (
                self.magnitude * torch.sigmoid(post_logits * self.hc_scale[1] + self.hc_base[self.hc_mult :])
                + self.hc_eps
            )
            collapsed = torch.sum(pre_gates.unsqueeze(-1) * hidden_states.reshape(original_shape), dim=2)
        return post_gates, collapsed.to(input_dtype), hidden_states


class HYV4HyperHead(nn.Module):
    """Final iHC head: collapse the multi-stream hidden state back to a single stream."""

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.input_norm = HYV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * self.hidden_size, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        input_dtype = hidden_states.dtype
        device_type = hidden_states.device.type if hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            flat = hidden_states.flatten(2).float()
            inverse_rms = self.input_norm.inverse_rms(flat)
            mixes = F.linear(flat.to(self.hc_head_fn.dtype), self.hc_head_fn) * inverse_rms
            pre_gates = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
            output = torch.sum(pre_gates.unsqueeze(-1) * hidden_states.reshape(original_shape), dim=2)
        return output.to(input_dtype)


class HYV4DecoderLayer(Glm4MoeLiteDecoderLayer, nn.Module):
    def __init__(self, config: HYV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mlp = HYV4MoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else HYV4MLP(config)
        self.input_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = HYV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hc_attn_layer = HYV4HyperConnection(config)
        self.hc_mlp_layer = HYV4HyperConnection(config)

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
        post_gates, collapsed, hidden_states = self.hc_attn_layer(hidden_states)
        attn_output, _, topk_indices = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = (
            post_gates.float().unsqueeze(-1) * attn_output.float().unsqueeze(-2) + hidden_states.float()
        ).to(attn_output.dtype)

        post_gates, collapsed, hidden_states = self.hc_mlp_layer(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed))
        hidden_states = (
            post_gates.float().unsqueeze(-1) * mlp_output.float().unsqueeze(-2) + hidden_states.float()
        ).to(mlp_output.dtype)
        return hidden_states, topk_indices


class HYV4PreTrainedModel(Glm4MoeLitePreTrainedModel):
    config: HYV4Config
    _no_split_modules = ["HYV4DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp_layers\..*"]
    _supports_flash_attn = False
    _supports_sdpa = False
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
        "weights_proj",
        "k_norm",
        "sinks",
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
        elif isinstance(module, HYV4HyperConnection):
            init.normal_(module.hc_fn, mean=0.0, std=6e-3)
            init.constant_(module.hc_scale, 0.01)
            if not getattr(module.hc_base, "_is_hf_initialized", False):
                base_value = -float(torch.log(torch.tensor(max(module.hc_mult - 1, 1), dtype=torch.float32)))
                module.hc_base[: module.hc_mult].fill_(base_value)
                module.hc_base[module.hc_mult :].zero_()
        elif isinstance(module, HYV4HyperHead):
            init.normal_(module.hc_head_fn, mean=0.0, std=6e-3)
            init.constant_(module.hc_head_scale, 0.01)
            if not getattr(module.hc_head_base, "_is_hf_initialized", False):
                base_value = -float(torch.log(torch.tensor(max(module.hc_mult - 1, 1), dtype=torch.float32)))
                module.hc_head_base.fill_(base_value)
        elif isinstance(module, HYV4Attention):
            if not getattr(module.sinks, "_is_hf_initialized", False):
                init.constant_(module.sinks, self.config.learnable_sink_init)


class HYV4Model(Glm4MoeLiteModel):
    def __init__(self, config: HYV4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [HYV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = HYV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HYV4RotaryEmbedding(config=config)
        self.hc_head = HYV4HyperHead(config)

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
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class HYV4ForCausalLM(Glm4MoeLiteForCausalLM):
    _keep_in_fp32_modules_strict = HYV4PreTrainedModel._keep_in_fp32_modules_strict + ["lm_head"]

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
        logits = self.lm_head(head_input.to(self.lm_head.weight.dtype))

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
