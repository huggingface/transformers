# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team.
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

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils.generic import is_flash_attention_requested
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from ..mixtral.modeling_mixtral import MixtralExperts
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


class DeepseekV32Config(DeepseekV3Config):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV32Model`]. It extends
    [`DeepseekV3Config`] with the Dynamic Sparse Attention (DSA) indexer parameters introduced in DeepSeek-V3.2.

    Args:
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top tokens selected by the DSA indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for the DSA indexer projections.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of heads used by the DSA indexer.
        ep_size (`int`, *optional*, defaults to 1):
            Parallelism metadata stored in the official DeepSeek-V3.2 configs.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            MoE layer frequency metadata stored in the official DeepSeek-V3.2 configs.
        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of next-token prediction layers recorded in official configs.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Router scoring function metadata stored in the official configs.
        topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
            Router top-k selection metadata stored in the official configs.
    """

    model_type = "deepseek_v32"
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

    def __init__(
        self,
        vocab_size: int | None = 129280,
        hidden_size: int | None = 7168,
        intermediate_size: int | None = 18432,
        moe_intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 61,
        num_attention_heads: int | None = 128,
        num_key_value_heads: int | None = 128,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_nope_head_dim: int | None = 128,
        n_group: int | None = 8,
        topk_group: int | None = 4,
        num_experts_per_tok: int | None = 8,
        first_k_dense_replace: int | None = 3,
        norm_topk_prob: bool | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 163840,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rope_interleave: bool | None = True,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        index_topk: int | None = 2048,
        index_head_dim: int | None = 128,
        index_n_heads: int | None = 64,
        ep_size: int | None = 1,
        moe_layer_freq: int | None = 1,
        num_nextn_predict_layers: int | None = 1,
        scoring_func: str | None = "sigmoid",
        topk_method: str | None = "noaux_tc",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=n_shared_experts,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            num_experts_per_tok=num_experts_per_tok,
            first_k_dense_replace=first_k_dense_replace,
            norm_topk_prob=norm_topk_prob,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            rope_interleave=rope_interleave,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.ep_size = ep_size
        self.moe_layer_freq = moe_layer_freq
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.scoring_func = scoring_func
        self.topk_method = topk_method

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize official DeepSeek-V3.2 configs before validation. Hub checkpoints store some YaRN values as ints.
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()
        for key in ["beta_fast", "beta_slow", "factor"]:
            if key in self.rope_parameters:
                self.rope_parameters[key] = float(self.rope_parameters[key])
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)

        return kwargs


class DeepseekV32RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV32RotaryEmbedding(LlamaRotaryEmbedding):
    pass


class DeepseekV32MLP(Qwen2MoeMLP):
    pass


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_interleave(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV32TopkRouter(nn.Module):
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


class DeepseekV32NaiveMoe(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_local_experts
        self.intermediate_dim = config.moe_intermediate_size


class DeepseekV32MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV32NaiveMoe(config)
        self.gate = DeepseekV32TopkRouter(config)
        self.shared_experts = DeepseekV32MLP(
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


class DeepseekV32Indexer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
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
        self._cached_keys: torch.Tensor | None = None

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        use_cache: bool = False,
    ) -> torch.LongTensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV32Attention(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
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
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV32RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV32RMSNorm(self.kv_lora_rank)
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

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            raise ValueError("DeepseekV32Attention requires `q_lora_rank` to build the DSA indexer.")

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if self.config.rope_interleave:
            q_pe, _ = apply_rotary_pos_emb_interleave(q_pe, q_pe, cos, sin)
        else:
            q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        if self.config.rope_interleave:
            k_pe, _ = apply_rotary_pos_emb_interleave(k_pe, k_pe, cos, sin)
        else:
            k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            indexer_mask,
            use_cache=past_key_values is not None,
        )

        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)
        index_mask = index_mask.unsqueeze(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[..., :total_len]
            combined_mask = index_mask + causal_mask
        else:
            combined_mask = (
                attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
                if attention_mask is not None
                else index_mask
            )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=topk_indices,
            **kwargs,
        )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV32Attention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV32MoE(config)
        else:
            self.mlp = DeepseekV32MLP(config)

        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV32PreTrainedModel(LlamaPreTrainedModel):
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _preload_module_classes = ["DeepseekV32MoE"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _compatible_flash_implementations = ["kernels-community/flash-mla"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekV32TopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, DeepseekV32NaiveMoe):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class DeepseekV32Model(LlamaModel):
    pass


class DeepseekV32ForCausalLM(LlamaForCausalLM):
    pass


__all__ = ["DeepseekV32Config", "DeepseekV32PreTrainedModel", "DeepseekV32Model", "DeepseekV32ForCausalLM"]
