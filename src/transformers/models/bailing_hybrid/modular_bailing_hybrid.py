# Copyright 2026 The HuggingFace Inc. team
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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ...models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
)
from ...models.glm5_next.modeling_glm5_next import (
    Glm5NextTextRMSNormGated,
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    recurrent_kimi_delta_attention,
)
from ...models.llama.modeling_llama import LlamaRMSNorm, apply_rotary_pos_emb, eager_attention_forward
from ...models.qwen3_next.modeling_qwen3_next import Qwen3NextModel, Qwen3NextPreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder


@auto_docstring(checkpoint="inclusionAI/Ling-3.0-flash")
@strict
class BailingHybridConfig(DeepseekV3Config):
    r"""
    layer_group_size (`int`, *optional*, defaults to 6):
        Number of layers in each hybrid attention group. The last layer in every group uses full MLA and the other
        layers use KDA.
    short_conv_kernel_size (`int`, *optional*, defaults to 4):
        Kernel size of the short depthwise convolutions in KDA layers.
    kda_safe_gate (`bool`, *optional*, defaults to `True`):
        Whether to use the bounded KDA forget gate.
    kda_lower_bound (`float`, *optional*, defaults to -5.0):
        Lower bound of the KDA forget gate in log space.
    gated_attention_proj_granularity_type (`str`, *optional*, defaults to `"head_wise"`):
        Granularity of the output gate applied to MLA heads.
    n_group (`int`, *optional*, defaults to 8):
        Number of expert groups used by the group-limited router.
    first_k_dense_replace (`int`, *optional*, defaults to 2):
        Number of initial decoder layers that use a dense MLP instead of routed experts.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether rotary dimensions in MLA layers are stored as interleaved pairs.
    rope_theta (`float`, *optional*, defaults to 6000000.0):
        Base period of the rotary position embeddings used by MLA layers.
    no_kda_lora (`bool`, *optional*, defaults to `True`):
        Whether KDA forget and output gates use direct projections instead of low-rank projections.
    number_of_conv_states (`int`, *optional*, defaults to 3):
        Number of short-convolution cache states per KDA layer, one each for queries, keys, and values.
    num_nextn_predict_layers (`int`, *optional*, defaults to 1):
        Number of auxiliary multi-token-prediction layers stored in released training checkpoints. These layers are
        not instantiated for standard causal language modeling.
    num_mtp_layers (`int`, *optional*, defaults to 1):
        Legacy alias for the number of multi-token-prediction layers in a training checkpoint.
    mtp_loss_scaling_factor (`float`, *optional*, defaults to 0.0):
        Scaling factor used for the auxiliary multi-token-prediction loss during pretraining.
    """

    model_type = "bailing_hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
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
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }
    attribute_map = {
        "max_position_embeddings": "model_max_length",
        "norm_topk_prob": "moe_renormalize",
        "n_group": "num_expert_group",
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_token",
        "n_shared_experts": "num_shared_experts",
    }

    vocab_size: int = 157184
    hidden_size: int = 2560
    intermediate_size: int = 6144
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    num_local_experts: int = 512
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 156892
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 156895
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    output_router_logits: bool = False

    q_lora_rank: int | None = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_interleave: bool = True
    rope_theta: float | int = 6_000_000.0

    head_dim: int = 128
    short_conv_kernel_size: int = 4
    layer_group_size: int = 6
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    no_kda_lora: bool = True
    gated_attention_proj_granularity_type: str | None = "head_wise"
    layer_types: list[str] | None = None
    number_of_conv_states: int = 3

    num_nextn_predict_layers: int = 1
    mtp_loss_scaling_factor: float | int = 0.0

    # Ling checkpoints use `num_experts`; the DeepSeek compatibility name is not part of this architecture.
    n_routed_experts = AttributeError()

    def __post_init__(self, **kwargs):
        self.linear_head_dim = self.head_dim
        self.linear_num_heads = self.num_attention_heads
        self.linear_conv_kernel_dim = self.short_conv_kernel_size

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (layer_idx + 1) % self.layer_group_size == 0 else "linear_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]

        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": self.rope_theta}

        super().__post_init__(**kwargs)


class BailingHybridRMSNorm(LlamaRMSNorm):
    pass


class BailingHybridRMSNormGated(Glm5NextTextRMSNormGated):
    pass


class BailingHybridRotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class BailingHybridExperts(DeepseekV3Experts):
    pass


class BailingHybridMLP(DeepseekV3MLP):
    pass


class BailingHybridAttention(DeepseekV3Attention):
    """Multi-head latent attention with the head-wise output gate used by Ling 3.0."""

    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gated_attention_proj_granularity_type = config.gated_attention_proj_granularity_type
        if self.gated_attention_proj_granularity_type is None:
            self.g_proj = None
        elif self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = nn.Linear(config.hidden_size, self.num_heads * self.v_head_dim, bias=False)
        else:
            raise ValueError(
                "`gated_attention_proj_granularity_type` must be one of None, 'head_wise', or 'element_wise'"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(kv_nope, k_rot)

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
            **kwargs,
        )

        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states).float()).to(hidden_states.dtype)
            if self.gated_attention_proj_granularity_type == "head_wise":
                attn_output = attn_output * gate.unsqueeze(-1)
            else:
                attn_output = attn_output * gate.view(batch_size, seq_length, self.num_heads, self.v_head_dim)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BailingHybridShortConvolution(nn.Module):
    """Depthwise causal convolution with the parameter layout used by FLA's `ShortConvolution`."""

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))


class BailingHybridKimiDeltaAttention(nn.Module):
    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = "silu"
        self.safe_gate = config.kda_safe_gate
        self.lower_bound = config.kda_lower_bound
        self.no_kda_lora = config.no_kda_lora

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.q_conv1d = BailingHybridShortConvolution(self.qkv_dim, self.conv_kernel_size)
        self.k_conv1d = BailingHybridShortConvolution(self.qkv_dim, self.conv_kernel_size)
        self.v_conv1d = BailingHybridShortConvolution(self.qkv_dim, self.conv_kernel_size)

        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        if self.no_kda_lora:
            self.f_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
            self.g_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        else:
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.o_norm = BailingHybridRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

    def _convolution(
        self,
        projected_states: torch.Tensor,
        convolution: BailingHybridShortConvolution,
        cache_params: Cache | None,
        state_idx: int,
        use_precomputed_states: bool,
    ) -> torch.Tensor:
        projected_states = projected_states.transpose(1, 2)
        if use_precomputed_states and projected_states.shape[-1] == 1:
            conv_state = cache_params.layers[self.layer_idx].conv_states[state_idx]
            if projected_states.device.type != "cuda":
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
                conv_state[:, :, -1:] = projected_states
                output = (conv_state * convolution.weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
                return F.silu(output)
            return causal_conv1d_update(
                projected_states,
                conv_state,
                weight=convolution.weight,
                activation=self.activation,
            )

        if cache_params is not None:
            projected_states = cache_params.update_conv_state(
                projected_states,
                self.layer_idx,
                state_idx=state_idx,
                conv_kernel_size=self.conv_kernel_size,
            )
        if projected_states.device.type != "cuda":
            output = F.conv1d(
                projected_states,
                convolution.weight.unsqueeze(1),
                padding=self.conv_kernel_size - 1,
                groups=self.qkv_dim,
            )[..., : projected_states.shape[-1]]
            return F.silu(output)
        return causal_conv1d_fn(
            projected_states,
            weight=convolution.weight,
            activation=self.activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask[:, -hidden_states.shape[1] :, None].to(hidden_states.dtype)

        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None

        query = self._convolution(self.q_proj(hidden_states), self.q_conv1d, cache_params, 0, use_precomputed_states)
        key = self._convolution(self.k_proj(hidden_states), self.k_conv1d, cache_params, 1, use_precomputed_states)
        value = self._convolution(self.v_proj(hidden_states), self.v_conv1d, cache_params, 2, use_precomputed_states)
        query, key, value = [
            states[..., -seq_len:].transpose(1, 2).view(hidden_shape) for states in (query, key, value)
        ]

        if self.no_kda_lora:
            raw_gate = self.f_proj(hidden_states)
        else:
            raw_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        raw_gate = raw_gate.float().view(hidden_shape)
        gate_input = raw_gate + self.dt_bias.float().view(1, 1, self.num_heads, self.head_dim)
        decay_rate = self.A_log.float().exp().view(1, 1, self.num_heads, 1)
        if self.safe_gate:
            decay = self.lower_bound * torch.sigmoid(decay_rate * gate_input)
        else:
            decay = -decay_rate * F.softplus(gate_input)

        beta = torch.sigmoid(self.b_proj(hidden_states).float())
        if use_precomputed_states and seq_len == 1:
            attention_fn = recurrent_kimi_delta_attention
            if query.device.type != "cuda":
                attention_fn = attention_fn.__wrapped__
            output, last_recurrent_state = attention_fn(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            attention_fn = chunk_kimi_delta_attention
            if query.device.type != "cuda":
                attention_fn = attention_fn.__wrapped__
            output, last_recurrent_state = attention_fn(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        if self.no_kda_lora:
            output_gate = self.g_proj(hidden_states)
        else:
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output = self.o_norm(output, output_gate.view(hidden_shape)).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class BailingHybridTopkRouter(DeepseekV3TopkRouter):
    pass


class BailingHybridMoE(DeepseekV3MoE):
    pass


class BailingHybridDecoderLayer(nn.Module):
    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        self.self_attn = (
            BailingHybridAttention(config, layer_idx)
            if self.block_type == "full_attention"
            else BailingHybridKimiDeltaAttention(config, layer_idx)
        )
        self.mlp = (
            BailingHybridMoE(config)
            if config.num_experts is not None and layer_idx >= config.first_k_dense_replace
            else BailingHybridMLP(config)
        )
        self.input_layernorm = BailingHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
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


@auto_docstring
class BailingHybridPreTrainedModel(Qwen3NextPreTrainedModel):
    _no_split_modules = ["BailingHybridDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.42\..*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(BailingHybridTopkRouter, index=0),
        "hidden_states": BailingHybridDecoderLayer,
        "attentions": BailingHybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, BailingHybridKimiDeltaAttention):
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))
        elif isinstance(module, BailingHybridShortConvolution):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BailingHybridExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BailingHybridTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, BailingHybridRMSNormGated):
            init.ones_(module.weight)


@auto_docstring
class BailingHybridModel(Qwen3NextModel):
    def __init__(self, config: BailingHybridConfig):
        super().__init__(config)
        self.rotary_emb = BailingHybridRotaryEmbedding(config=config)

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=(
                    position_embeddings if self.config.layer_types[layer_idx] == "full_attention" else None
                ),
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class BailingHybridForCausalLM(DeepseekV3ForCausalLM, GenerationMixin):
    _tied_weights_keys = {}


__all__ = [
    "BailingHybridConfig",
    "BailingHybridPreTrainedModel",
    "BailingHybridModel",
    "BailingHybridForCausalLM",
]
