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
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...integrations.accelerate import force_accelerate_hooks
from ...modeling_flash_attention_utils import FlashAttentionKwargs
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
    Glm5NextTextLinearAttention,
    Glm5NextTextRMSNormGated,
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    recurrent_kimi_delta_attention,
)
from ...models.kimi_linear.modeling_kimi_linear import KimiLinearDecoderLayer
from ...models.llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ...models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
    apply_mask_to_padding_states,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder


@auto_docstring(checkpoint="inclusionAI/Ling-3.0-Flash")
@strict
class BailingHybridConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 8):
        Number of expert groups used by the group-limited router.
    first_k_dense_replace (`int`, *optional*, defaults to 2):
        Number of initial decoder layers that use a dense MLP instead of routed experts.
    num_mtp_layers (`int`, *optional*, defaults to 1):
        Number of Multi-Token Prediction (MTP) modules available to append after the base transformer model. When `0`,
        the model behaves as a standard decoder. When `>0`, each extra module can predict one additional future token at inference
        time (speculative decoding via `generate(..., use_mtp=True)`).
    rope_theta (`float`, *optional*, defaults to 6000000.0):
        Base period of the rotary position embeddings used by MLA layers.
    short_conv_kernel_size (`int`, *optional*, defaults to 4):
        Kernel size of the short depthwise convolutions in KDA layers.
    layer_group_size (`int`, *optional*, defaults to 6):
        Number of layers in each hybrid attention group. The last layer in every group uses full MLA and the other
        layers use KDA.
    kda_safe_gate (`bool`, *optional*, defaults to `True`):
        Whether to use the bounded KDA forget gate.
    kda_lower_bound (`float`, *optional*, defaults to -5.0):
        Lower bound of the KDA forget gate in log space.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP types. Derived from `first_k_dense_replace` when omitted.
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
        "num_mtp_layers": "num_nextn_predict_layers",
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

    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float | int = 6_000_000.0

    head_dim: int = 128
    short_conv_kernel_size: int = 4
    layer_group_size: int = 6
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None

    mtp_loss_scaling_factor: float | int = 0.0

    # Ling checkpoints use `num_experts`; the DeepSeek compatibility name is not part of this architecture.
    n_routed_experts = AttributeError()
    rope_interleave = AttributeError()
    q_lora_rank = AttributeError()

    def __post_init__(self, **kwargs):
        self.linear_head_dim = self.head_dim
        self.linear_num_heads = self.num_attention_heads
        self.linear_conv_kernel_dim = self.short_conv_kernel_size

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (layer_idx + 1) % self.layer_group_size == 0 else "linear_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if layer_idx < self.first_k_dense_replace else "sparse"
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
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        del self.q_a_proj
        del self.q_a_layernorm
        del self.q_b_proj
        del self.q_lora_rank

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

        q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

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

        gate = torch.sigmoid(self.g_proj(hidden_states).float()).to(hidden_states.dtype)
        attn_output = attn_output * gate.unsqueeze(-1)

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BailingHybridForgetGate(nn.Module):
    def __init__(self, config: BailingHybridConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.safe_gate = config.kda_safe_gate
        self.lower_bound = config.kda_lower_bound
        self.A_log = nn.Parameter(torch.empty(self.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.f_proj = nn.Linear(config.hidden_size, self.qkv_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        gate_input = self.f_proj(hidden_states).float().view(hidden_shape)
        gate_input = gate_input + self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        decay_rate = self.A_log.exp().view(1, 1, self.num_heads, 1)
        if self.safe_gate:
            gate = self.lower_bound * torch.sigmoid(decay_rate * gate_input)
        else:
            gate = -decay_rate * F.softplus(gate_input)
        return gate.to(hidden_states.dtype)


class BailingHybridDeltaAttention(Glm5NextTextLinearAttention):
    """Linear attention module."""

    def __init__(self, config: BailingHybridConfig, layer_idx: int):
        super().__init__()
        self.forget_gate = BailingHybridForgetGate(config)
        self.g_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.o_norm = BailingHybridRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        del self.g_a_proj
        del self.g_b_proj
        del self.layer_norm_epsilon
        del self.layer_type

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)], dim=-1
        ).transpose(1, 2)

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.qkv_dim] * 3, dim=-1)
        query, key, value = [states.view(hidden_shape) for states in (query, key, value)]

        decay = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))
        if use_precomputed_states and seq_len == 1:
            output, last_recurrent_state = recurrent_kimi_delta_attention(
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
            output, last_recurrent_state = chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        output_gate = self.g_proj(hidden_states)
        output = self.o_norm(output, output_gate.view(hidden_shape)).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class BailingHybridTopkRouter(DeepseekV3TopkRouter):
    pass


class BailingHybridMoE(DeepseekV3MoE):
    pass


class BailingHybridDecoderLayer(KimiLinearDecoderLayer):
    pass


@auto_docstring
class BailingHybridPreTrainedModel(Qwen3NextPreTrainedModel):
    _no_split_modules = ["BailingHybridDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.42\..*"]
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "conv1d",
        "dt_bias",
        "A_log",
    ]
    _can_record_outputs = {
        "router_logits": OutputRecorder(BailingHybridTopkRouter, index=0),
        "hidden_states": BailingHybridDecoderLayer,
        "attentions": BailingHybridAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, BailingHybridForgetGate):
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))
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
    pass


class BailingHybridForCausalLM(DeepseekV3ForCausalLM, GenerationMixin):
    _tied_weights_keys = {}


__all__ = [
    "BailingHybridConfig",
    "BailingHybridPreTrainedModel",
    "BailingHybridModel",
    "BailingHybridForCausalLM",
]
