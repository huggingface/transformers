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
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations.accelerate import force_accelerate_hooks
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
    DeepseekV3TopkRouter,
)
from ...models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32DecoderLayer
from ...models.glm5_next.modeling_glm5_next import (
    Glm5NextTextForgetGate,
    Glm5NextTextLinearAttention,
    Glm5NextTextRMSNormGated,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    recurrent_kimi_delta_attention,
)
from ...models.llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ...models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextPreTrainedModel,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder


@auto_docstring(checkpoint="moonshotai/Kimi-Linear-48B-A3B-Instruct")
@strict
class KimiLinearConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 8):
        Number of groups for routed experts.
    mlp_layer_types (`list[str]`, *optional*):
        List of layer types for the MLP or MoE layers. Defaults to None.
    linear_head_dim (`int`, *optional*):
        Dimension of each head in linear attention layers. Defaults to 128.
    linear_num_heads (`int`, *optional*):
        Number of heads for the linear attention layers. Defaults to 32.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size for the short convolution applied to queries, keys, and values in linear attention layers.
    mla_use_nope (`bool`, *optional*, defaults to `False`):
        Whether the MLA full-attention layers operate in pure NoPE mode (no rotary positional encoding).
        When `True`, ``qk_rope_head_dim`` is expected to be 0.
    mla_use_output_gate (`bool`, *optional*, defaults to `False`):
        Whether the MLA full-attention layers apply a learned sigmoid output gate after the attention
        projection, as used in Kimi K3.
    attn_res_block_size (`int`, *optional*, defaults to 0):
        Block size for Block Attention Residuals (AttnRes). When > 0, layers are grouped into blocks of
        this many transformer layers and each sub-layer input is attended over completed block
        representations instead of the standard additive residual. Set to 0 to disable.
    dense_ffn_hidden (`int`, *optional*):
        Hidden dimension for dense (non-MoE) FFN layers. Falls back to ``intermediate_size`` when unset.
    activation_situ_beta (`float`, *optional*, defaults to 1.0):
        Gate-projection beta for the SiTU activation ``gate * sigmoid(beta * gate)``. Only used
        when ``hidden_act == "situ"``.
    activation_situ_linear_beta (`float`, *optional*, defaults to 1.0):
        Up-projection beta for the SiTU activation. Only used when ``hidden_act == "situ"``.
    """

    model_type = "kimi_linear"
    attribute_map = {
        "max_position_embeddings": "model_max_length",
        "norm_topk_prob": "moe_renormalize",
        "n_group": "num_expert_group",
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_token",
        "n_shared_experts": "num_shared_experts",
    }

    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216
    moe_intermediate_size: int = 1024
    num_hidden_layers: int = 27
    num_local_experts: int = 256
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    routed_scaling_factor: float = 2.446
    q_lora_rank: int | None = None
    n_group: int = 1
    mlp_layer_types: list[str] | None = None
    topk_group: int | None = 1
    norm_topk_prob: bool = True
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 163839
    bos_token_id: int | None = 163584
    eos_token_id: int | list[int] | None = 163586
    layer_types: list[str] | None = None

    linear_head_dim: int = 128
    linear_num_heads: int = 32
    linear_conv_kernel_dim: int = 4

    # Kimi K3 / Kimi Linear extensions
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False
    attn_res_block_size: int = 0
    dense_ffn_hidden: int | None = None
    activation_situ_beta: float = 1.0
    activation_situ_linear_beta: float = 1.0

    rope_parameters = AttributeError()
    rope_interleave = AttributeError()
    first_k_dense_replace = AttributeError()
    num_mtp_layers = AttributeError()
    n_routed_experts = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Checkpoint stores linear attention attributes in a config sub-dict
        linear_attn_config = kwargs.get("linear_attn_config", {})
        self.linear_head_dim = linear_attn_config.get("head_dim", self.linear_head_dim)
        self.linear_num_heads = linear_attn_config.get("num_heads", self.linear_num_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)

        # Infer NoPE mode from qk_rope_head_dim == 0 when not explicit
        if not self.mla_use_nope and self.qk_rope_head_dim == 0:
            self.mla_use_nope = True

        # Layer types: checkpoint config > explicit layer_types > default
        if self.layer_types is None:
            if "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
                layer_types = [None] * self.num_hidden_layers
                for layer in linear_attn_config["full_attn_layers"]:
                    layer_types[layer - 1] = "full_attention"  # 1-indexed
                for layer in linear_attn_config["kda_layers"]:
                    layer_types[layer - 1] = "linear_attention"
                self.layer_types = layer_types
            else:
                self.layer_types = [
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(self.num_hidden_layers)
                ]

        # MLP layer types: dense vs sparse (MoE)
        if self.mlp_layer_types is None:
            first_k_dense_replace = kwargs.get("first_k_dense_replace", 1)
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]

        # dense_ffn_hidden falls back to intermediate_size
        if self.dense_ffn_hidden is None:
            self.dense_ffn_hidden = self.intermediate_size


class KimiLinearRMSNorm(LlamaRMSNorm):
    pass


class KimiLinearRMSNormGated(Glm5NextTextRMSNormGated):
    pass


class KimiLinearExperts(DeepseekV3Experts):
    pass


class KimiLinearMLP(DeepseekV3MLP):
    """Dense FFN for Kimi Linear / Kimi K3.

    Subclasses DeepseekV3MLP and overrides only ``intermediate_size`` (uses
    ``config.dense_ffn_hidden``) and ``forward`` (adds the paired-beta SiTU path).
    """

    def __init__(self, config: "KimiLinearConfig", intermediate_size: int | None = None):
        super().__init__(config, intermediate_size=intermediate_size or config.dense_ffn_hidden)
        if config.hidden_act == "situ":
            self.situ_gate_beta = config.activation_situ_beta
            self.situ_up_beta = config.activation_situ_linear_beta

    def forward(self, x):
        if self.config.hidden_act == "situ":
            gate = self.gate_proj(x).float()
            gate_out = self.situ_gate_beta * torch.tanh(gate / self.situ_gate_beta) * torch.sigmoid(gate)
            up = self.up_proj(x).float()
            up_out = self.situ_up_beta * torch.tanh(up / self.situ_up_beta)
            return self.down_proj((gate_out * up_out).to(x.dtype))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class KimiLinearAttention(DeepseekV3Attention):
    """MLA for Kimi Linear / Kimi K3.

    Extends DeepseekV3Attention with:
    - Pure NoPE mode (``mla_use_nope=True``): ``qk_rope_head_dim`` is 0, no RoPE applied.
    - Output gate (``mla_use_output_gate=True``): learned sigmoid gate on attention output.
    """

    def __init__(self, config: "KimiLinearConfig", layer_idx: int):
        super().__init__(config, layer_idx)
        self.use_nope = config.mla_use_nope
        self.use_output_gate = config.mla_use_output_gate
        # Override layernorms to use the Kimi variant
        self.kv_a_layernorm = KimiLinearRMSNorm(config.kv_lora_rank)
        if self.q_lora_rank is not None:
            self.q_a_layernorm = KimiLinearRMSNorm(self.q_lora_rank)
        if self.use_output_gate:
            self.o_gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)
        # NoPE: plain dot-product scaling (no yarn mscale)
        self.scaling = self.qk_head_dim ** (-0.5)

    def expand_kv(self, kv_nope: torch.Tensor, k_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, seq_length, _ = kv_nope.shape
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if self.use_nope:
            return k_nope.contiguous(), value_states
        k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)
        key_states = kv_nope.new_empty(*kv_nope.shape[:-1], self.qk_nope_head_dim + self.qk_rope_head_dim)
        key_states[..., : self.qk_nope_head_dim].copy_(k_nope)
        key_states[..., self.qk_nope_head_dim :].copy_(k_rot)
        return key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
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
        query_states = q_states.view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        if self.use_nope:
            kv_nope = self.kv_a_layernorm(compressed_kv)
            k_rot = compressed_kv.new_empty(batch_size, 1, seq_length, 0)
        else:
            kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_nope = self.kv_a_layernorm(kv_nope)
            k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        if past_key_values is not None:
            kv_nope, k_rot = past_key_values.update(kv_nope, k_rot, self.layer_idx)

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

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_output_gate:
            attn_output = attn_output * torch.sigmoid(self.o_gate_proj(hidden_states))
        return self.o_proj(attn_output), attn_weights


class KimiLinearForgetGate(Glm5NextTextForgetGate):
    """Same as Glm5NextTextForgetGate but with no gate_lower_bound and no A_log reshape."""

    def __init__(self, config: "KimiLinearConfig"):
        super().__init__(config)
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1))
        del self.safe_gate_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        decay_rate = torch.exp(self.A_log.float())
        # Softplus with upper bound to avoid overflows (Softplus(x) ≈ x for x > 20)
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        return -decay_rate * g_softplus


class KimiLinearDeltaAttention(Glm5NextTextLinearAttention):
    """Kimi Linear Attention (KDA): gated delta net with per-channel rather than per-token decay."""

    def __init__(self, config: "KimiLinearConfig", layer_idx: int):
        super().__init__(config, layer_idx)
        self.forget_gate = KimiLinearForgetGate(config)
        self.o_norm = KimiLinearRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Zero out padding
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # Single token decode path — skip when record_past is set to avoid mutating cache during rollback
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
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
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.qkv_dim] * 3,
            dim=-1,
        )

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # Single token decode path — same guard as conv to stay consistent
        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            core_attn_out, last_recurrent_state = recurrent_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)


class KimiLinearTopkRouter(DeepseekV3TopkRouter):
    pass


class KimiLinearMoE(DeepseekV3MoE):
    pass


def _attn_res_forward(
    proj: nn.Linear,
    norm: "KimiLinearRMSNorm",
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
) -> torch.Tensor:
    """Attend over completed block reps + current partial sum (Block AttnRes).

    Args:
        proj:          Linear(hidden_size, 1) — learned pseudo-query weight.
        norm:          RMSNorm for key normalisation.
        blocks:        N completed block representations, each [B, T, D].
        partial_block: current intra-block accumulator [B, T, D].
    Returns:
        Attended hidden state [B, T, D].
    """
    sources = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    keys = norm(sources)
    logits = torch.einsum("d,nbtd->nbt", proj.weight.squeeze(0), keys)  # [N+1, B, T]
    weights = torch.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, sources)


class KimiLinearDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: "KimiLinearConfig", layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_type = config.layer_types[layer_idx]
        # Select the correct attention implementation for this layer
        self.self_attn = (
            # CODEPATH: TODO: remove this once the mlinter rule is relaxed
            KimiLinearAttention(config, layer_idx)
            if config.layer_types[layer_idx] == "full_attention"
            else KimiLinearDeltaAttention(config, layer_idx)
        )
        self.attn_res_block_size = config.attn_res_block_size
        if self.attn_res_block_size > 0:
            # Named to match checkpoint weight keys directly.
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.self_attention_res_norm = KimiLinearRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
            self.mlp_res_norm = KimiLinearRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        attn_res_partial: torch.Tensor | None = None,
        attn_res_blocks: list[torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Returns ``(hidden_states, updated_partial)`` where ``updated_partial`` is
        ``attn_res_partial + attn_out + mlp_out`` when AttnRes is active, else ``None``.
        The model loop owns the ``attn_res_blocks`` list and the running partial; it
        pre-computes the attended residual tensors and passes them in, so the layer
        never mutates shared state.
        """
        use_attn_res = self.attn_res_block_size > 0

        # ---- Attention sub-layer ----
        if use_attn_res:
            h_attn = _attn_res_forward(
                self.self_attention_res_proj, self.self_attention_res_norm, attn_res_blocks, attn_res_partial
            )
            h_attn_in = self.input_layernorm(h_attn)
        else:
            h_attn_in = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            attn_out = self.self_attn(
                hidden_states=h_attn_in,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            attn_out, _ = self.self_attn(
                hidden_states=h_attn_in,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if not use_attn_res:
            hidden_states = hidden_states + attn_out

        # ---- MLP sub-layer ----
        if use_attn_res:
            # partial_after_attn is local — no mutation of the caller's tensor
            partial_after_attn = attn_res_partial + attn_out
            h_mlp = _attn_res_forward(self.mlp_res_proj, self.mlp_res_norm, attn_res_blocks, partial_after_attn)
            mlp_out = self.mlp(self.post_attention_layernorm(h_mlp))
            updated_partial = partial_after_attn + mlp_out
            return updated_partial, updated_partial
        else:
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
            return hidden_states, None


@auto_docstring
class KimiLinearPreTrainedModel(Qwen3NextPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(KimiLinearTopkRouter, index=0),
        "hidden_states": KimiLinearDecoderLayer,
        "attentions": KimiLinearAttention,
    }
    _keys_to_ignore_on_load_unexpected = None

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, KimiLinearForgetGate):  # following FLA initialization
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))  # stable inverse softplus
        elif isinstance(module, KimiLinearExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, KimiLinearTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, KimiLinearRMSNormGated):
            init.ones_(module.weight)


@auto_docstring
class KimiLinearModel(Qwen3NextModel):
    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        del self.rotary_emb
        self.attn_res_block_size = config.attn_res_block_size

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
            # Use the cache seq length only when a pre-filled cache is supplied;
            # a freshly-created empty cache reports 0 tokens seen.
            past_seen_tokens = 0
            if past_key_values is not None:
                try:
                    past_seen_tokens = past_key_values.get_seq_length()
                except (ValueError, StopIteration):
                    past_seen_tokens = 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            position_ids = position_ids + past_seen_tokens

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

        # AttnRes: the model loop owns the block history and running partial.
        # Token embeddings serve as completed block 0.
        use_attn_res = self.attn_res_block_size > 0
        if use_attn_res:
            attn_res_blocks: list[torch.Tensor] = [hidden_states]
            attn_res_partial: torch.Tensor = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_kwargs = dict(
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            if use_attn_res:
                layer_kwargs["attn_res_partial"] = attn_res_partial
                layer_kwargs["attn_res_blocks"] = attn_res_blocks

            hidden_states, updated_partial = decoder_layer(hidden_states, **layer_kwargs)

            if use_attn_res:
                attn_res_partial = updated_partial
                # Block boundary: archive the accumulated partial, start fresh
                if (i + 1) % self.attn_res_block_size == 0:
                    attn_res_blocks.append(attn_res_partial)
                    attn_res_partial = torch.zeros_like(hidden_states)

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class KimiLinearForCausalLM(DeepseekV3ForCausalLM, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}


__all__ = [
    "KimiLinearConfig",
    "KimiLinearPreTrainedModel",
    "KimiLinearModel",
    "KimiLinearForCausalLM",
]
