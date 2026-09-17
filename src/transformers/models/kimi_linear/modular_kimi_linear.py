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
from ...activations import ACT2FN, SiTUActivation
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ...models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Experts,
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
)
from ...models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32DecoderLayer
from ...models.glm5_next.modeling_glm5_next import (
    Glm5NextTextForgetGate,
    Glm5NextTextLinearAttention,
    Glm5NextTextRMSNormGated,
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

        # Guard token IDs that were inherited from the production checkpoint defaults but are
        # outside the configured vocabulary.  This allows constructing small-vocab configs
        # (e.g. for testing) without hitting nn.Embedding's padding_idx assertion.
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            val = getattr(self, attr, None)
            if val is not None and isinstance(val, int) and val >= self.vocab_size:
                setattr(self, attr, None)


class KimiLinearRMSNorm(LlamaRMSNorm):
    pass


class KimiLinearRMSNormGated(Glm5NextTextRMSNormGated):
    pass


class KimiLinearExperts(DeepseekV3Experts):
    pass


class KimiLinearMLP(nn.Module):
    """Dense FFN for Kimi Linear / Kimi K3.

    Uses ``config.dense_ffn_hidden`` as the intermediate dimension (falls back to
    ``config.intermediate_size``). When ``hidden_act == "situ"``, uses separate SiTU
    activations with per-projection betas.
    """

    def __init__(self, config: "KimiLinearConfig", intermediate_size: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.dense_ffn_hidden
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.hidden_act == "situ":
            self.gate_act = SiTUActivation(beta=config.activation_situ_beta)
            self.up_act = SiTUActivation(beta=config.activation_situ_linear_beta)
        else:
            act = ACT2FN[config.hidden_act]
            self.gate_act = act
            self.up_act = act

    def forward(self, x):
        return self.down_proj(self.gate_act(self.gate_proj(x)) * self.up_act(self.up_proj(x)))


class KimiLinearAttention(nn.Module):
    """Multi-headed Latent Attention (MLA) for Kimi Linear / Kimi K3.

    Extends the DeepSeek V3 MLA with:
    - Pure NoPE mode (``mla_use_nope=True``): no rotary embeddings; ``qk_rope_head_dim``
      must be 0.
    - Output gate (``mla_use_output_gate=True``): a learned sigmoid gate is applied to
      the attention output before ``o_proj``.
    """

    def __init__(self, config: "KimiLinearConfig", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.use_nope = config.mla_use_nope
        self.use_output_gate = config.mla_use_output_gate
        self.is_causal = True

        self.q_proj = (
            None
            if self.q_lora_rank is not None
            else nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        )
        self.q_a_proj = (
            nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            if self.q_lora_rank is not None
            else None
        )
        self.q_a_layernorm = KimiLinearRMSNorm(self.q_lora_rank) if self.q_lora_rank is not None else None
        self.q_b_proj = (
            nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
            if self.q_lora_rank is not None
            else None
        )

        # In NoPE mode qk_rope_head_dim == 0; projection only outputs the compressed latent.
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = KimiLinearRMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        if self.use_output_gate:
            self.o_gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)

    def expand_kv(self, kv_nope: torch.Tensor, k_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand compressed latents into full key and value states."""
        batch_size, _, seq_length, _ = kv_nope.shape
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        if self.use_nope:
            # Pure NoPE: key == k_nope only, no RoPE concatenation
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
            # qk_rope_head_dim == 0: whole output is the latent KV
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


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


class KimiLinearTopkRouter(DeepseekV3TopkRouter):
    pass


class KimiLinearMoE(DeepseekV3MoE):
    pass


class KimiLinearAttnRes(nn.Module):
    """Block Attention Residuals (Block AttnRes) from the Kimi K3 architecture.

    Replaces the standard additive residual with learned softmax attention over completed
    block representations plus the current partial-block sum.  See arXiv:2603.15031.
    """

    def __init__(self, config: "KimiLinearConfig"):
        super().__init__()
        # Single learned pseudo-query weight per sub-layer projection
        self.proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = KimiLinearRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor) -> torch.Tensor:
        """Attend over completed block reps + current partial sum.

        Args:
            blocks:        N completed block representations, each [B, T, D]
            partial_block: current intra-block accumulator [B, T, D]
        Returns:
            Attended hidden state [B, T, D]
        """
        sources = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
        keys = self.norm(sources)
        logits = torch.einsum("d,nbtd->nbt", self.proj.weight.squeeze(0), keys)  # [N+1, B, T]
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
            self.attn_res_attn = KimiLinearAttnRes(config)
            self.attn_res_mlp = KimiLinearAttnRes(config)

    @property
    def attn_res(self) -> "KimiLinearAttnRes | None":
        """Convenience accessor: returns the pre-attention AttnRes module, or None when disabled."""
        return self.attn_res_attn if self.attn_res_block_size > 0 else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        attn_res_blocks: list[torch.Tensor] | None = None,
        attn_res_partial: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """Forward pass.  Returns ``hidden_states`` only.

        When AttnRes is enabled (``attn_res_block_size > 0``), ``attn_res_blocks`` and
        ``attn_res_partial`` are modified in-place so the model loop can track state
        without changing the return signature expected by gradient checkpointing.
        """
        use_attn_res = self.attn_res_block_size > 0

        # ---- Attention sub-layer ----
        if use_attn_res:
            h = self.attn_res_attn(attn_res_blocks, attn_res_partial)
            h_attn_in = self.input_layernorm(h)
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

        if use_attn_res:
            # In-place update so the model loop sees the change via the same tensor reference
            attn_res_partial.add_(attn_out)
        else:
            hidden_states = hidden_states + attn_out

        # ---- MLP sub-layer ----
        if use_attn_res:
            h = self.attn_res_mlp(attn_res_blocks, attn_res_partial)
            mlp_out = self.mlp(self.post_attention_layernorm(h))
            attn_res_partial.add_(mlp_out)
            hidden_states = attn_res_partial
        else:
            hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


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

        # AttnRes state: completed block reps list + current partial-block sum.
        # Token embeddings serve as block 0.
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
                layer_kwargs["attn_res_blocks"] = attn_res_blocks
                layer_kwargs["attn_res_partial"] = attn_res_partial

            hidden_states = decoder_layer(hidden_states, **layer_kwargs)

            if use_attn_res:
                # Block boundary: save accumulated partial, reset for next block
                if (i + 1) % self.attn_res_block_size == 0:
                    attn_res_blocks.append(attn_res_partial.clone())
                    attn_res_partial.zero_()

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
