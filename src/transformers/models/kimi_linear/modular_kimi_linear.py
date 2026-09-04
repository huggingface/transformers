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

    rope_parameters = AttributeError()
    rope_interleave = AttributeError()
    first_k_dense_replace = AttributeError()
    num_mtp_layers = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # Checkpoint stores linear attention attributes in a config sub-dict: if it's there, extract them
        linear_attn_config = kwargs.get("linear_attn_config", {})
        self.linear_head_dim = linear_attn_config.get("head_dim", self.linear_head_dim)
        self.linear_num_heads = linear_attn_config.get("num_heads", self.linear_num_heads)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", self.linear_conv_kernel_dim)

        # For layer types, the precedence is: checkpoint config > layer types > default
        if self.layer_types is None:
            if "full_attn_layers" in linear_attn_config and "kda_layers" in linear_attn_config:
                layer_types = [None] * self.num_hidden_layers
                for layer in linear_attn_config["full_attn_layers"]:
                    layer_types[layer - 1] = "full_attention"  # types are 1-indexed in the checkpoint
                for layer in linear_attn_config["kda_layers"]:
                    layer_types[layer - 1] = "linear_attention"
                self.layer_types = layer_types
            else:
                self.layer_types = [
                    "full_attention" if i and i % 4 == 0 else "linear_attention" for i in range(self.num_hidden_layers)
                ]

        # Same for MLP layer types, which indicate MLP or MoE
        if self.mlp_layer_types is None:
            first_k_dense_replace = kwargs.get("first_k_dense_replace", 1)
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]


class KimiLinearRMSNorm(LlamaRMSNorm):
    pass


class KimiLinearRMSNormGated(Glm5NextTextRMSNormGated):
    pass


class KimiLinearExperts(DeepseekV3Experts):
    pass


class KimiLinearMLP(DeepseekV3MLP):
    pass


class KimiLinearAttention(DeepseekV3Attention):
    """Multi-headed Latent Attention (MLA) from Deepseek V2 with NoPE, but the part of the keys where RoPE is applied is
    still shared."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query_states = q_states.view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_nope = self.kv_a_layernorm(kv_nope)
        # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
        kv_nope = kv_nope.view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Cache read / write is performed while latent KV is still compressed
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class KimiLinearForgetGate(Glm5NextTextForgetGate):
    """Same as Glm5NextTextForgetGate but with no gate_lower_bound and no A_log reshape."""

    def __init__(self, config: KimiLinearConfig):
        super().__init__(config)
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_heads, 1))
        del self.safe_gate_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float()
        decay_rate = torch.exp(A_log)

        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

        return -decay_rate * g_softplus


class KimiLinearDeltaAttention(Glm5NextTextLinearAttention):
    """Kimi Linear Attention: this is essentialy the same a gated delta net (GDN) but decay is per-channel instead of
    per-token."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.forget_gate = KimiLinearForgetGate(config)
        self.o_norm = KimiLinearRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)


class KimiLinearTopkRouter(DeepseekV3TopkRouter):
    pass


class KimiLinearMoE(DeepseekV3MoE):
    pass


class KimiLinearDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: KimiLinearConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_type = config.layer_types[layer_idx]
        self.self_attn = (
            KimiLinearAttention(config, layer_idx)
            if config.layer_types[layer_idx] == "full_attention"
            else KimiLinearDeltaAttention(config, layer_idx)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        # Self attn
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
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
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
            # A_log
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            # dt_bias
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))  # (stable) inverse softplus
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
            position_ids: torch.LongTensor = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = (position_ids + past_seen_tokens).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class KimiLinearForCausalLM(DeepseekV3ForCausalLM, GenerationMixin):
    _tied_weights_keys = {}


__all__ = [
    "KimiLinearConfig",
    "KimiLinearPreTrainedModel",
    "KimiLinearModel",
    "KimiLinearForCausalLM",
]
