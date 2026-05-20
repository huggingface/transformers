# Copyright 2026 Zyphra and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Zaya model."""

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import init

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..afmoe.modeling_afmoe import AfmoeForCausalLM
from ..laguna.configuration_laguna import LagunaConfig
from ..laguna.modeling_laguna import LagunaModel, LagunaRotaryEmbedding
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaPreTrainedModel, repeat_kv
from ..phi3.modeling_phi3 import Phi3Attention
from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts, Qwen3MoeRMSNorm


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
@strict
class ZayaConfig(LagunaConfig):
    r"""
    lm_head_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the language modeling head.
    router_hidden_size (`int`, *optional*, defaults to 256):
        Hidden size used by the ZAYA router.
    cca_time0 (`int`, *optional*, defaults to 2):
        First temporal parameter of the CCA projection.
    cca_time1 (`int`, *optional*, defaults to 2):
        Second temporal parameter of the CCA projection.

    ```python
    >>> from transformers import ZayaConfig, ZayaModel

    >>> configuration = ZayaConfig()
    >>> model = ZayaModel(configuration)

    >>> configuration = model.config
    ```
    """

    model_type = "zaya"

    vocab_size: int = 262272
    moe_intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    tie_word_embeddings: bool = True
    rms_norm_eps: float = 1e-5
    sliding_window: int | None = None
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 106

    num_experts_per_tok: int = 1
    num_experts: int = 16

    lm_head_bias: bool = False
    router_hidden_size: int = 256
    cca_time0: int = 2
    cca_time1: int = 2

    # Fields declared by LagunaConfig but not used by ZAYA.
    # TODO: add TP/PP plans. TP needs the router mlp, moe experts, and CCA projections to shard consistently; PP needs coverage for the cross-layer router state. For TP, see discussion https://github.com/huggingface/transformers/pull/45862#discussion_r3266709862
    base_model_tp_plan = AttributeError()
    base_model_pp_plan = AttributeError()
    intermediate_size = AttributeError()
    shared_expert_intermediate_size = AttributeError()
    router_aux_loss_coef = AttributeError()
    num_attention_heads_per_layer = AttributeError()
    mlp_layer_types = AttributeError()
    moe_routed_scaling_factor = AttributeError()
    moe_apply_router_weight_on_input = AttributeError()
    moe_router_logit_softcapping = AttributeError()

    def __post_init__(self, **kwargs):
        self.layer_types = ["hybrid"] * self.num_hidden_layers if self.layer_types is None else list(self.layer_types)

        default_rope_params: dict[Literal["hybrid", "hybrid_sliding"], dict[str, Any]] = {
            "hybrid": {
                "rope_type": "default",
                "rope_theta": 5_000_000.0,
                "partial_rotary_factor": 0.5,
            },
            "hybrid_sliding": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.5,
            },
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        PreTrainedConfig.__post_init__(self, **kwargs, ignore_keys_at_rope_validation={"hybrid", "hybrid_sliding"})

    def convert_rope_params_to_dict(self, **kwargs):
        # No legacy flat RoPE format is supported here; conversion writes the nested ZAYA layer-type format directly.
        return kwargs

    def validate_architecture(self):
        if self.num_experts_per_tok != 1:
            raise ValueError("ZAYA currently supports `num_experts_per_tok=1` only.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_attention_heads` must be a multiple of `num_key_value_heads`.")
        if "hybrid_sliding" in self.layer_types and self.sliding_window is None:
            raise ValueError("`sliding_window` must be set when `layer_types` contains `hybrid_sliding`.")


class ZayaRotaryEmbedding(LagunaRotaryEmbedding):
    pass


class ZayaRMSNorm(Qwen3MoeRMSNorm):
    pass


class ZayaCCAProjection(nn.Module):
    """
    Projects hidden states into attention q/k/v states with ZAYA's CCA path.

    This follows the usual q/k/v projection flow, with three ZAYA-specific changes: q/k are mixed by a causal 1D
    convolution, q/k keep residual projection paths, and v uses a delayed recurrent state.
    """

    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size

        self.depthwise_kernel_size = config.cca_time0
        self.grouped_kernel_size = config.cca_time1
        self.conv_kernel_size = (self.depthwise_kernel_size - 1) + (self.grouped_kernel_size - 1)

        self.num_key_value_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        query_hidden_size = self.num_attention_heads * self.head_dim
        key_value_hidden_size = self.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, query_hidden_size, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, key_value_hidden_size, bias=self.config.attention_bias)
        self.v_proj_current = nn.Linear(self.hidden_size, key_value_hidden_size // 2, bias=self.config.attention_bias)
        self.v_proj_delayed = nn.Linear(self.hidden_size, key_value_hidden_size // 2, bias=self.config.attention_bias)

        conv_channels = key_value_hidden_size + query_hidden_size
        self.conv_qk_depthwise = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.depthwise_kernel_size,
            groups=conv_channels,
            padding=0,
            stride=1,
        )
        self.conv_qk_grouped = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=self.grouped_kernel_size,
            groups=(self.num_key_value_heads + self.num_attention_heads),
            padding=0,
            stride=1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
        padding_mask: torch.Tensor | None = None,
    ):
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask[:, :, None].to(hidden_states.dtype)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        projected_queries = self.q_proj(hidden_states)
        projected_keys = self.k_proj(hidden_states)
        qk_states = torch.cat([projected_queries, projected_keys], dim=-1)

        query_residual = projected_queries.view(*hidden_shape)
        key_residual = projected_keys.view(*hidden_shape).transpose(1, 2)
        key_residual = repeat_kv(key_residual, self.num_key_value_groups).transpose(1, 2)
        query_residual = (query_residual + key_residual) * 0.5
        key_residual = query_residual.view(*input_shape, -1, self.num_key_value_groups, self.head_dim).mean(dim=-2)

        qk_states = qk_states.transpose(1, 2)
        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            cached_qk_states = past_key_values.layers[self.layer_idx].conv_states
            qk_states = torch.cat([cached_qk_states, qk_states], dim=-1)
        else:
            qk_states = F.pad(qk_states, (self.conv_kernel_size, 0))

        if past_key_values is not None:
            new_conv_state = qk_states[..., -self.conv_kernel_size :]
            new_conv_state = F.pad(new_conv_state, (self.conv_kernel_size - new_conv_state.shape[-1], 0))
            past_key_values.update_conv_state(new_conv_state, self.layer_idx)

        qk_states = self.conv_qk_depthwise(qk_states)
        qk_states = self.conv_qk_grouped(qk_states).transpose(1, 2)

        query_hidden_size = query_residual.shape[-2] * query_residual.shape[-1]
        query = qk_states[..., :query_hidden_size].view(*hidden_shape) + query_residual
        key = qk_states[..., query_hidden_size:].view(*hidden_shape) + key_residual

        # The value path carries half of each value head from the current token and half from the previous token.
        # During cached decoding, `recurrent_v_state` is the previous token's delayed projection.
        value_current = self.v_proj_current(hidden_states)
        delayed_v_state = self.v_proj_delayed(hidden_states)
        if use_precomputed_states:
            recurrent_v_state = past_key_values.layers[self.layer_idx].recurrent_states.unsqueeze(1)
        else:
            recurrent_v_state = self.v_proj_delayed(hidden_states.new_zeros(input_shape[0], 1, self.hidden_size))
        value_delayed = torch.cat([recurrent_v_state, delayed_v_state[:, :-1]], dim=1)

        if past_key_values is not None:
            past_key_values.update_recurrent_state(delayed_v_state[:, -1, :], self.layer_idx)

        value = torch.cat([value_current, value_delayed], dim=-1).view(*hidden_shape)

        return query, key, value


class ZayaQKNorm(nn.Module):
    """
    L2-normalizes q/k states to sqrt(head_dim) and applies ZAYA's learned per-KV-head key scale.
    """

    def __init__(self, config: ZayaConfig):
        super().__init__()
        self.head_dim_scale = config.head_dim**0.5
        self.temp = nn.Parameter(torch.zeros(config.num_key_value_heads))

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_eps = torch.finfo(query_states.dtype).eps
        query_states = query_states * (
            self.head_dim_scale / query_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)
        )
        key_states = key_states * (
            self.head_dim_scale / key_states.norm(p=2, dim=-1, keepdim=True).clamp_min(norm_eps)
        )
        key_states = key_states * self.temp[None, None, :, None]
        return query_states, key_states


class ZayaAttention(Phi3Attention):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del op_size  # noqa: F821
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "hybrid_sliding" else None
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.qk_norm = ZayaQKNorm(config)
        self.qkv_proj = ZayaCCAProjection(
            config=self.config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]

        mask_mapping = attention_mask or {}
        causal_mask = mask_mapping.get("causal")
        padding_mask = mask_mapping.get("padding")

        # ZAYA replaces the usual independent q/k/v projections with CCA projection followed by special q/k normalization.
        query_states, key_states, value_states = self.qkv_proj(hidden_states, past_key_values, padding_mask)
        query_states, key_states = self.qk_norm(query_states, key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            causal_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.view(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class ZayaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: ZayaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = ZayaSparseMoeBlock(config, layer_idx)
        self.post_attention_residual_scale = ZayaResidualScaling(config.hidden_size)
        self.post_mlp_residual_scale = ZayaResidualScaling(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
        attention_mask: dict[str, Any] | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual.to(dtype=self.input_layernorm.weight.dtype))

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        residual = self.post_attention_residual_scale(hidden_states, residual)
        hidden_states = self.post_attention_layernorm(residual.to(dtype=self.post_attention_layernorm.weight.dtype))

        hidden_states, prev_router_hidden_states = self.mlp(
            hidden_states,
            prev_router_hidden_states,
        )

        hidden_states = self.post_mlp_residual_scale(hidden_states, residual)

        return hidden_states, prev_router_hidden_states


class ZayaResidualScaling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_states_scale = nn.Parameter(torch.ones(hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(hidden_size))
        self.residual_scale = nn.Parameter(torch.ones(hidden_size))
        self.residual_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        # Keep the residual stream in fp32 to match the original ZAYA `residual_in_fp32` path.
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        residual = (residual + self.residual_bias) * self.residual_scale
        return hidden_states + residual


class ZayaRouterMLP(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, rms_norm_eps: float):
        super().__init__()
        self.norm = ZayaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.act_fn(self.fc1(hidden_states))
        hidden_states = self.act_fn(self.fc2(hidden_states))
        return self.out_proj(hidden_states)


class ZayaRouter(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.num_experts = config.num_experts + 1
        self.top_k = config.num_experts_per_tok
        self.router_hidden_size = config.router_hidden_size

        self.down_proj = nn.Linear(self.hidden_size, self.router_hidden_size, bias=True)

        self.use_eda = self.layer_idx != 0
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.router_hidden_size))

        self.router_mlp = ZayaRouterMLP(self.router_hidden_size, self.num_experts, config.rms_norm_eps)

        self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
        self.balancing_biases[-1] = -1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        final_shape = (-1, self.top_k)
        seq_length = hidden_states.shape[1]

        router_hidden_states = self.down_proj(hidden_states)

        if self.use_eda and router_states is not None:
            router_hidden_states = router_hidden_states + router_states * self.router_states_scale

        router_hidden_states_next = router_hidden_states[:, -seq_length:].clone()
        router_logits = self.router_mlp(router_hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1)

        biased_router_probs = router_probs.detach().to(torch.float32) + self.balancing_biases
        _, router_indices = torch.topk(biased_router_probs, self.top_k, dim=-1)
        router_probs = torch.gather(router_probs, dim=2, index=router_indices)

        # If the router selects the extra skip expert, mask it before `ZayaExperts` builds its one-hot expert mask.
        skip_expert = router_indices == self.config.num_experts
        router_probs = router_probs.masked_fill(skip_expert, 0)
        router_indices = router_indices.masked_fill(skip_expert, 0)

        return (
            router_logits.reshape(-1, self.num_experts),
            router_probs.reshape(final_shape),
            router_indices.reshape(final_shape),
            router_hidden_states_next,
        )


class ZayaExperts(Qwen3MoeExperts):
    pass


class ZayaSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.gate = ZayaRouter(config, layer_idx)
        self.experts = ZayaExperts(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, router_probs, router_indices, prev_router_hidden_states = self.gate(
            hidden_states, router_states=prev_router_hidden_states
        )

        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(batch_size * seq_length, emb_dim)
        expert_output = self.experts(hidden_states_flat, router_indices, router_probs)
        expert_output = expert_output.view(batch_size, seq_length, emb_dim)

        return expert_output, prev_router_hidden_states


class ZayaPreTrainedModel(LlamaPreTrainedModel):
    config: ZayaConfig
    # ZAYA generation uses the native hybrid dynamic cache, which is not a compileable cache.
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(ZayaRouter, index=0),
        "hidden_states": ZayaDecoderLayer,
        "attentions": ZayaAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ZayaResidualScaling):
            init.ones_(module.hidden_states_scale)
            init.zeros_(module.hidden_states_bias)
            init.ones_(module.residual_scale)
            init.zeros_(module.residual_bias)
        elif isinstance(module, ZayaModel):
            init.ones_(module.input_hidden_states_scale)
            init.zeros_(module.input_hidden_states_bias)
        elif isinstance(module, ZayaQKNorm):
            init.zeros_(module.temp)
        elif isinstance(module, ZayaRouter):
            if module.use_eda:
                init.ones_(module.router_states_scale)
            init.zeros_(module.balancing_biases)
            module.balancing_biases[-1] = -1.0  # trf-ignore: TRF012
        elif isinstance(module, ZayaExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ZayaRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                getattr(module, f"{layer_type}_inv_freq").copy_(curr_inv_freq)
                getattr(module, f"{layer_type}_original_inv_freq").copy_(curr_inv_freq)


@auto_docstring
class ZayaModel(LagunaModel):
    def __init__(self, config: ZayaConfig):
        super().__init__(config)
        self.input_hidden_states_scale = nn.Parameter(torch.ones(config.hidden_size))
        self.input_hidden_states_bias = nn.Parameter(torch.zeros(config.hidden_size))

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
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

        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "ZAYA CCA projection requires a 2D `attention_mask` to mask padding tokens before convolution."
            )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        mask_creation_functions = {
            "hybrid": lambda: create_causal_mask(**mask_kwargs),
            "hybrid_sliding": lambda: create_sliding_window_causal_mask(**mask_kwargs),
        }
        causal_mask_mapping = {
            layer_type: mask_creation_functions[layer_type]() for layer_type in set(self.config.layer_types)
        }
        cca_mask = self._update_cca_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        # Keep the residual stream in fp32 to match the original ZAYA `residual_in_fp32` path.
        hidden_states = ((hidden_states + self.input_hidden_states_bias) * self.input_hidden_states_scale).to(
            torch.float32
        )

        prev_router_hidden_states = None

        for idx, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[idx]
            hidden_states, prev_router_hidden_states = decoder_layer(
                hidden_states,
                prev_router_hidden_states,
                attention_mask={"causal": causal_mask_mapping[layer_type], "padding": cca_mask},
                past_key_values=past_key_values,
                position_embeddings=position_embeddings[layer_type],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _update_cca_mask(self, attention_mask, past_key_values):
        """
        No need to zero padding states when cached convolution states are already available or all inputs are valid.
        """
        cca_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            cca_mask = None
        return cca_mask


@auto_docstring(checkpoint="Zyphra/ZAYA1-8B")
class ZayaForCausalLM(AfmoeForCausalLM, ZayaPreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _is_stateful = True

    _tp_plan = AttributeError()
    _pp_plan = AttributeError()

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)


__all__ = [
    "ZayaConfig",
    "ZayaPreTrainedModel",
    "ZayaModel",
    "ZayaForCausalLM",
]
