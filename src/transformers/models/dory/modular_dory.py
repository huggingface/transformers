# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Dory: a looped normalized transformer (nGPT, https://huggingface.co/papers/2410.01131)."""

from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_dory import DoryConfig


def l2_normalize(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states_fp32 = hidden_states.float()
    normalized = hidden_states_fp32 * torch.rsqrt(hidden_states_fp32.pow(2).sum(-1, keepdim=True) + 1e-6)
    return normalized.to(hidden_states.dtype)


def normalized_residual_update(
    hidden_states: torch.Tensor, update: torch.Tensor, alpha: torch.Tensor, alpha_scale: float
) -> torch.Tensor:
    """nGPT residual: move `hidden_states` towards `update` on the unit sphere, at a learned per-channel rate."""
    hidden_states_fp32 = l2_normalize(hidden_states).float()
    update_fp32 = l2_normalize(update).float()
    interpolation_rate = (alpha.float() * alpha_scale).abs()
    return l2_normalize(hidden_states_fp32 + interpolation_rate * (update_fp32 - hidden_states_fp32)).to(
        hidden_states.dtype
    )


class DoryRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: DoryConfig, device=None):
        super().__init__(config)

    def compute_default_rope_parameters(
        config: DoryConfig, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        return super().compute_default_rope_parameters(config, layer_type)


class DoryAttention(LlamaAttention):
    """Llama attention with normalized queries/keys (scaled by `sqk`), a sigmoid output gate, and one cache slot per loop."""

    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.g_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.sqk = nn.Parameter(torch.full((config.num_attention_heads * self.head_dim,), config.initializer_range))
        self.s_gate = nn.Parameter(torch.full((config.num_attention_heads * self.head_dim,), config.initializer_range))
        self.sqk_scale = self.head_dim**0.5 / config.initializer_range
        self.s_gate_scale = config.hidden_size**0.25 / config.initializer_range
        # CODEPATH: every 4th layer is full attention in all released checkpoints, the others slide
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        loop_idx: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        gate = self.g_proj(hidden_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        sqk = (self.sqk.float() * self.sqk_scale).view(1, -1, 1, self.head_dim)
        query_states = (sqk.square() * l2_normalize(query_states).float()).to(query_states.dtype)
        key_states = l2_normalize(key_states)

        if past_key_values is not None:
            # The recurrent layers run once per loop, each loop has its own cache slot
            cache_idx = self.layer_idx + loop_idx * self.config.num_hidden_layers
            key_states, value_states = past_key_values.update(key_states, value_states, cache_idx)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        s_gate = (self.s_gate.float() * self.s_gate_scale).square()
        attn_output = (attn_output.float() * torch.sigmoid(gate.float() * s_gate)).to(attn_output.dtype)
        attn_output = self.o_proj(l2_normalize(attn_output))
        return attn_output, attn_weights


class DoryMLP(LlamaMLP):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.suv_gate = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_scale = config.hidden_size**0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).float() * (self.suv_gate.float() * self.suv_scale)
        up = self.up_proj(hidden_states).float() * (self.suv_up.float() * self.suv_scale)
        hidden_states = (self.act_fn(gate) * up).to(hidden_states.dtype)
        return self.down_proj(l2_normalize(hidden_states))


class DoryDecoderLayer(LlamaDecoderLayer):
    """Llama layer where nGPT normalized updates (`lambda_1`, `lambda_2`) replace norms + residual adds."""

    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]
        self.lambda_1 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_2 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.alpha_scale = config.ngpt_alpha_init / config.initializer_range
        del self.input_layernorm
        del self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = normalized_residual_update(hidden_states, attn_output, self.lambda_1, self.alpha_scale)
        mlp_output = self.mlp(hidden_states)
        hidden_states = normalized_residual_update(hidden_states, mlp_output, self.lambda_2, self.alpha_scale)
        return hidden_states


class DoryLMHead(nn.Linear):
    """`lm_head` with a learned per-token logit scale."""

    def __init__(self, config: DoryConfig):
        super().__init__(config.hidden_size, config.vocab_size, bias=False)
        self.logit_scale = nn.Parameter(torch.full((config.vocab_size,), config.initializer_range))
        self.logit_scale_factor = 1 / config.initializer_range

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = super().forward(hidden_states).float()
        return logits * (self.logit_scale.float() * self.logit_scale_factor)


class DoryPreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DoryRotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq, _ = module.compute_default_rope_parameters(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)
        elif isinstance(module, DoryAttention):
            init.constant_(module.sqk, self.config.initializer_range)
            init.constant_(module.s_gate, self.config.initializer_range)
        elif isinstance(module, DoryMLP):
            init.ones_(module.suv_gate)
            init.ones_(module.suv_up)
        elif isinstance(module, DoryDecoderLayer):
            init.constant_(module.lambda_1, self.config.initializer_range)
            init.constant_(module.lambda_2, self.config.initializer_range)
        elif isinstance(module, DoryLMHead):
            init.constant_(module.logit_scale, self.config.initializer_range)


class DoryModel(LlamaModel):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        del self.norm

    def create_cache(self) -> DynamicCache:
        """One cache slot per layer and per loop: the recurrent layers attend to their own history in each loop."""
        cache = DynamicCache(config=self.config)
        for _ in range(self.config.n_recurrent_loops - 1):
            cache.layers.extend(DynamicCache(config=self.config).layers)
        return cache

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
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = self.create_cache()

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
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        layer_inputs = {
            layer_type: {
                "attention_mask": causal_mask_mapping[layer_type],
                "position_embeddings": self.rotary_emb(inputs_embeds, position_ids, layer_type),
            }
            for layer_type in set(self.config.layer_types)
        }
        recurrent_start = self.config.n_input_layers
        recurrent_end = recurrent_start + self.config.n_recurrent_layers

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[:recurrent_start]:
            hidden_states = decoder_layer(
                hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.attention_type], **kwargs
            )

        input_hidden_states = hidden_states
        for loop_idx in range(self.config.n_recurrent_loops):
            hidden_states = l2_normalize(hidden_states + input_hidden_states)
            for decoder_layer in self.layers[recurrent_start:recurrent_end]:
                hidden_states = decoder_layer(
                    hidden_states,
                    past_key_values=past_key_values,
                    loop_idx=loop_idx,
                    **layer_inputs[decoder_layer.attention_type],
                    **kwargs,
                )
            hidden_states = l2_normalize(hidden_states)

        for decoder_layer in self.layers[recurrent_end:]:
            hidden_states = decoder_layer(
                hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.attention_type], **kwargs
            )

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DoryForCausalLM(LlamaForCausalLM):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.lm_head = DoryLMHead(config)

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        # `DynamicCache(config)` has one slot per layer, Dory needs one per layer and per loop
        if (
            generation_config.use_cache
            and model_kwargs.get("past_key_values") is None
            and generation_config.cache_implementation in (None, "dynamic")
        ):
            model_kwargs["past_key_values"] = self.model.create_cache()
        else:
            super()._prepare_cache_for_generation(generation_config, model_kwargs, *args, **kwargs)


__all__ = ["DoryPreTrainedModel", "DoryModel", "DoryForCausalLM"]
