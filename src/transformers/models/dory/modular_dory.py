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
from ...cache_utils import (
    Cache,
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
    StaticLayer,
    StaticSlidingWindowLayer,
)
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..afmoe.modeling_afmoe import AfmoeAttention
from ..gemma2.modeling_gemma2 import Gemma2DecoderLayer
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llama4.modeling_llama4 import Llama4TextL2Norm
from .configuration_dory import DoryConfig


class DoryL2Norm(Llama4TextL2Norm):
    """Projects onto the unit sphere: nGPT keeps every hidden state at norm 1, not RMS 1."""

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + self.eps)


class DoryRecurrentLayerMixin:
    """
    Cache of a recurrent layer, which runs once per loop. Each `update` passes the `loop_idx`:

        per_loop:  every loop keeps its own history, loop 0 in this layer and loops 1, 2, ... in `later_loops`
        last_loop: (`share_across_loops=True`) only the last loop is cached; earlier loops `read` the cache and
                   their own tokens without writing, like KV-shared layers, so memory does not grow with the loops
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_args, self.init_kwargs = args, kwargs
        self.later_loops = []

    def update(
        self,
        key_states,
        value_states,
        *args,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        share_across_loops: bool = False,
        **kwargs,
    ):
        if share_across_loops and not is_last_loop:
            return self.read(key_states, value_states)
        if share_across_loops or loop_idx == 0:
            return super().update(key_states, value_states, *args, **kwargs)
        while len(self.later_loops) < loop_idx:
            self.later_loops.append(type(self)(*self.init_args, **self.init_kwargs))
        return self.later_loops[loop_idx - 1].update(key_states, value_states, *args, **kwargs)

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        for layer in self.later_loops:
            layer.crop(tokens_to_remove)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        for layer in self.later_loops:
            layer.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        for layer in self.later_loops:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        for layer in self.later_loops:
            layer.batch_select_indices(indices)

    def reset(self) -> None:
        super().reset()
        self.later_loops = []


class DoryRecurrentLayer(DoryRecurrentLayerMixin, DynamicLayer):
    _layer_type = "recurrent_full_attention"

    def read(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            return key_states, value_states
        return torch.cat([self.keys, key_states], dim=-2), torch.cat([self.values, value_states], dim=-2)


class DoryRecurrentSlidingWindowLayer(DoryRecurrentLayerMixin, DynamicSlidingWindowLayer):
    _layer_type = "recurrent_sliding_attention"

    def read(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            return key_states, value_states
        # The cache may hold more than the window if past recording is on
        num_visible = self.sliding_window - 1 + key_states.shape[-2]
        key_states = torch.cat([self.keys, key_states], dim=-2)[..., -num_visible:, :]
        value_states = torch.cat([self.values, value_states], dim=-2)[..., -num_visible:, :]
        return key_states, value_states


class DoryStaticRecurrentLayer(DoryRecurrentLayerMixin, StaticLayer):
    _layer_type = "recurrent_full_attention"

    def read(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Write at the next positions without moving past them: the last loop overwrites the same slots
        kv_length = key_states.shape[-2]
        key_states, value_states = StaticLayer.update(self, key_states, value_states)
        self.cumulative_length.sub_(kv_length)
        return key_states, value_states


class DoryStaticRecurrentSlidingWindowLayer(DoryRecurrentLayerMixin, StaticSlidingWindowLayer):
    _layer_type = "recurrent_sliding_attention"

    def read(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Same states `update` returns, without rolling the window
        kv_length = key_states.shape[-2]
        current_length = self.cumulative_length_int
        if current_length >= self.max_cache_len:
            key_states = torch.cat([self.keys[:, :, 1:, :], key_states], dim=-2)
            value_states = torch.cat([self.values[:, :, 1:, :], value_states], dim=-2)
        elif current_length + kv_length > self.max_cache_len:
            if current_length > 0:
                key_states = torch.cat([self.keys[:, :, :current_length, :], key_states], dim=-2)
                value_states = torch.cat([self.values[:, :, :current_length, :], value_states], dim=-2)
        else:
            # Room left: write at the next positions without moving past them, the last loop overwrites them
            key_states, value_states = StaticSlidingWindowLayer.update(self, key_states, value_states)
            self.cumulative_length_int -= kv_length
            self.cumulative_length.sub_(kv_length)
        return key_states, value_states


class DoryRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: DoryConfig, device=None):
        super().__init__(config)

    def compute_default_rope_parameters(
        config: DoryConfig, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        return super().compute_default_rope_parameters(config, layer_type)


class DoryAttention(AfmoeAttention):
    """
    Afmoe attention with nGPT details: unit-norm queries and keys, queries rescaled per channel by `sqk` after rope,
    a learned scale `s_gate` inside the output gate, and a cache that follows the recurrent loops.
    """

    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = DoryL2Norm()
        self.k_norm = DoryL2Norm()
        self.attn_sub_norm = DoryL2Norm()
        self.sqk = nn.Parameter(torch.full((config.num_attention_heads * self.head_dim,), config.initializer_range))
        self.s_gate = nn.Parameter(torch.full((config.num_attention_heads * self.head_dim,), config.initializer_range))
        self.sqk_scale = self.head_dim**0.5 / config.initializer_range
        self.s_gate_scale = config.hidden_size**0.25 / config.initializer_range
        self.is_local_attention = config.layer_types[layer_idx].endswith("sliding_attention")
        self.sliding_window = config.sliding_window if self.is_local_attention else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        loop_idx: int = 0,
        is_last_loop: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_scale = (self.sqk * self.sqk_scale).view(-1, 1, self.head_dim)
        query_states = query_states * query_scale.square()

        if past_key_values is not None:
            # CODEPATH: released configs use `per_loop` (one cache per loop); `last_loop` (opt-in) only caches the last loop
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                loop_idx=loop_idx,
                is_last_loop=is_last_loop,
                share_across_loops=self.config.recurrent_kv_cache_mode == "last_loop",
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        output = output.view(*input_shape, -1).contiguous()
        output = output * torch.sigmoid(gate_states * (self.s_gate * self.s_gate_scale).square())
        attn_output = self.o_proj(self.attn_sub_norm(output))
        return attn_output, attn_weights


class DoryMLP(LlamaMLP):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.suv_gate = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_up = nn.Parameter(torch.ones(config.intermediate_size))
        self.suv_scale = config.hidden_size**0.5
        self.ffn_sub_norm = DoryL2Norm()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states) * (self.suv_gate * self.suv_scale)
        up = self.up_proj(hidden_states) * (self.suv_up * self.suv_scale)
        return self.down_proj(self.ffn_sub_norm(self.act_fn(gate) * up))


class DoryDecoderLayer(Gemma2DecoderLayer):
    """
    Gemma2 sandwich layer on the unit sphere: the residual is an interpolation towards the normalized update, at a
    learned per-channel rate (`lambda_1`, `lambda_2`), instead of an addition.
    """

    def __init__(self, config: DoryConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = DoryL2Norm()
        self.post_attention_layernorm = DoryL2Norm()
        self.pre_feedforward_layernorm = DoryL2Norm()
        self.post_feedforward_layernorm = DoryL2Norm()
        self.lambda_1 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_2 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_scale = config.ngpt_alpha_init / config.initializer_range

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            **kwargs,
        )
        attn_output = self.post_attention_layernorm(attn_output)
        attention_rate = (self.lambda_1 * self.lambda_scale).abs()
        hidden_states = hidden_states + attention_rate * (attn_output - hidden_states)

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.post_feedforward_layernorm(self.mlp(hidden_states))
        mlp_rate = (self.lambda_2 * self.lambda_scale).abs()
        hidden_states = hidden_states + mlp_rate * (mlp_output - hidden_states)
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
        self.norm = DoryL2Norm()

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
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # Recurrent layers only differ by their cache: same mask as their attention type
        layer_inputs = {
            layer_type: {
                "attention_mask": causal_mask_mapping[layer_type.removeprefix("recurrent_")],
                "position_embeddings": self.rotary_emb(inputs_embeds, position_ids, layer_type),
            }
            for layer_type in set(self.config.layer_types)
        }
        recurrent_start = self.config.n_input_layers
        recurrent_end = recurrent_start + self.config.n_recurrent_layers

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[:recurrent_start]:
            hidden_states = decoder_layer(
                hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.layer_type], **kwargs
            )

        # Each loop restarts from the input section's output
        input_hidden_states = self.norm(hidden_states)
        for loop_idx in range(self.config.n_recurrent_loops):
            hidden_states = self.norm(hidden_states) + input_hidden_states
            for decoder_layer in self.layers[recurrent_start:recurrent_end]:
                hidden_states = decoder_layer(
                    hidden_states,
                    past_key_values=past_key_values,
                    loop_idx=loop_idx,
                    is_last_loop=loop_idx == self.config.n_recurrent_loops - 1,
                    **layer_inputs[decoder_layer.layer_type],
                    **kwargs,
                )

        for decoder_layer in self.layers[recurrent_end:]:
            hidden_states = decoder_layer(
                hidden_states, past_key_values=past_key_values, **layer_inputs[decoder_layer.layer_type], **kwargs
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DoryForCausalLM(LlamaForCausalLM):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.lm_head = DoryLMHead(config)


__all__ = ["DoryPreTrainedModel", "DoryModel", "DoryForCausalLM"]
