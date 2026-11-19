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

import copy
from collections.abc import Callable

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults, no_inherit_decorator
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
    and a learned scale `s_gate` inside the output gate.
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
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

    @no_inherit_decorator
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        cache_offset: int = 0,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
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

        if shared_kv_states is not None:
            # `last_loop`: earlier loops attend over the last loop's cached past and their own tokens, without writing
            if (past := shared_kv_states.get(self.layer_idx)) is not None:
                key_states = torch.cat([past[0], key_states], dim=-2)
                value_states = torch.cat([past[1], value_states], dim=-2)
                if self.sliding_window is not None:
                    # The cache may hold more than the window if past recording is on
                    num_visible = self.sliding_window - 1 + query_states.shape[-2]
                    key_states, value_states = key_states[..., -num_visible:, :], value_states[..., -num_visible:, :]
        elif cache is not None:
            # `per_loop` loops past the first write to their own copy of this layer's cache, see `DoryModel.forward`
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx + cache_offset)

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
        self.attention_type = config.layer_types[layer_idx]
        self.input_layernorm = DoryL2Norm()
        self.post_attention_layernorm = DoryL2Norm()
        self.pre_feedforward_layernorm = DoryL2Norm()
        self.post_feedforward_layernorm = DoryL2Norm()
        self.lambda_1 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_2 = nn.Parameter(torch.full((config.hidden_size,), config.initializer_range))
        self.lambda_scale = config.ngpt_alpha_init / config.initializer_range

    @no_inherit_decorator
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            cache=cache,
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
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
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
        self.rotary_emb = DoryRotaryEmbedding(config)
        self.norm = DoryL2Norm()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        position_embeddings = {
            attention_type: self.rotary_emb(inputs_embeds, position_ids, attention_type)
            for attention_type in {layer.attention_type for layer in self.layers}
        }

        n_loops = self.config.n_recurrent_loops
        recurrent_start = self.config.n_input_layers
        n_recurrent = self.config.n_recurrent_layers
        recurrent_end = recurrent_start + n_recurrent
        last_loop_only = self.config.recurrent_kv_cache_mode == "last_loop"

        shared_kv_states = None
        if cache is not None and last_loop_only:
            # Only the last loop writes, the earlier ones read the past
            shared_kv_states = {i: (c.keys, c.values) for i, c in enumerate(cache.layers) if c.is_initialized}
        elif cache is not None and len(cache.layers) == self.config.num_hidden_layers:
            # Loops 1, 2, ... write to copies of the recurrent cache layers, appended at the end
            cache.layers += [copy.deepcopy(c) for c in cache.layers[recurrent_start:recurrent_end] * (n_loops - 1)]

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[:recurrent_start]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                cache=cache,
                **kwargs,
            )

        # Loop i > 0 writes to its copies of the recurrent cache layers, appended after the model's layers
        offset = len(self.layers) - recurrent_start - n_recurrent
        # Each loop restarts from the input section's output
        input_hidden_states = self.norm(hidden_states)
        for loop_idx in range(n_loops):
            hidden_states = self.norm(hidden_states) + input_hidden_states
            cache_offset = 0 if loop_idx == 0 or last_loop_only else offset + loop_idx * n_recurrent
            for decoder_layer in self.layers[recurrent_start:recurrent_end]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_embeddings=position_embeddings[decoder_layer.attention_type],
                    cache=cache,
                    cache_offset=cache_offset,
                    shared_kv_states=shared_kv_states if loop_idx < n_loops - 1 else None,
                    **kwargs,
                )

        output_start = self.config.num_hidden_layers - self.config.n_output_layers
        for decoder_layer in self.layers[output_start:]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                cache=cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


class DoryForCausalLM(LlamaForCausalLM):
    def __init__(self, config: DoryConfig):
        super().__init__(config)
        self.lm_head = DoryLMHead(config)

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        return super().forward(**super_kwargs)


__all__ = ["DoryPreTrainedModel", "DoryModel", "DoryForCausalLM"]
