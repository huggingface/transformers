# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.import_utils import is_causal_conv1d_available, is_torchdynamo_compiling
from ..bamba.modeling_bamba import apply_mask_to_padding_states
from ..gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_lfm2 import Lfm2Config


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_fn, causal_conv1d_update = None, None


kernel_modules = (causal_conv1d_fn, causal_conv1d_update)
is_fast_path_available = all(kernel_modules)


logger = logging.get_logger(__name__)


class Lfm2RMSNorm(LlamaRMSNorm):
    pass


class Lfm2RotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class Lfm2MLP(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        intermediate_size = config.intermediate_size
        if config.block_auto_adjust_ff_dim:
            intermediate_size = int(2 * intermediate_size / 3)
            # custom dim factor multiplier
            if config.block_ffn_dim_multiplier is not None:
                intermediate_size = int(config.block_ffn_dim_multiplier * intermediate_size)
                intermediate_size = config.block_multiple_of * (
                    (intermediate_size + config.block_multiple_of - 1) // config.block_multiple_of
                )
        self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm2Attention(LlamaAttention):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2RMSNorm(self.head_dim, eps=config.norm_eps)
        del self.o_proj
        del self.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

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
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.out_proj(attn_output)
        return output, attn_weights


class Lfm2ShortConv(nn.Module):
    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def cuda_kernels_forward(
        self,
        x: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx):
            conv_out = causal_conv1d_update(
                Bx.squeeze(-1),
                past_key_values.layers[self.layer_idx].conv_states,
                conv_weights,
                self.conv.bias,
                None,
            )
            conv_out = conv_out.unsqueeze(-1)
        else:
            if past_key_values is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                conv_state = past_key_values.update_conv_state(conv_state, self.layer_idx)

            conv_out = causal_conv1d_fn(Bx, conv_weights, self.conv.bias, activation=None)

        y = C * conv_out
        y = self.out_proj(y.transpose(-1, -2).contiguous())
        return y

    def slow_forward(
        self,
        x: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        seqlen = x.shape[1]

        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if past_key_values is not None and past_key_values.has_previous_state(self.layer_idx):
            conv_state = past_key_values.update_conv_state(Bx, self.layer_idx)
            conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias

            conv_out = conv_out.unsqueeze(-1)
        else:
            if past_key_values is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                conv_state = past_key_values.update_conv_state(conv_state, self.layer_idx)

            conv_out = self.conv(Bx)[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if is_fast_path_available and "cuda" in hidden_states.device.type and not is_torchdynamo_compiling():
            return self.cuda_kernels_forward(hidden_states, past_key_values, attention_mask)
        return self.slow_forward(hidden_states, past_key_values, attention_mask)


class Lfm2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm2Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"

        if self.is_attention_layer:
            self.self_attn = Lfm2Attention(config, layer_idx)
        else:
            self.conv = Lfm2ShortConv(config, layer_idx)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))

        return hidden_states


class Lfm2PreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = False


class Lfm2Model(LlamaModel):
    def __init__(self, config: Lfm2Config):
        super().__init__(config)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        del self.norm

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

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        # Skip masking for decoding stage. We check shape here to be compile-friendly
        linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = causal_mask if self.config.layer_types[i] == "full_attention" else linear_attention
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Lfm2ForCausalLM(LlamaForCausalLM):
    pass


__all__ = ["Lfm2ForCausalLM", "Lfm2Model", "Lfm2PreTrainedModel"]
