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
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available
from ..bamba.modeling_bamba import apply_mask_to_padding_states
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
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


class Lfm2RotaryEmbedding(LlamaRotaryEmbedding):
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


class Lfm2HybridConvCache:
    """
    Attention and conv cache for Lfm2.

    It stores the Key and Value states as a list of tensors, one for each layer.
    Attention layer cache shape: `[batch_size, num_heads, seq_len, head_dim]`.
    Conv layer cache shape: `[batch_size, hidden_size, L_cache-1]`.
    """

    # Override @property existing in Cache
    max_batch_size = None
    is_compileable = False
    key_cache = None
    value_cache = None

    def __init__(
        self,
        config: Lfm2Config,
        max_batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str, None] = None,
    ):
        self.key_cache = []
        self.value_cache = []
        self.max_batch_size = max_batch_size
        self.layer_types = config.layer_types
        self.first_attention_layer = self.layer_types.index("full_attention")
        self.conv_L_cache = config.conv_L_cache
        self._dtype = dtype

        self.conv_cache: list[torch.Tensor] = []
        device = torch.device(device) if device is not None else None

        for _ in range(config.num_hidden_layers):
            conv_state = torch.zeros(
                self.max_batch_size,
                config.hidden_size,
                self.conv_L_cache,
                dtype=self._dtype,
                device=device,
            )
            torch._dynamo.mark_static_address(conv_state)
            self.conv_cache.append(conv_state)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_cache[layer_idx].device
            self.conv_cache[layer_idx] = self.conv_cache[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.first_attention_layer if self.layer_types[layer_idx] != "full_attention" else layer_idx
        if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx].numel() == 0:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """
        full_mask_kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, full_mask_kv_offset

    def crop(self, max_length: int):
        """Crop the cache to the given length"""
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_cache)):
            # In-place ops prevent breaking the static address
            self.conv_cache[layer_idx].zero_()


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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def cuda_kernels_forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
        if past_key_values is not None and cache_position[0] > 0:
            conv_out = causal_conv1d_update(
                Bx.squeeze(-1),
                past_key_values.conv_cache[self.layer_idx],
                conv_weights,
                self.conv.bias,
                None,
            )
            conv_out = conv_out.unsqueeze(-1)
        else:
            if past_key_values is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                past_key_values.conv_cache[self.layer_idx].copy_(conv_state)

            conv_out = causal_conv1d_fn(Bx, conv_weights, self.conv.bias, activation=None)

        y = C * conv_out
        y = self.out_proj(y.transpose(-1, -2).contiguous())
        return y

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def slow_forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        seqlen = x.shape[1]

        x = apply_mask_to_padding_states(x, attention_mask)
        BCx = self.in_proj(x).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)

        Bx = B * x

        if past_key_values is not None and cache_position[0] > 0:
            conv_state = past_key_values.conv_cache[self.layer_idx]
            cache_position = cache_position.clamp(0, self.L_cache - 1)
            conv_state = conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, cache_position] = Bx.to(device=conv_state.device, dtype=conv_state.dtype)
            past_key_values.conv_cache[self.layer_idx].copy_(conv_state)
            conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
            if self.bias:
                conv_out += self.conv.bias

            conv_out = conv_out.unsqueeze(-1)
        else:
            if past_key_values is not None:
                conv_state = nn.functional.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
                past_key_values.conv_cache[self.layer_idx].copy_(conv_state)

            conv_out = self.conv(Bx)[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        y = self.out_proj(y)
        return y

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in hidden_states.device.type and not torch._dynamo.is_compiling():
            return self.cuda_kernels_forward(hidden_states, past_key_values, cache_position, attention_mask)
        return self.slow_forward(hidden_states, past_key_values, cache_position, attention_mask)


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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
                cache_position=cache_position,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                cache_position=cache_position,
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
        self.pos_emb = Lfm2RotaryEmbedding(config)
        self.embedding_norm = Lfm2RMSNorm(config.hidden_size, eps=config.norm_eps)
        del self.norm
        del self.rotary_emv

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2HybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2HybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.pos_emb(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
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
