# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_nanochat import NanoChatConfig


def rotate_half(x):
    """Rotates half the hidden dims of the input.
    
    NanoChat uses a different rotation convention than standard Llama.
    Llama uses: [-x2, x1], NanoChat uses: [x2, -x1] to match the original nanochat implementation.
    This results in: [q1 * cos + q2 * sin, -(q1 * sin) + q2 * cos] instead of [q1 * cos - q2 * sin, q1 * sin + q2 * cos]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


class NanoChatRotaryEmbedding(LlamaRotaryEmbedding):
    """Inherits from LlamaRotaryEmbedding but uses NanoChat's rotate_half."""
    pass


class NanoChatAttention(LlamaAttention):
    """
    Multi-headed attention from NanoChat with custom QK normalization.
    Inherits from LlamaAttention but adds RMSNorm to queries and keys after RoPE.
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = True
        # Override bias settings for NanoChat
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.qkv_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # NanoChat-specific: Apply QK normalization after RoPE
        query_states = F.rms_norm(query_states, (query_states.size(-1),), eps=self.config.rms_norm_eps)
        key_states = F.rms_norm(key_states, (key_states.size(-1),), eps=self.config.rms_norm_eps)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NanoChatMLP(nn.Module):
    """MLP module for NanoChat with ReLU^2 activation."""

    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states)
        hidden_states = F.relu(hidden_states).square()
        hidden_states = self.proj(hidden_states)
        return hidden_states


class NanoChatDecoderLayer(LlamaDecoderLayer):
    """
    NanoChat decoder layer with pre-norm architecture.
    Uses functional RMSNorm instead of module-based norm layers.
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMLP(config)
        # Remove norm layers as NanoChat uses functional RMSNorm
        del self.input_layernorm
        del self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class NanoChatPreTrainedModel(LlamaPreTrainedModel):
    config_class = NanoChatConfig
    base_model_prefix = "model"
    _no_split_modules = ["NanoChatDecoderLayer"]
    _supports_attention_backend = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    
    _can_record_outputs = {
        "hidden_states": NanoChatDecoderLayer,
        "attentions": NanoChatAttention,
    }
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

        for name, param in module.named_parameters():
            if name == "o_proj.weight":
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
                )


@auto_docstring
class NanoChatModel(LlamaModel):
    """
    NanoChat model that inherits from LlamaModel but uses NanoChat-specific layers
    and functional RMSNorm instead of module-based normalization.
    """

    def __init__(self, config: NanoChatConfig):
        # Call PreTrainedModel.__init__ directly to avoid LlamaModel's __init__
        PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NanoChatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = NanoChatRotaryEmbedding(config=config)
        # Remove the norm layer as NanoChat uses functional RMSNorm
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        from ...cache_utils import DynamicCache
        from ...masking_utils import create_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
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

        # NanoChat-specific: Apply RMSNorm to embeddings
        hidden_states = F.rms_norm(inputs_embeds, (inputs_embeds.size(-1),), eps=self.config.rms_norm_eps)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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

        # NanoChat-specific: Apply final RMSNorm
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class NanoChatForCausalLM(LlamaForCausalLM):
    """
    The NanoChat Model transformer with a language modeling head on top.
    Inherits from LlamaForCausalLM but uses NanoChatModel and supports logits soft capping.
    """

    def __init__(self, config: NanoChatConfig):
        # Call PreTrainedModel.__init__ directly
        PreTrainedModel.__init__(self, config)
        self.model = NanoChatModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # NanoChat-specific: Apply logits soft capping if configured
        if self.config.logits_soft_cap is not None:
            cap = self.config.logits_soft_cap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "NanoChatPreTrainedModel",
    "NanoChatModel",
    "NanoChatForCausalLM",
]

