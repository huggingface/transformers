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
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from .configuration_nanochat import NanoChatConfig


logger = logging.get_logger(__name__)


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom implementation of Rotary Position Embedding.
    Copied from [nanochat](https://github.com/karpathy/nanochat/blob/4346536ab2e57917ec543b20e88c4bdc47eda572/nanochat/gpt.py#L41)
    and modified to work with the shape of the query and key tensors.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim]
        cos: Cosine part of the rotary embedding of shape [1, seq_len, 1, head_dim//2]
        sin: Sine part of the rotary embedding of shape [1, seq_len, 1, head_dim//2]

    Returns:
        Tuple of rotated query and key tensors of shape [batch, seq_len, num_heads, head_dim] and [batch, seq_len, num_kv_heads, head_dim]
    """
    # Expects query/key as [B, T, H, D] and cos/sin as [1, T, 1, D//2]
    d = query.shape[3] // 2
    q1, q2 = query[..., :d], query[..., d:]
    k1, k2 = key[..., :d], key[..., d:]

    query_rot = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
    key_rot = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)

    return query_rot.to(query.dtype), key_rot.to(key.dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats the key and value tensors n_rep times.
    Copied from [nanochat](https://github.com/karpathy/nanochat/blob/4346536ab2e57917ec543b20e88c4bdc47eda572/nanochat/gpt.py#L52)
    Args:
        hidden_states: Hidden states tensor of shape [batch, seq_len, num_kv_heads, head_dim]
        n_rep: Number of times to repeat the key and value tensors

    Returns:
        Repeated key and value tensors of shape [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Eager attention implementation for NanoChat.

    Args:
        module: The attention module
        query: Query states of shape [batch, num_heads, seq_len, head_dim]
        key: Key states of shape [batch, num_kv_heads, seq_len, head_dim]
        value: Value states of shape [batch, num_kv_heads, seq_len, head_dim]
        attention_mask: Attention mask
        scaling: Scaling factor for attention scores
        dropout: Dropout probability
    """
    # Handle GQA by repeating key/value heads
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    # Compute attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Apply softmax and dropout
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class NanoChatAttention(nn.Module):
    """
    Multi-headed attention from NanoChat with custom RoPE and QK normalization.

    Based on: https://github.com/karpathy/nanochat/blob/main/nanochat/gpt.py#L64
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.qkv_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        # Project the input to get queries, keys, and values
        # Shape: [batch, seq_len, num_heads, head_dim]
        query_states = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Apply QK normalization (RMSNorm) - a key feature of NanoChat architecture
        # This helps stabilize training and is applied AFTER RoPE
        query_states = F.rms_norm(query_states, (query_states.size(-1),), eps=self.config.rms_norm_eps)
        key_states = F.rms_norm(key_states, (key_states.size(-1),), eps=self.config.rms_norm_eps)

        # Transpose to make head dimension the batch dimension
        # Shape: [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Use attention interface pattern for vLLM compatibility
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
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

        # Reshape and project output
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


class NanoChatDecoderLayer(GradientCheckpointingLayer):
    """NanoChat decoder layer with pre-norm architecture."""

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMLP(config)

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights


@auto_docstring
class NanoChatPreTrainedModel(PreTrainedModel):
    config_class = NanoChatConfig
    base_model_prefix = "model"
    _no_split_modules = ["NanoChatDecoderLayer"]
    _supports_attention_backend = True

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
class NanoChatModel(NanoChatPreTrainedModel):
    def __init__(self, config: NanoChatConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NanoChatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Rotary embeddings are cached for efficiency
        self.register_buffer("_rotary_cos", None, persistent=False)
        self.register_buffer("_rotary_sin", None, persistent=False)
        self.gradient_checkpointing = False

        self.post_init()

    def _precompute_rotary_embeddings(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute rotary embeddings (RoPE) for all positions up to max_position_embeddings.

        This implementation is specific to NanoChat and produces cos/sin tensors with shape
        [1, max_seq_len, 1, head_dim//2] instead of the standard full head_dim. so did not use `dynamic_rope_update` decorator.
        """
        # Return cached embeddings if they exist and match device/dtype
        if self._rotary_cos is not None and self._rotary_cos.device == device and self._rotary_cos.dtype == dtype:
            return self._rotary_cos, self._rotary_sin

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        # Stride the time steps (positions)
        positions = torch.arange(self.config.max_position_embeddings, device=device, dtype=torch.float32)
        # Stride the channels (only even channels, head_dim//2 frequencies)
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.config.rope_theta ** (freqs / head_dim))
        # Calculate the rotation frequencies at each (time, channel) pair
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        # Compute cos and sin, add batch and head dims for broadcasting
        cos = angles.cos()[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
        sin = angles.sin()[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
        # Cache the embeddings
        self._rotary_cos = cos.to(dtype=dtype)
        self._rotary_sin = sin.to(dtype=dtype)
        return self._rotary_cos, self._rotary_sin

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

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

        cos, sin = self._precompute_rotary_embeddings(inputs_embeds.device, inputs_embeds.dtype)
        # Clamp cache_position to max_position_embeddings to avoid out-of-bounds indexing
        position_ids_to_use = cache_position.clamp(0, self.config.max_position_embeddings - 1)
        cos = cos[:, position_ids_to_use]
        sin = sin[:, position_ids_to_use]

        hidden_states = inputs_embeds
        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)

        # Collect hidden states and attentions if requested
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, self_attn_weights = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )

            if output_attentions:
                all_self_attns = all_self_attns + (self_attn_weights,)

        hidden_states = F.rms_norm(hidden_states, (hidden_states.size(-1),), eps=self.config.rms_norm_eps)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class NanoChatForCausalLM(NanoChatPreTrainedModel, GenerationMixin):
    """
    The NanoChat Model transformer with a language modeling head on top.
    """

    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: NanoChatConfig):
        super().__init__(config)
        self.model = NanoChatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(new_embeddings)

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
        outputs = self.model(
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
