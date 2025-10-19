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

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..llama.modeling_llama import LlamaPreTrainedModel, LlamaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from ..llama4.modeling_llama4 import Llama4TextL2Norm
from .configuration_nanochat import NanoChatConfig


class NanoChatRMSNorm(Llama4TextL2Norm):
    """
    NanoChatRMSNorm inherits from Llama4TextL2Norm (weight-less RMS normalization).
    Overrides __init__ to match NanoChat's API with hidden_size parameter.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(eps=eps)
        self.hidden_size = hidden_size


class NanoChatRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class NanoChatAttention(nn.Module):
    """
    Multi-headed attention from NanoChat with custom RoPE and QK normalization.

    Based on: https://github.com/karpathy/nanochat/blob/main/nanochat/gpt.py#L64
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True

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
        self.query_norm = NanoChatRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_norm = NanoChatRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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

        # Project the input to get queries, keys, and values [batch, num_heads, seq_len, head_dim]
        query_states = (
            self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = position_embeddings
        # NanoChat uses a negative sine for the rotary embedding
        sin = -sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Apply QK normalization (RMSNorm)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

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
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class NanoChatDecoderLayer(GradientCheckpointingLayer):
    """NanoChat decoder layer with pre-norm architecture."""

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMLP(config)
        self.input_layernorm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states = self.input_layernorm(hidden_states)
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, self_attn_weights


@auto_docstring
class NanoChatPreTrainedModel(LlamaPreTrainedModel):
    config_class = NanoChatConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanoChatDecoderLayer"]
    _supports_attention_backend = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True

    _can_record_outputs = {
        "hidden_states": NanoChatDecoderLayer,
        "attentions": NanoChatAttention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)

        # NanoChat-specific: scaled initialization for output projection
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
        self.initial_norm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [NanoChatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = NanoChatRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @check_model_inputs()
    @auto_docstring
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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = self.initial_norm(hidden_states)

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
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if output_attentions:
                all_self_attns = all_self_attns + (self_attn_weights,)

        hidden_states = self.norm(hidden_states)
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

    _tied_weights_keys = ["lm_head.weight"]
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

    @can_return_tuple
    @auto_docstring
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
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained(model_id"karpathy/nanochat-d32")

        >>> tokenizer = AutoTokenizer.from_pretrained("karpathy/nanochat-d32")

        >>> conversation = [
                {"role": "user", "content": "What is the capital of France?"},
            ]

        >>> inputs = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to(device)

        >>> with torch.no_grad():
            >>> outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        >>> generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        >>> output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        ```"""
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
