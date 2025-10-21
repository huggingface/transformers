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

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..clip.modeling_clip import CLIPMLP
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..qwen3.modeling_qwen3 import Qwen3Attention
from ..llama4.modeling_llama4 import Llama4TextL2Norm
from .configuration_nanochat import NanoChatConfig


class NanoChatRMSNorm(Llama4TextL2Norm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(eps=eps)
        self.hidden_size = hidden_size


class NanoChatRotaryEmbedding(LlamaRotaryEmbedding):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input with flipped signs for NanoChat."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors using NanoChat's rotate_half."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NanoChatAttention(Qwen3Attention):
    """
    Multi-headed attention from NanoChat with custom RoPE and QK normalization.

    Based on: https://github.com/karpathy/nanochat/blob/main/nanochat/gpt.py#L64
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        # Temporarily add attention_bias and layer_types for Qwen3 compatibility
        config.attention_bias = config.qkv_bias
        if not hasattr(config, "layer_types"):
            config.layer_types = ["full_attention"] * config.num_hidden_layers
        super().__init__(config, layer_idx)
        self.sliding_window = None

        self.q_norm = NanoChatRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = NanoChatRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        NanoChat uses a different order than Qwen3: RoPE is applied before QK normalization.
        Qwen3 order: project -> norm -> transpose -> RoPE
        NanoChat order: project -> transpose -> RoPE -> norm

        NanoChat also uses a custom rotate_half with flipped signs.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class NanoChatMLP(CLIPMLP):
    def __init__(self, config):
        super().__init__(config)
        # Override with bias=False for NanoChat
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


class NanoChatDecoderLayer(LlamaDecoderLayer):
    """NanoChat decoder layer with pre-norm architecture."""

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NanoChatAttention(config, layer_idx)
        self.mlp = NanoChatMLP(config)
        self.input_layernorm = NanoChatRMSNorm(config, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanoChatRMSNorm(config, eps=config.rms_norm_eps)


@auto_docstring
class NanoChatPreTrainedModel(LlamaPreTrainedModel):
    config: NanoChatConfig
    _no_split_modules = ["NanoChatDecoderLayer"]

    _can_record_outputs = {
        "hidden_states": NanoChatDecoderLayer,
        "attentions": NanoChatAttention,
    }

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)

        for name, param in module.named_parameters():
            if name == "o_proj.weight":
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
                )


@auto_docstring
class NanoChatModel(LlamaModel):
    def __init__(self, config: NanoChatConfig):
        self.initial_norm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [NanoChatDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NanoChatRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = NanoChatRotaryEmbedding(config=config)
        super().__init__(config)

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

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class NanoChatForCausalLM(LlamaForCausalLM, GenerationMixin):
    def __init__(self, config: NanoChatConfig):
        super().__init__(config)
        self.model = NanoChatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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
        >>>     outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        >>> generated_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        >>> output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        ```"""
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

        # Soft-cap the logits. The main difference to LlamaForCausalLM.forward.
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
