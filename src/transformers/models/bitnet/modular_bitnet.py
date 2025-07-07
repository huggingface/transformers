# coding=utf-8
# Copyright 2025 The BitNet Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BitNet model."""

from typing import Callable, Optional

import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_bitnet import BitNetConfig


logger = logging.get_logger(__name__)


class BitNetRMSNorm(LlamaRMSNorm):
    pass


class BitNetMLP(GemmaMLP):
    def __init__(self, config: BitNetConfig):
        super().__init__(config)
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
        down_proj = self.down_proj(self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj


class BitNetAttention(LlamaAttention):
    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        attn_output = self.attn_sub_norm(attn_output)  # diff with Llama
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class BitNetDecoderLayer(LlamaDecoderLayer):
    pass


class BitNetModel(LlamaModel):
    pass


class BitNetForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = None
    _pp_plan = None

    def forward(
        self,
        **super_kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BitNetForCausalLM

        >>> model = BitNetForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

        >>> prompt = f'<|begin_of_text|>User: Hey, are you conscious? Can you talk to me?<|eot_id|>Assistant: '
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=100)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "User: Hey, are you conscious? Can you talk to me?Assistant: No, I'm not conscious. I'm an artificial intelligence designed to assist with information and tasks. How can I help you today?"
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "BitNetForCausalLM",
    "BitNetModel",
    "BitNetPreTrainedModel",  # noqa: F822
]
