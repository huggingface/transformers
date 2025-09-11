# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3 model."""

from typing import Callable, Optional

import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.deprecation import deprecate_kwarg
from ..gemma.modeling_gemma import GemmaMLP
from ..llama.modeling_llama import (
    LlamaAttention,
)
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_qwen3 import Qwen3Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"


class Qwen3RMSNorm(Qwen2RMSNorm):
    pass


class Qwen3MLP(GemmaMLP):
    pass


class Qwen3Attention(LlamaAttention):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
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
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(Qwen2DecoderLayer):
    pass


class Qwen3PreTrainedModel(Qwen2PreTrainedModel):
    pass


class Qwen3Model(Qwen2Model):
    pass


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


class Qwen3ForSequenceClassification(Qwen2ForSequenceClassification):
    pass


class Qwen3ForTokenClassification(Qwen2ForTokenClassification):
    pass


class Qwen3ForQuestionAnswering(Qwen2ForQuestionAnswering):
    pass


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
