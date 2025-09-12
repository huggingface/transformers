# coding=utf-8
# Copyright 2025 The GLM4 & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
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
from typing import Optional, Union

import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.deprecation import deprecate_kwarg
from ..glm.modeling_glm import GlmAttention, GlmForCausalLM, GlmForSequenceClassification, GlmForTokenClassification
from ..phi3.modeling_phi3 import Phi3MLP
from .configuration_glm4 import Glm4Config
from .modeling_glm4 import Glm4RMSNorm


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/GLM-4-9B-0414"


class Glm4MLP(Phi3MLP):
    pass


class Glm4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Glm4Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4Attention(config=config, layer_idx=layer_idx)

        self.mlp = Glm4MLP(config)
        self.input_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = Glm4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Glm4Attention(GlmAttention):
    pass


class Glm4ForCausalLM(GlmForCausalLM):
    def forward(
        self,
        **super_kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Glm4ForCausalLM

        >>> model = Glm4ForCausalLM.from_pretrained("THUDM/GLM-4-9B-0414")
        >>> tokenizer = AutoTokenizer.from_pretrained("THUDM/GLM-4-9B-0414")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


class Glm4ForSequenceClassification(GlmForSequenceClassification):
    pass


class Glm4ForTokenClassification(GlmForTokenClassification):
    pass


__all__ = [
    "Glm4PreTrainedModel",  # noqa: F822
    "Glm4Model",  # noqa: F822
    "Glm4ForCausalLM",
    "Glm4ForSequenceClassification",
    "Glm4ForTokenClassification",
]
