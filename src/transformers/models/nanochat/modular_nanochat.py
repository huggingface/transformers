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

import torch
import torch.nn as nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..gemma2.modeling_gemma2 import Gemma2ForCausalLM
from ..hunyuan_v1_dense.modeling_hunyuan_v1_dense import HunYuanDenseV1Attention
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
)
from ..llama4.modeling_llama4 import Llama4TextL2Norm
from ..mimi.modeling_mimi import MimiMLP
from .configuration_nanochat import NanoChatConfig


class NanoChatRMSNorm(Llama4TextL2Norm):
    pass


class NanoChatRotaryEmbedding(LlamaRotaryEmbedding):
    pass


def rotate_half(x):
    """Rotates half the hidden dims of the input with flipped signs for NanoChat."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


class NanoChatAttention(HunYuanDenseV1Attention):
    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # RoPE -> Norm (instead of usual Norm -> RoPE)
        self.q_norm = NanoChatRMSNorm(eps=config.rms_norm_eps)
        self.k_norm = NanoChatRMSNorm(eps=config.rms_norm_eps)


class NanoChatMLP(MimiMLP):
    pass


class NanoChatDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()

        self.input_layernorm = NanoChatRMSNorm(eps=config.rms_norm_eps)
        self.post_attention_layernorm = NanoChatRMSNorm(eps=config.rms_norm_eps)


@auto_docstring
class NanoChatPreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, NanoChatAttention):
            init.normal_(
                module.o_proj.weight,
                mean=0.0,
                std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers),
            )


@auto_docstring
class NanoChatModel(LlamaModel):
    def __init__(self, config: NanoChatConfig):
        super().__init__(config)

        self.norm = NanoChatRMSNorm(eps=config.rms_norm_eps)

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
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

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

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        hidden_states = self.norm(hidden_states)  # Additional norm before the layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class NanoChatForCausalLM(Gemma2ForCausalLM):
    _tp_plan = {"lm_head": "colwise_gather_output"}

    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def forward(self, **super_kwargs) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("karpathy/nanochat-d32")

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
        super().forward(**super_kwargs)


__all__ = [
    "NanoChatPreTrainedModel",
    "NanoChatModel",
    "NanoChatForCausalLM",
]
