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

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F

from ...cache_utils import Cache
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from ..mixtral.modeling_mixtral import (
    MixtralExperts,
    MixtralForCausalLM,
    MixtralModel,
    MixtralSparseMoeBlock,
    load_balancing_loss_func,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeMLP
from ..qwen3.modeling_qwen3 import Qwen3Attention
from .configuration_qwen3_moe import Qwen3MoeConfig


logger = logging.get_logger(__name__)


class Qwen3MoeAttention(Qwen3Attention):  # This is the main diff with qwen2Moe!
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", None)


class Qwen3MoeMLP(Qwen2MoeMLP):
    pass


class Qwen3MoeRouter(nn.Linear):
    def __init__(self, config: Qwen3MoeConfig):
        self.num_experts = config.num_experts
        super().__init__(config.hidden_size, self.num_experts, bias=False)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, sequence_length, hidden_dim)
        Returns:
            router_logits: (batch_size * sequence_length, num_experts)
            selected_experts:s (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = super().forward(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, selected_experts, routing_weights


class Qwen3MoeExperts(MixtralExperts, nn.ModuleList):
    def __init__(self, config: Qwen3MoeConfig):
        nn.ModuleList.__init__(self)
        self.num_experts = config.num_experts
        for _ in range(self.num_experts):
            self.append(Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size))


class Qwen3MoeSparseMoeBlock(MixtralSparseMoeBlock):
    pass


class Qwen3MoeRMSNorm(LlamaRMSNorm):
    pass


class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer):
    pass


class Qwen3MoeModel(MixtralModel):
    pass


class Qwen3MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Qwen3MoeForSequenceClassification(LlamaForSequenceClassification):
    pass


class Qwen3MoeForTokenClassification(LlamaForTokenClassification):
    pass


class Qwen3MoeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForQuestionAnswering",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",  # noqa: F822
    "Qwen3MoeForSequenceClassification",
    "Qwen3MoeForTokenClassification",
]
