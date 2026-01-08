# Copyright 2025 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
import torch

from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3TopkRouter,
)
from ..qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    TransformersKwargs,
)
from .configuration_dots1 import Dots1Config


logger = logging.get_logger(__name__)


class Dots1RMSNorm(Qwen3RMSNorm):
    pass


class Dots1RotaryEmbedding(Qwen3RotaryEmbedding):
    pass


class Dots1Attention(Qwen3Attention):
    pass


class Dots1MLP(DeepseekV3MLP):
    pass


class Dots1TopkRouter(DeepseekV3TopkRouter):
    pass


class Dots1MoE(DeepseekV3MoE):
    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()  # main diff with deepseekv3
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class Dots1DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config: Dots1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]


class Dots1PreTrainedModel(DeepseekV3PreTrainedModel):
    pass


class Dots1Model(Qwen3Model):
    pass


class Dots1ForCausalLM(Qwen3ForCausalLM):
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
        >>> from transformers import AutoTokenizer, Dots1ForCausalLM

        >>> model = Dots1ForCausalLM.from_pretrained("rednote-hilab/dots1.llm1.inst")
        >>> tokenizer = AutoTokenizer.from_pretrained("rednote-hilab/dots1.llm1.inst")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Dots1PreTrainedModel",
    "Dots1Model",
    "Dots1ForCausalLM",
]
