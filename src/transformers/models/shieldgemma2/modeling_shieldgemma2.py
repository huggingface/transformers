# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_outputs import ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    logging,
)
from ..auto import AutoModelForImageTextToText
from .configuration_shieldgemma2 import ShieldGemma2Config


logger = logging.get_logger(__name__)


@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(ImageClassifierOutputWithNoAttention):
    """ShieldGemma2 classifies imags as violative or not relative to a specific policy
    Args:
    """

    probabilities: Optional[torch.Tensor] = None


@auto_docstring
class ShieldGemma2ForImageClassification(PreTrainedModel):
    config_class = ShieldGemma2Config

    def __init__(self, config: ShieldGemma2Config):
        super().__init__(config=config)
        self.yes_token_index = getattr(config, "yes_token_index", 10_784)
        self.no_token_index = getattr(config, "no_token_index", 3771)
        self.model = AutoModelForImageTextToText.from_config(config=config)

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.language_model.get_decoder()

    def tie_weights(self):
        return self.model.language_model.tie_weights()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> ShieldGemma2ImageClassifierOutputWithNoAttention:
        r"""
        Returns:
            A `ShieldGemma2ImageClassifierOutputWithNoAttention` instance containing the logits and probabilities
            associated with the model predicting the `Yes` or `No` token as the response to that prompt, captured in the
            following properties.

                *   `logits` (`torch.Tensor` of shape `(batch_size, 2)`):
                    The first position along dim=1 is the logits for the `Yes` token and the second position along dim=1 is
                    the logits for the `No` token.
                *   `probabilities` (`torch.Tensor` of shape `(batch_size, 2)`):
                    The first position along dim=1 is the probability of predicting the `Yes` token and the second position
                    along dim=1 is the probability of predicting the `No` token.

            ShieldGemma prompts are constructed such that predicting the `Yes` token means the content *does violate* the
            policy as described. If you are only interested in the violative condition, use
            `violated = outputs.probabilities[:, 1]` to extract that slice from the output tensors.

            When used with the `ShieldGemma2Processor`, the `batch_size` will be equal to `len(images) * len(policies)`,
            and the order within the batch will be img1_policy1, ... img1_policyN, ... imgM_policyN.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        logits = outputs.logits
        selected_logits = logits[:, -1, [self.yes_token_index, self.no_token_index]]
        probabilities = torch.softmax(selected_logits, dim=-1)
        return ShieldGemma2ImageClassifierOutputWithNoAttention(
            logits=selected_logits,
            probabilities=probabilities,
        )


__all__ = [
    "ShieldGemma2ForImageClassification",
]
