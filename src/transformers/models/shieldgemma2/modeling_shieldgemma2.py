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
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from ..auto import AutoModelForImageTextToText
from .configuration_shieldgemma2 import ShieldGemma2Config


_CHECKPOINT_FOR_DOC = "google/shieldgemma-2-4b-it"
_CONFIG_FOR_DOC = "ShieldGemma2Config"

logger = logging.get_logger(__name__)

SHIELDGEMMA2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@dataclass
class ShieldGemma2ImageClassifierOutputWithNoAttention(ImageClassifierOutputWithNoAttention):
    """ShieldGemma2 classifies imags as violative or not relative to a specific policy
    Args:
    """

    probabilities: Optional[torch.Tensor] = None


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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(SHIELDGEMMA2_INPUTS_DOCSTRING)
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
        """Predicts the binary probability that the image violates the specified policy.

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
