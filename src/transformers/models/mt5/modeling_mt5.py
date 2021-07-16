# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch mT5 model. """

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import T5PreTrainedModel
from ...modeling_outputs import SequenceClassifierOutput
from ...utils import logging
from ..t5.modeling_t5 import T5EncoderModel, T5ForConditionalGeneration, T5Model
from .configuration_mt5 import MT5Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"


class MT5Model(T5Model):
    r"""
    This class overrides :class:`~transformers.T5Model`. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples::

        >>> from transformers import MT5Model, T5Tokenizer
        >>> model = MT5Model.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="pt")

        >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
        >>> hidden_states = outputs.last_hidden_state
    """
    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]


class MT5ForConditionalGeneration(T5ForConditionalGeneration):
    r"""
    This class overrides :class:`~transformers.T5ForConditionalGeneration`. Please check the superclass for the
    appropriate documentation alongside usage examples.

    Examples::

        >>> from transformers import MT5ForConditionalGeneration, T5Tokenizer
        >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> summary = "Weiter Verhandlung in Syrien."
        >>> inputs = tokenizer(article, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...     labels = tokenizer(summary, return_tensors="pt")

        >>> outputs = model(**inputs,labels=labels["input_ids"])
        >>> loss = outputs.loss
    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]


class MT5EncoderModel(T5EncoderModel):
    r"""
    This class overrides :class:`~transformers.T5EncoderModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples::

        >>> from transformers import MT5EncoderModel, T5Tokenizer
        >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")
        >>> tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
        >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
        >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
        >>> outputs = model(input_ids)
        >>> hidden_state = outputs.last_hidden_state
    """

    model_type = "mt5"
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_save = [
        r"encoder\.embed_tokens\.weight",
    ]


# TODO: src/transformers/models/mt5/modeling_mt5.py:122:0: W0223: Method '_reorder_cache' is abstract in class 'GenerationMixin' but is not overridden (abstract-method)
class MT5ForSequenceClassification(T5PreTrainedModel):
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.encoder_model = MT5EncoderModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # FIXME: this is a hack
        self.model_parallel = self.encoder_model.model_parallel

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder_model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
