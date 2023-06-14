# coding=utf-8
# Copyright 2020 Google AI, Google Brain, the HuggingFace Inc. team and Microsoft Corporation.
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
"""PyTorch ALBERT model with Patience-based Early Exit. """

import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.albert.modeling_albert import (
    ALBERT_INPUTS_DOCSTRING,
    ALBERT_START_DOCSTRING,
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertTransformer,
)


logger = logging.getLogger(__name__)


class AlbertTransformerWithPabee(AlbertTransformer):
    def adaptive_forward(self, hidden_states, current_layer, attention_mask=None, head_mask=None):
        if current_layer == 0:
            hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        else:
            hidden_states = hidden_states[0]

        layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

        # Index of the hidden group
        group_idx = int(current_layer / (self.config.num_hidden_layers / self.config.num_hidden_groups))

        layer_group_output = self.albert_layer_groups[group_idx](
            hidden_states,
            attention_mask,
            head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
        )
        hidden_states = layer_group_output[0]

        return (hidden_states,)


@add_start_docstrings(
    "The bare ALBERT Model transformer with PABEE outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class AlbertModelWithPabee(AlbertModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = AlbertTransformerWithPabee(config)

        self.init_weights()
        self.patience = 0
        self.inference_instances_num = 0
        self.inference_layers_num = 0

        self.regression_threshold = 0

    def set_regression_threshold(self, threshold):
        self.regression_threshold = threshold

    def set_patience(self, patience):
        self.patience = patience

    def reset_stats(self):
        self.inference_instances_num = 0
        self.inference_layers_num = 0

    def log_stats(self):
        avg_inf_layers = self.inference_layers_num / self.inference_instances_num
        message = (
            f"*** Patience = {self.patience} Avg. Inference Layers = {avg_inf_layers:.2f} Speed Up ="
            f" {1 - avg_inf_layers / self.config.num_hidden_layers:.2f} ***"
        )
        print(message)

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_dropout=None,
        output_layers=None,
        regression=False,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = embedding_output

        if self.training:
            res = []
            for i in range(self.config.num_hidden_layers):
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=i,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                )

                pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
                logits = output_layers[i](output_dropout(pooled_output))
                res.append(logits)
        elif self.patience == 0:  # Use all layers for inference
            encoder_outputs = self.encoder(encoder_outputs, extended_attention_mask, head_mask=head_mask)
            pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
            res = [output_layers[self.config.num_hidden_layers - 1](pooled_output)]
        else:
            patient_counter = 0
            patient_result = None
            calculated_layer_num = 0
            for i in range(self.config.num_hidden_layers):
                calculated_layer_num += 1
                encoder_outputs = self.encoder.adaptive_forward(
                    encoder_outputs,
                    current_layer=i,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                )

                pooled_output = self.pooler_activation(self.pooler(encoder_outputs[0][:, 0]))
                logits = output_layers[i](pooled_output)
                if regression:
                    labels = logits.detach()
                    if patient_result is not None:
                        patient_labels = patient_result.detach()
                    if (patient_result is not None) and torch.abs(patient_result - labels) < self.regression_threshold:
                        patient_counter += 1
                    else:
                        patient_counter = 0
                else:
                    labels = logits.detach().argmax(dim=1)
                    if patient_result is not None:
                        patient_labels = patient_result.detach().argmax(dim=1)
                    if (patient_result is not None) and torch.all(labels.eq(patient_labels)):
                        patient_counter += 1
                    else:
                        patient_counter = 0

                patient_result = logits
                if patient_counter == self.patience:
                    break
            res = [patient_result]
            self.inference_layers_num += calculated_layer_num
            self.inference_instances_num += 1

        return res


@add_start_docstrings(
    """Albert Model transformer with PABEE and a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    ALBERT_START_DOCSTRING,
)
class AlbertForSequenceClassificationWithPabee(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModelWithPabee(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, self.config.num_labels) for _ in range(config.num_hidden_layers)]
        )

        self.init_weights()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

        Returns:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.AlbertConfig`) and inputs:
            loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            logits ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

            Examples::

                from transformers import AlbertTokenizer
                from pabee import AlbertForSequenceClassificationWithPabee
                from torch import nn
                import torch

                tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
                model = AlbertForSequenceClassificationWithPabee.from_pretrained('albert-base-v2')
                input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
                labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
                outputs = model(input_ids, labels=labels)
                loss, logits = outputs[:2]

        """

        logits = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_dropout=self.dropout,
            output_layers=self.classifiers,
            regression=self.num_labels == 1,
        )

        outputs = (logits[-1],)

        if labels is not None:
            total_loss = None
            total_weights = 0
            for ix, logits_item in enumerate(logits):
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits_item.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits_item.view(-1, self.num_labels), labels.view(-1))
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss * (ix + 1)
                total_weights += ix + 1
            outputs = (total_loss / total_weights,) + outputs

        return outputs
