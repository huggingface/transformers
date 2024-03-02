"""Convert CodeReviewer checkpoint."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    CodeReviewerForConditionalGeneration,
    RobertaTokenizer,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.utils import logging


#  logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class ReviewerModel(T5ForConditionalGeneration):

    def __init__(self, config):
        super().__init__(config)
        self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        factor = self.config.initializer_factor
        self.cls_head.weight.data.normal_(mean=0.0, \
            std=factor * ((self.config.d_model) ** -0.5))
        self.cls_head.bias.data.zero_()

    def forward(
        self, *argv, **kwargs
    ):
        #  print(kwargs)
        if "cls" in kwargs:
            assert (
                "input_ids" in kwargs and \
                "labels" in kwargs and \
                "attention_mask" in kwargs
            )
            return self.cls(
                input_ids=kwargs["input_ids"],
                labels=kwargs["labels"],
                attention_mask=kwargs["attention_mask"],
            )
        if "input_labels" in kwargs:
            assert (
                "input_ids" in kwargs and \
                "input_labels" in kwargs and \
                "decoder_input_ids" in kwargs and \
                "attention_mask" in kwargs and \
                "decoder_attention_mask" in kwargs
            ), "Please give these arg keys."
            input_ids = kwargs["input_ids"]
            input_labels = kwargs["input_labels"]
            decoder_input_ids = kwargs["decoder_input_ids"]
            attention_mask = kwargs["attention_mask"]
            decoder_attention_mask = kwargs["decoder_attention_mask"]
            if "encoder_loss" not in kwargs:
                encoder_loss = True
            else:
                encoder_loss = kwargs["encoder_loss"]
            return self.review_forward(input_ids, input_labels, decoder_input_ids, attention_mask, decoder_attention_mask, encoder_loss)
        return super().forward(*argv, **kwargs)

    def cls(
        self,
        input_ids,
        labels,
        attention_mask,
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]
        first_hidden = nn.Dropout(0.3)(first_hidden)
        logits = self.cls_head(first_hidden)
        loss_fct = CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits, labels)
            return loss
        return logits

    def review_forward(
        self,
        input_ids,
        input_labels,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        encoder_loss=True
    ):
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        hidden_states = encoder_outputs[0]
        decoder_inputs = self._shift_right(decoder_input_ids)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=False,
            return_dict=False
        )
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings: # this is True default
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        if encoder_loss:
            # print(self.encoder.get_input_embeddings().weight.shape)
            cls_logits = nn.functional.linear(hidden_states, self.encoder.get_input_embeddings().weight)
            # cls_logits = self.cls_head(hidden_states)
        lm_logits = self.lm_head(sequence_output)
        if decoder_input_ids is not None:
            lm_loss_fct = CrossEntropyLoss(ignore_index=0)      # Warning: PAD_ID should be 0
            loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))
            if encoder_loss and input_labels is not None:
                cls_loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss += cls_loss_fct(cls_logits.view(-1, cls_logits.size(-1)), input_labels.view(-1))
            return loss
        return cls_logits, lm_logits

SAMPLE_TEXT = """@@ -11,6 +11,8 @@\n \n         invoiceDtoCopy.setState(InvoiceState.OPEN);\n         _invoiceAggregateRepository.updateInvoiceState(invoiceCopy, InvoiceState.OPEN);\n+        _erpIntegrationService.createAndSendInvoiceEvent(invoiceCopy);\n+\n       }\n     }\n \n"""

lm_rename_keys = [
("shared.weight","transformer.shared.weight"),
("decoder.embed_tokens.weight","transformer.encoder.embed_tokens.weight"),
("decoder.block.0.layer.0.SelfAttention.q.weight","transformer.encoder.block.0.layer.0.SelfAttention.q.weight"),
("decoder.block.0.layer.0.SelfAttention.k.weight","transformer.encoder.block.0.layer.0.SelfAttention.k.weight"),
("decoder.block.0.layer.0.SelfAttention.v.weight","transformer.encoder.block.0.layer.0.SelfAttention.v.weight"),
("decoder.block.0.layer.0.SelfAttention.o.weight","transformer.encoder.block.0.layer.0.SelfAttention.o.weight"),
("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight","transformer.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
("decoder.block.0.layer.0.layer_norm.weight","transformer.encoder.block.0.layer.0.layer_norm.weight"),
("decoder.block.0.layer.1.EncDecAttention.q.weight","transformer.encoder.block.0.layer.1.EncDecAttention.q.weight"),
("decoder.block.0.layer.1.EncDecAttention.k.weight","transformer.encoder.block.0.layer.1.EncDecAttention.k.weight"),
("decoder.block.0.layer.1.EncDecAttention.v.weight","transformer.encoder.block.0.layer.1.EncDecAttention.v.weight"),
("decoder.block.0.layer.1.EncDecAttention.o.weight","transformer.encoder.block.0.layer.1.EncDecAttention.o.weight"),
("decoder.block.0.layer.1.layer_norm.weight","transformer.encoder.block.0.layer.1.layer_norm.weight"),
("decoder.block.0.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.0.layer.2.DenseReluDense.wi.weight"),
("decoder.block.0.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.0.layer.2.DenseReluDense.wo.weight"),
("decoder.block.0.layer.2.layer_norm.weight","transformer.encoder.block.0.layer.2.layer_norm.weight"),
("decoder.block.1.layer.0.SelfAttention.q.weight","transformer.encoder.block.1.layer.0.SelfAttention.q.weight"),
("decoder.block.1.layer.0.SelfAttention.k.weight","transformer.encoder.block.1.layer.0.SelfAttention.k.weight"),
("decoder.block.1.layer.0.SelfAttention.v.weight","transformer.encoder.block.1.layer.0.SelfAttention.v.weight"),
("decoder.block.1.layer.0.SelfAttention.o.weight","transformer.encoder.block.1.layer.0.SelfAttention.o.weight"),
("decoder.block.1.layer.0.layer_norm.weight","transformer.encoder.block.1.layer.0.layer_norm.weight"),
("decoder.block.1.layer.1.EncDecAttention.q.weight","transformer.encoder.block.1.layer.1.EncDecAttention.q.weight"),
("decoder.block.1.layer.1.EncDecAttention.k.weight","transformer.encoder.block.1.layer.1.EncDecAttention.k.weight"),
("decoder.block.1.layer.1.EncDecAttention.v.weight","transformer.encoder.block.1.layer.1.EncDecAttention.v.weight"),
("decoder.block.1.layer.1.EncDecAttention.o.weight","transformer.encoder.block.1.layer.1.EncDecAttention.o.weight"),
("decoder.block.1.layer.1.layer_norm.weight","transformer.encoder.block.1.layer.1.layer_norm.weight"),
("decoder.block.1.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.1.layer.2.DenseReluDense.wi.weight"),
("decoder.block.1.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.1.layer.2.DenseReluDense.wo.weight"),
("decoder.block.1.layer.2.layer_norm.weight","transformer.encoder.block.1.layer.2.layer_norm.weight"),
("decoder.block.2.layer.0.SelfAttention.q.weight","transformer.encoder.block.2.layer.0.SelfAttention.q.weight"),
("decoder.block.2.layer.0.SelfAttention.k.weight","transformer.encoder.block.2.layer.0.SelfAttention.k.weight"),
("decoder.block.2.layer.0.SelfAttention.v.weight","transformer.encoder.block.2.layer.0.SelfAttention.v.weight"),
("decoder.block.2.layer.0.SelfAttention.o.weight","transformer.encoder.block.2.layer.0.SelfAttention.o.weight"),
("decoder.block.2.layer.0.layer_norm.weight","transformer.encoder.block.2.layer.0.layer_norm.weight"),
("decoder.block.2.layer.1.EncDecAttention.q.weight","transformer.encoder.block.2.layer.1.EncDecAttention.q.weight"),
("decoder.block.2.layer.1.EncDecAttention.k.weight","transformer.encoder.block.2.layer.1.EncDecAttention.k.weight"),
("decoder.block.2.layer.1.EncDecAttention.v.weight","transformer.encoder.block.2.layer.1.EncDecAttention.v.weight"),
("decoder.block.2.layer.1.EncDecAttention.o.weight","transformer.encoder.block.2.layer.1.EncDecAttention.o.weight"),
("decoder.block.2.layer.1.layer_norm.weight","transformer.encoder.block.2.layer.1.layer_norm.weight"),
("decoder.block.2.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.2.layer.2.DenseReluDense.wi.weight"),
("decoder.block.2.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.2.layer.2.DenseReluDense.wo.weight"),
("decoder.block.2.layer.2.layer_norm.weight","transformer.encoder.block.2.layer.2.layer_norm.weight"),
("decoder.block.3.layer.0.SelfAttention.q.weight","transformer.encoder.block.3.layer.0.SelfAttention.q.weight"),
("decoder.block.3.layer.0.SelfAttention.k.weight","transformer.encoder.block.3.layer.0.SelfAttention.k.weight"),
("decoder.block.3.layer.0.SelfAttention.v.weight","transformer.encoder.block.3.layer.0.SelfAttention.v.weight"),
("decoder.block.3.layer.0.SelfAttention.o.weight","transformer.encoder.block.3.layer.0.SelfAttention.o.weight"),
("decoder.block.3.layer.0.layer_norm.weight","transformer.encoder.block.3.layer.0.layer_norm.weight"),
("decoder.block.3.layer.1.EncDecAttention.q.weight","transformer.encoder.block.3.layer.1.EncDecAttention.q.weight"),
("decoder.block.3.layer.1.EncDecAttention.k.weight","transformer.encoder.block.3.layer.1.EncDecAttention.k.weight"),
("decoder.block.3.layer.1.EncDecAttention.v.weight","transformer.encoder.block.3.layer.1.EncDecAttention.v.weight"),
("decoder.block.3.layer.1.EncDecAttention.o.weight","transformer.encoder.block.3.layer.1.EncDecAttention.o.weight"),
("decoder.block.3.layer.1.layer_norm.weight","transformer.encoder.block.3.layer.1.layer_norm.weight"),
("decoder.block.3.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.3.layer.2.DenseReluDense.wi.weight"),
("decoder.block.3.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.3.layer.2.DenseReluDense.wo.weight"),
("decoder.block.3.layer.2.layer_norm.weight","transformer.encoder.block.3.layer.2.layer_norm.weight"),
("decoder.block.4.layer.0.SelfAttention.q.weight","transformer.encoder.block.4.layer.0.SelfAttention.q.weight"),
("decoder.block.4.layer.0.SelfAttention.k.weight","transformer.encoder.block.4.layer.0.SelfAttention.k.weight"),
("decoder.block.4.layer.0.SelfAttention.v.weight","transformer.encoder.block.4.layer.0.SelfAttention.v.weight"),
("decoder.block.4.layer.0.SelfAttention.o.weight","transformer.encoder.block.4.layer.0.SelfAttention.o.weight"),
("decoder.block.4.layer.0.layer_norm.weight","transformer.encoder.block.4.layer.0.layer_norm.weight"),
("decoder.block.4.layer.1.EncDecAttention.q.weight","transformer.encoder.block.4.layer.1.EncDecAttention.q.weight"),
("decoder.block.4.layer.1.EncDecAttention.k.weight","transformer.encoder.block.4.layer.1.EncDecAttention.k.weight"),
("decoder.block.4.layer.1.EncDecAttention.v.weight","transformer.encoder.block.4.layer.1.EncDecAttention.v.weight"),
("decoder.block.4.layer.1.EncDecAttention.o.weight","transformer.encoder.block.4.layer.1.EncDecAttention.o.weight"),
("decoder.block.4.layer.1.layer_norm.weight","transformer.encoder.block.4.layer.1.layer_norm.weight"),
("decoder.block.4.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.4.layer.2.DenseReluDense.wi.weight"),
("decoder.block.4.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.4.layer.2.DenseReluDense.wo.weight"),
("decoder.block.4.layer.2.layer_norm.weight","transformer.encoder.block.4.layer.2.layer_norm.weight"),
("decoder.block.5.layer.0.SelfAttention.q.weight","transformer.encoder.block.5.layer.0.SelfAttention.q.weight"),
("decoder.block.5.layer.0.SelfAttention.k.weight","transformer.encoder.block.5.layer.0.SelfAttention.k.weight"),
("decoder.block.5.layer.0.SelfAttention.v.weight","transformer.encoder.block.5.layer.0.SelfAttention.v.weight"),
("decoder.block.5.layer.0.SelfAttention.o.weight","transformer.encoder.block.5.layer.0.SelfAttention.o.weight"),
("decoder.block.5.layer.0.layer_norm.weight","transformer.encoder.block.5.layer.0.layer_norm.weight"),
("decoder.block.5.layer.1.EncDecAttention.q.weight","transformer.encoder.block.5.layer.1.EncDecAttention.q.weight"),
("decoder.block.5.layer.1.EncDecAttention.k.weight","transformer.encoder.block.5.layer.1.EncDecAttention.k.weight"),
("decoder.block.5.layer.1.EncDecAttention.v.weight","transformer.encoder.block.5.layer.1.EncDecAttention.v.weight"),
("decoder.block.5.layer.1.EncDecAttention.o.weight","transformer.encoder.block.5.layer.1.EncDecAttention.o.weight"),
("decoder.block.5.layer.1.layer_norm.weight","transformer.encoder.block.5.layer.1.layer_norm.weight"),
("decoder.block.5.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.5.layer.2.DenseReluDense.wi.weight"),
("decoder.block.5.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.5.layer.2.DenseReluDense.wo.weight"),
("decoder.block.5.layer.2.layer_norm.weight","transformer.encoder.block.5.layer.2.layer_norm.weight"),
("decoder.block.6.layer.0.SelfAttention.q.weight","transformer.encoder.block.6.layer.0.SelfAttention.q.weight"),
("decoder.block.6.layer.0.SelfAttention.k.weight","transformer.encoder.block.6.layer.0.SelfAttention.k.weight"),
("decoder.block.6.layer.0.SelfAttention.v.weight","transformer.encoder.block.6.layer.0.SelfAttention.v.weight"),
("decoder.block.6.layer.0.SelfAttention.o.weight","transformer.encoder.block.6.layer.0.SelfAttention.o.weight"),
("decoder.block.6.layer.0.layer_norm.weight","transformer.encoder.block.6.layer.0.layer_norm.weight"),
("decoder.block.6.layer.1.EncDecAttention.q.weight","transformer.encoder.block.6.layer.1.EncDecAttention.q.weight"),
("decoder.block.6.layer.1.EncDecAttention.k.weight","transformer.encoder.block.6.layer.1.EncDecAttention.k.weight"),
("decoder.block.6.layer.1.EncDecAttention.v.weight","transformer.encoder.block.6.layer.1.EncDecAttention.v.weight"),
("decoder.block.6.layer.1.EncDecAttention.o.weight","transformer.encoder.block.6.layer.1.EncDecAttention.o.weight"),
("decoder.block.6.layer.1.layer_norm.weight","transformer.encoder.block.6.layer.1.layer_norm.weight"),
("decoder.block.6.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.6.layer.2.DenseReluDense.wi.weight"),
("decoder.block.6.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.6.layer.2.DenseReluDense.wo.weight"),
("decoder.block.6.layer.2.layer_norm.weight","transformer.encoder.block.6.layer.2.layer_norm.weight"),
("decoder.block.7.layer.0.SelfAttention.q.weight","transformer.encoder.block.7.layer.0.SelfAttention.q.weight"),
("decoder.block.7.layer.0.SelfAttention.k.weight","transformer.encoder.block.7.layer.0.SelfAttention.k.weight"),
("decoder.block.7.layer.0.SelfAttention.v.weight","transformer.encoder.block.7.layer.0.SelfAttention.v.weight"),
("decoder.block.7.layer.0.SelfAttention.o.weight","transformer.encoder.block.7.layer.0.SelfAttention.o.weight"),
("decoder.block.7.layer.0.layer_norm.weight","transformer.encoder.block.7.layer.0.layer_norm.weight"),
("decoder.block.7.layer.1.EncDecAttention.q.weight","transformer.encoder.block.7.layer.1.EncDecAttention.q.weight"),
("decoder.block.7.layer.1.EncDecAttention.k.weight","transformer.encoder.block.7.layer.1.EncDecAttention.k.weight"),
("decoder.block.7.layer.1.EncDecAttention.v.weight","transformer.encoder.block.7.layer.1.EncDecAttention.v.weight"),
("decoder.block.7.layer.1.EncDecAttention.o.weight","transformer.encoder.block.7.layer.1.EncDecAttention.o.weight"),
("decoder.block.7.layer.1.layer_norm.weight","transformer.encoder.block.7.layer.1.layer_norm.weight"),
("decoder.block.7.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.7.layer.2.DenseReluDense.wi.weight"),
("decoder.block.7.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.7.layer.2.DenseReluDense.wo.weight"),
("decoder.block.7.layer.2.layer_norm.weight","transformer.encoder.block.7.layer.2.layer_norm.weight"),
("decoder.block.8.layer.0.SelfAttention.q.weight","transformer.encoder.block.8.layer.0.SelfAttention.q.weight"),
("decoder.block.8.layer.0.SelfAttention.k.weight","transformer.encoder.block.8.layer.0.SelfAttention.k.weight"),
("decoder.block.8.layer.0.SelfAttention.v.weight","transformer.encoder.block.8.layer.0.SelfAttention.v.weight"),
("decoder.block.8.layer.0.SelfAttention.o.weight","transformer.encoder.block.8.layer.0.SelfAttention.o.weight"),
("decoder.block.8.layer.0.layer_norm.weight","transformer.encoder.block.8.layer.0.layer_norm.weight"),
("decoder.block.8.layer.1.EncDecAttention.q.weight","transformer.encoder.block.8.layer.1.EncDecAttention.q.weight"),
("decoder.block.8.layer.1.EncDecAttention.k.weight","transformer.encoder.block.8.layer.1.EncDecAttention.k.weight"),
("decoder.block.8.layer.1.EncDecAttention.v.weight","transformer.encoder.block.8.layer.1.EncDecAttention.v.weight"),
("decoder.block.8.layer.1.EncDecAttention.o.weight","transformer.encoder.block.8.layer.1.EncDecAttention.o.weight"),
("decoder.block.8.layer.1.layer_norm.weight","transformer.encoder.block.8.layer.1.layer_norm.weight"),
("decoder.block.8.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.8.layer.2.DenseReluDense.wi.weight"),
("decoder.block.8.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.8.layer.2.DenseReluDense.wo.weight"),
("decoder.block.8.layer.2.layer_norm.weight","transformer.encoder.block.8.layer.2.layer_norm.weight"),
("decoder.block.9.layer.0.SelfAttention.q.weight","transformer.encoder.block.9.layer.0.SelfAttention.q.weight"),
("decoder.block.9.layer.0.SelfAttention.k.weight","transformer.encoder.block.9.layer.0.SelfAttention.k.weight"),
("decoder.block.9.layer.0.SelfAttention.v.weight","transformer.encoder.block.9.layer.0.SelfAttention.v.weight"),
("decoder.block.9.layer.0.SelfAttention.o.weight","transformer.encoder.block.9.layer.0.SelfAttention.o.weight"),
("decoder.block.9.layer.0.layer_norm.weight","transformer.encoder.block.9.layer.0.layer_norm.weight"),
("decoder.block.9.layer.1.EncDecAttention.q.weight","transformer.encoder.block.9.layer.1.EncDecAttention.q.weight"),
("decoder.block.9.layer.1.EncDecAttention.k.weight","transformer.encoder.block.9.layer.1.EncDecAttention.k.weight"),
("decoder.block.9.layer.1.EncDecAttention.v.weight","transformer.encoder.block.9.layer.1.EncDecAttention.v.weight"),
("decoder.block.9.layer.1.EncDecAttention.o.weight","transformer.encoder.block.9.layer.1.EncDecAttention.o.weight"),
("decoder.block.9.layer.1.layer_norm.weight","transformer.encoder.block.9.layer.1.layer_norm.weight"),
("decoder.block.9.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.9.layer.2.DenseReluDense.wi.weight"),
("decoder.block.9.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.9.layer.2.DenseReluDense.wo.weight"),
("decoder.block.9.layer.2.layer_norm.weight","transformer.encoder.block.9.layer.2.layer_norm.weight"),
("decoder.block.10.layer.0.SelfAttention.q.weight","transformer.encoder.block.10.layer.0.SelfAttention.q.weight"),
("decoder.block.10.layer.0.SelfAttention.k.weight","transformer.encoder.block.10.layer.0.SelfAttention.k.weight"),
("decoder.block.10.layer.0.SelfAttention.v.weight","transformer.encoder.block.10.layer.0.SelfAttention.v.weight"),
("decoder.block.10.layer.0.SelfAttention.o.weight","transformer.encoder.block.10.layer.0.SelfAttention.o.weight"),
("decoder.block.10.layer.0.layer_norm.weight","transformer.encoder.block.10.layer.0.layer_norm.weight"),
("decoder.block.10.layer.1.EncDecAttention.q.weight","transformer.encoder.block.10.layer.1.EncDecAttention.q.weight"),
("decoder.block.10.layer.1.EncDecAttention.k.weight","transformer.encoder.block.10.layer.1.EncDecAttention.k.weight"),
("decoder.block.10.layer.1.EncDecAttention.v.weight","transformer.encoder.block.10.layer.1.EncDecAttention.v.weight"),
("decoder.block.10.layer.1.EncDecAttention.o.weight","transformer.encoder.block.10.layer.1.EncDecAttention.o.weight"),
("decoder.block.10.layer.1.layer_norm.weight","transformer.encoder.block.10.layer.1.layer_norm.weight"),
("decoder.block.10.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.10.layer.2.DenseReluDense.wi.weight"),
("decoder.block.10.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.10.layer.2.DenseReluDense.wo.weight"),
("decoder.block.10.layer.2.layer_norm.weight","transformer.encoder.block.10.layer.2.layer_norm.weight"),
("decoder.block.11.layer.0.SelfAttention.q.weight","transformer.encoder.block.11.layer.0.SelfAttention.q.weight"),
("decoder.block.11.layer.0.SelfAttention.k.weight","transformer.encoder.block.11.layer.0.SelfAttention.k.weight"),
("decoder.block.11.layer.0.SelfAttention.v.weight","transformer.encoder.block.11.layer.0.SelfAttention.v.weight"),
("decoder.block.11.layer.0.SelfAttention.o.weight","transformer.encoder.block.11.layer.0.SelfAttention.o.weight"),
("decoder.block.11.layer.0.layer_norm.weight","transformer.encoder.block.11.layer.0.layer_norm.weight"),
("decoder.block.11.layer.1.EncDecAttention.q.weight","transformer.encoder.block.11.layer.1.EncDecAttention.q.weight"),
("decoder.block.11.layer.1.EncDecAttention.k.weight","transformer.encoder.block.11.layer.1.EncDecAttention.k.weight"),
("decoder.block.11.layer.1.EncDecAttention.v.weight","transformer.encoder.block.11.layer.1.EncDecAttention.v.weight"),
("decoder.block.11.layer.1.EncDecAttention.o.weight","transformer.encoder.block.11.layer.1.EncDecAttention.o.weight"),
("decoder.block.11.layer.1.layer_norm.weight","transformer.encoder.block.11.layer.1.layer_norm.weight"),
("decoder.block.11.layer.2.DenseReluDense.wi.weight","transformer.encoder.block.11.layer.2.DenseReluDense.wi.weight"),
("decoder.block.11.layer.2.DenseReluDense.wo.weight","transformer.encoder.block.11.layer.2.DenseReluDense.wo.weight"),
("decoder.block.11.layer.2.layer_norm.weight","transformer.encoder.block.11.layer.2.layer_norm.weight"),
("decoder.final_layer_norm.weight","transformer.encoder.final_layer_norm.weight"),
("cls_head.weight","classifier.weight"),
("cls_head.bias","classifier.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    remove_keys(state_dict, ignore_keys)

ignore_lm_keys = [
    "encoder.embed_tokens.weight",
    "encoder.block.0.layer.0.SelfAttention.q.weight",
    "encoder.block.0.layer.0.SelfAttention.k.weight",
    "encoder.block.0.layer.0.SelfAttention.v.weight",
    "encoder.block.0.layer.0.SelfAttention.o.weight",
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
    "encoder.block.0.layer.0.layer_norm.weight",
    "encoder.block.0.layer.1.DenseReluDense.wi.weight",
    "encoder.block.0.layer.1.DenseReluDense.wo.weight",
    "encoder.block.0.layer.1.layer_norm.weight",
    "encoder.block.1.layer.0.SelfAttention.q.weight",
    "encoder.block.1.layer.0.SelfAttention.k.weight",
    "encoder.block.1.layer.0.SelfAttention.v.weight",
    "encoder.block.1.layer.0.SelfAttention.o.weight",
    "encoder.block.1.layer.0.layer_norm.weight",
    "encoder.block.1.layer.1.DenseReluDense.wi.weight",
    "encoder.block.1.layer.1.DenseReluDense.wo.weight",
    "encoder.block.1.layer.1.layer_norm.weight",
    "encoder.block.2.layer.0.SelfAttention.q.weight",
    "encoder.block.2.layer.0.SelfAttention.k.weight",
    "encoder.block.2.layer.0.SelfAttention.v.weight",
    "encoder.block.2.layer.0.SelfAttention.o.weight",
    "encoder.block.2.layer.0.layer_norm.weight",
    "encoder.block.2.layer.1.DenseReluDense.wi.weight",
    "encoder.block.2.layer.1.DenseReluDense.wo.weight",
    "encoder.block.2.layer.1.layer_norm.weight",
    "encoder.block.3.layer.0.SelfAttention.q.weight",
    "encoder.block.3.layer.0.SelfAttention.k.weight",
    "encoder.block.3.layer.0.SelfAttention.v.weight",
    "encoder.block.3.layer.0.SelfAttention.o.weight",
    "encoder.block.3.layer.0.layer_norm.weight",
    "encoder.block.3.layer.1.DenseReluDense.wi.weight",
    "encoder.block.3.layer.1.DenseReluDense.wo.weight",
    "encoder.block.3.layer.1.layer_norm.weight",
    "encoder.block.4.layer.0.SelfAttention.q.weight",
    "encoder.block.4.layer.0.SelfAttention.k.weight",
    "encoder.block.4.layer.0.SelfAttention.v.weight",
    "encoder.block.4.layer.0.SelfAttention.o.weight",
    "encoder.block.4.layer.0.layer_norm.weight",
    "encoder.block.4.layer.1.DenseReluDense.wi.weight",
    "encoder.block.4.layer.1.DenseReluDense.wo.weight",
    "encoder.block.4.layer.1.layer_norm.weight",
    "encoder.block.5.layer.0.SelfAttention.q.weight",
    "encoder.block.5.layer.0.SelfAttention.k.weight",
    "encoder.block.5.layer.0.SelfAttention.v.weight",
    "encoder.block.5.layer.0.SelfAttention.o.weight",
    "encoder.block.5.layer.0.layer_norm.weight",
    "encoder.block.5.layer.1.DenseReluDense.wi.weight",
    "encoder.block.5.layer.1.DenseReluDense.wo.weight",
    "encoder.block.5.layer.1.layer_norm.weight",
    "encoder.block.6.layer.0.SelfAttention.q.weight",
    "encoder.block.6.layer.0.SelfAttention.k.weight",
    "encoder.block.6.layer.0.SelfAttention.v.weight",
    "encoder.block.6.layer.0.SelfAttention.o.weight",
    "encoder.block.6.layer.0.layer_norm.weight",
    "encoder.block.6.layer.1.DenseReluDense.wi.weight",
    "encoder.block.6.layer.1.DenseReluDense.wo.weight",
    "encoder.block.6.layer.1.layer_norm.weight",
    "encoder.block.7.layer.0.SelfAttention.q.weight",
    "encoder.block.7.layer.0.SelfAttention.k.weight",
    "encoder.block.7.layer.0.SelfAttention.v.weight",
    "encoder.block.7.layer.0.SelfAttention.o.weight",
    "encoder.block.7.layer.0.layer_norm.weight",
    "encoder.block.7.layer.1.DenseReluDense.wi.weight",
    "encoder.block.7.layer.1.DenseReluDense.wo.weight",
    "encoder.block.7.layer.1.layer_norm.weight",
    "encoder.block.8.layer.0.SelfAttention.q.weight",
    "encoder.block.8.layer.0.SelfAttention.k.weight",
    "encoder.block.8.layer.0.SelfAttention.v.weight",
    "encoder.block.8.layer.0.SelfAttention.o.weight",
    "encoder.block.8.layer.0.layer_norm.weight",
    "encoder.block.8.layer.1.DenseReluDense.wi.weight",
    "encoder.block.8.layer.1.DenseReluDense.wo.weight",
    "encoder.block.8.layer.1.layer_norm.weight",
    "encoder.block.9.layer.0.SelfAttention.q.weight",
    "encoder.block.9.layer.0.SelfAttention.k.weight",
    "encoder.block.9.layer.0.SelfAttention.v.weight",
    "encoder.block.9.layer.0.SelfAttention.o.weight",
    "encoder.block.9.layer.0.layer_norm.weight",
    "encoder.block.9.layer.1.DenseReluDense.wi.weight",
    "encoder.block.9.layer.1.DenseReluDense.wo.weight",
    "encoder.block.9.layer.1.layer_norm.weight",
    "encoder.block.10.layer.0.SelfAttention.q.weight",
    "encoder.block.10.layer.0.SelfAttention.k.weight",
    "encoder.block.10.layer.0.SelfAttention.v.weight",
    "encoder.block.10.layer.0.SelfAttention.o.weight",
    "encoder.block.10.layer.0.layer_norm.weight",
    "encoder.block.10.layer.1.DenseReluDense.wi.weight",
    "encoder.block.10.layer.1.DenseReluDense.wo.weight",
    "encoder.block.10.layer.1.layer_norm.weight",
    "encoder.block.11.layer.0.SelfAttention.q.weight",
    "encoder.block.11.layer.0.SelfAttention.k.weight",
    "encoder.block.11.layer.0.SelfAttention.v.weight",
    "encoder.block.11.layer.0.SelfAttention.o.weight",
    "encoder.block.11.layer.0.layer_norm.weight",
    "encoder.block.11.layer.1.DenseReluDense.wi.weight",
    "encoder.block.11.layer.1.DenseReluDense.wo.weight",
    "encoder.block.11.layer.1.layer_norm.weight",
    "encoder.final_layer_norm.weight",
    "lm_head.weight"
]

def remove_keys(state_dict, ignore_keys):
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

def build_or_load_gen_model(model_name_or_path, load_model_path):
    config_class, model_class, tokenizer_class = T5Config, ReviewerModel, RobertaTokenizer

    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    model = model_class.from_pretrained(model_name_or_path, config=config)

    tokenizer.special_dict = {
        f"<e{i}>" : tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    }

    tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    model_path = os.path.join(load_model_path, "pytorch_model.bin")
    logger.info("Reload model from {}".format(model_path))
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    except RuntimeError:
        saved = model.cls_head
        model.cls_head = None
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.cls_head = saved

    return config, model, tokenizer

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            print('✓')
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception(key_item_1, key_item_2, model_1, model_2)
    if models_differ == 0:
        print('Models match perfectly! :)')

@torch.no_grad()
def convert_CodeReviewer_checkpoint_for_conditional_generation(checkpoint_path, model_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    config, model, tokenizer = build_or_load_gen_model(checkpoint_path, model_path)

    tokens = tokenizer.encode(SAMPLE_TEXT, return_tensors="pt")

    state_dict = model.state_dict()
    remove_ignore_keys_(state_dict)

    new_lm_model = CodeReviewerForConditionalGeneration(config).eval()

    lm_state_dict = state_dict
    lm_state_dict.pop('cls_head.weight',None)
    lm_state_dict.pop('cls_head.bias',None)
    new_lm_model.load_state_dict(lm_state_dict)
    old_model_outputs = model.generate(tokens)[0]
    new_lm_model_outputs = new_lm_model.generate(tokens)[0]

    # Check results
    compare_models(model, new_lm_model)
    if old_model_outputs.shape != new_lm_model_outputs.shape:
        raise ValueError(
            f"`old_model_outputs` shape and `new_lm_model_output` shape are different: {old_model_outputs.shape}, {new_lm_model_outputs.shape}"
        )
    if (old_model_outputs != new_lm_model_outputs).any().item():
        raise ValueError("Some values in `old_model_output` are different from `new_lm_model_outputs`")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    new_lm_model.save_pretrained(pytorch_dump_folder_path)
    print('Model saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "model_name_or_path", type=str, default="microsoft/codereviewer", help="The base model (i.e. microsoft/codereviewer)"
    )
    parser.add_argument(
        "load_model_path", type=str, help="The location of the pymodel bin file"
    ),
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_CodeReviewer_checkpoint_for_conditional_generation(args.model_name_or_path, args.load_model_path, args.pytorch_dump_folder_path)
