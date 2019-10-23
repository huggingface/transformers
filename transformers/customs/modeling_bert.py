from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.configuration_bert import BertConfig


def create_mask_from_word_ids(tensor: torch.Tensor, excluded_ids: List[int]) -> torch.Tensor:
    word_ids_wo_padding = tensor.clone().to(tensor.device)
    if 0 in excluded_ids:
        excluded_ids.remove(0)
    for excluded_id in excluded_ids:
        word_ids_wo_padding = word_ids_wo_padding[word_ids_wo_padding != excluded_id]
    word_unique_ids = word_ids_wo_padding.unique()
    word_count = word_unique_ids.numel()
    return (tensor.repeat((word_count, 1)) == word_unique_ids.unsqueeze(0).T).type(torch.bool)


class CustomBertForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForNer, self).__init__(config)
        word_tokens_pooling_method = getattr(config, "word_tokens_pooling_method", "avg")
        self.tokens_pooler = getattr(torch, f"adaptive_{word_tokens_pooling_method}_pool1d")

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, word_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if word_ids is not None:
            """for each word get its tokens' representation and pool them"""
            sequence_outputs = []

            for idx in torch.arange(word_ids.shape[0]):
                sentence_emb = torch.stack([self.tokens_pooler(sequence_output[idx][word_mask.nonzero()].permute(2, 1, 0), 1).squeeze() for word_mask in create_mask_from_word_ids(word_ids[idx], self.config.word_mask_excluded_ids)])
                sentence_emb = F.pad(sentence_emb, (0, 0, 0, input_ids.shape[1] - sentence_emb.shape[0]), value=0)
                sequence_outputs.append(sentence_emb)

            sequence_outputs = torch.stack(sequence_outputs)

        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
