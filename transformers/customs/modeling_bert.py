import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from typing import List


def create_mask_from_word_ids(tensor: torch.Tensor, excluded_ids: List[int]) -> torch.Tensor:
    word_ids_wo_padding = tensor.clone()
    if 0 in excluded_ids:
        excluded_ids.remove(0)
    for excluded_id in excluded_ids:
        word_ids_wo_padding = word_ids_wo_padding[word_ids_wo_padding != excluded_id]
    word_unique_ids = word_ids_wo_padding.unique()
    word_count = word_unique_ids.numel()
    return (tensor.repeat((word_count, 1)) == word_unique_ids.unsqueeze(0).T).type(torch.long)


class SimpleConcatAvgMaxTokensPooler(nn.Module):
    def __init__(self, config):
        super(SimpleConcatAvgMaxTokensPooler, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, tokens_embs: torch.Tensor, *args) -> torch.Tensor:  # type: ignore
        return torch.cat([self.avg_pool(tokens_embs).squeeze(), self.max_pool(tokens_embs).squeeze()], dim=0).squeeze()


class SimpleAvgOrMaxTokensPoolerWithMask(nn.Module):
    def __init__(self, config):
        super(SimpleAvgOrMaxTokensPoolerWithMask, self).__init__()
        word_tokens_pooling_method = getattr(config, "word_tokens_pooling_method", "").lower().capitalize()
        self.pooler = getattr(nn, f"Adaptive{word_tokens_pooling_method}Pool1d")(1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.pooler(tensor).squeeze()


class CustomBertForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForNer, self).__init__(config)
        word_tokens_pooling_method = getattr(config, "word_tokens_pooling_method", "").lower()
        linear_hidden_size_mult = 1

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if word_tokens_pooling_method in ["avg", "max"]:
            self.tokens_pooler = SimpleAvgOrMaxTokensPoolerWithMask(config)
        elif word_tokens_pooling_method == "concatavgmax":
            self.tokens_pooler = SimpleConcatAvgMaxTokensPooler(config)
            linear_hidden_size_mult = 2
        self.classifier = nn.Linear(config.hidden_size * linear_hidden_size_mult, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, word_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # TODO OH GOD WHY?! CLEAN THIS CODE plz...
        if word_ids is not None:
            """for each word get its tokens' representation and pool them"""
            sequence_outputs, sequence_masks, sequence_labels = [], [], []

            for idx in torch.arange(word_ids.shape[0]):
                word_embs = []
                word_labels = []
                for word_mask in create_mask_from_word_ids(word_ids[idx], self.config.word_mask_excluded_ids):
                    word_tokens_embs = sequence_output[idx][word_mask.nonzero()].permute(2, 1, 0)
                    pooled_word = self.tokens_pooler(word_tokens_embs).detach().cpu()
                    word_label = labels[idx][word_mask.argmax()].detach().cpu()

                    word_embs.append(pooled_word)
                    word_labels.append(word_label)

                sentence_emb = torch.stack(word_embs).to(input_ids.device)
                sentence_labels = torch.stack(word_labels).to(input_ids.device)
                sentence_mask = torch.ones_like(sentence_labels).to(input_ids.device)

                sentence_emb = F.pad(sentence_emb, (0, 0, 0, input_ids.shape[1] - sentence_emb.shape[0]), value=0)
                sentence_mask = F.pad(sentence_mask, (0, input_ids.shape[1] - sentence_mask.shape[0]), value=0)
                sentence_labels = F.pad(sentence_labels, (0, input_ids.shape[1] - sentence_labels.shape[0]), value=-100)

                sequence_outputs.append(sentence_emb)
                sequence_masks.append(sentence_mask)
                sequence_labels.append(sentence_labels)

            sequence_output = torch.stack(sequence_outputs).to(input_ids.device)
            labels = torch.stack(sequence_labels).to(input_ids.device)
            attention_mask = torch.stack(sequence_masks).to(input_ids.device)

        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1).type(torch.bool)
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
