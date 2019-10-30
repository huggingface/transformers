from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter
from itertools import starmap
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class SimpleConcatAvgMaxTokensPooler(nn.Module):
    def __init__(self, config):
        super(SimpleConcatAvgMaxTokensPooler, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, tokens_embs: torch.Tensor, *args) -> torch.Tensor:  # type: ignore
        return torch.cat(
            [self.avg_pool(tokens_embs).squeeze(), self.max_pool(tokens_embs).squeeze()], dim=0
        ).squeeze()


class SimpleAvgOrMaxTokensPoolerWithMask(nn.Module):
    def __init__(self, config):
        super(SimpleAvgOrMaxTokensPoolerWithMask, self).__init__()
        word_tokens_pooling_method = (
            getattr(config, "word_tokens_pooling_method", "").lower().capitalize()
        )
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

    def _convert_bert_outputs_to_map(
        self, outputs: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        outputs_map = dict(last_hidden_state=outputs[0], pooler_output=outputs[1])
        if len(outputs) > 2:
            outputs_map["hidden_states"] = outputs[2]
        if len(outputs) > 3:
            outputs_map["attentions"] = outputs[3]
        return outputs_map


    def forward(
        self,
        input_ids,
        word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        outputs = self._convert_bert_outputs_to_map(outputs)
        sequence_output = outputs["last_hidden_state"]

        if word_ids is not None and hasattr(self, 'tokens_pooler'):

            word_ids[word_ids == -100] = -1

            _word_ids = word_ids.unsqueeze(-1) + 1
            _mean = scatter.scatter_mean(sequence_output, _word_ids, dim=1).type(sequence_output.dtype)
            _mean[:, 0, :] = sequence_output[:, 0, :]
            _max = scatter.scatter_max(sequence_output, _word_ids, dim=1, fill_value=0)[0].type(sequence_output.dtype)
            _max[:, 0, :] = sequence_output[:, 0, :]
            sequence_output = torch.cat([_mean, _max], dim=-1)

            word_ids[word_ids == -1] = -100

            def transform_ids(word_ids: torch.Tensor, labels: torch.Tensor, pad_id: int = -100) -> torch.Tensor:
                word_labels = labels[word_ids[word_ids != pad_id].unique_consecutive(return_counts=True)[1].cumsum(dim=0) - 1]
                tensor = F.pad(word_labels, (0, sequence_output.shape[1] - 1 - word_labels.shape[0]), value=pad_id)
                return tensor

            labels = torch.stack(list(starmap(transform_ids, zip(word_ids[:, 1:], labels[:, 1:]))), dim=0)
            labels = torch.cat((torch.tensor(-100).repeat(labels.shape[0], 1).to(labels.device), labels), dim=1)
            attention_mask = torch.zeros_like(labels)
            attention_mask[labels != -100] = 1

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

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
            outputs["loss"] = loss

        outputs["attention_mask"] = attention_mask
        outputs["logits"] = logits
        outputs["labels"] = labels

        return outputs  # (loss), scores, (hidden_states), (attentions)
