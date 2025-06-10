from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput
from .configuration_fireflies import FirefliesConfig


class FirefliesModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class FirefliesModel(PreTrainedModel):
    config_class = FirefliesConfig

    def __init__(self, config: FirefliesConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.fc = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, FirefliesModelOutput]:
        x = self.embedding(input_ids)  # [batch, seq, dim]
        x = x.transpose(0, 1)  # -> [seq, batch, dim] (for Transformer)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # -> [batch, seq, dim]

        logits = self.fc(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return FirefliesModelOutput(last_hidden_state=logits, loss=loss)


__all__ = ["FirefliesModel"]
