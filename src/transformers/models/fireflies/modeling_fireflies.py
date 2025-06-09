import torch.nn as nn
from transformers import PreTrainedModel
from .configuration_fireflies import FirefliesConfig

class FirefliesModel(PreTrainedModel):
    config_class = FirefliesConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.d_model, config.vocab_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return {"loss": loss, "logits": logits}