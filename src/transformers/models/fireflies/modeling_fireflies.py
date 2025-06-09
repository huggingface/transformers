import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from .configuration_fireflies import FirefliesConfig


class FirefliesPreTrainedModel(PreTrainedModel):
    config_class = FirefliesConfig
    base_model_prefix = "fireflies"


class FirefliesModel(FirefliesPreTrainedModel):
    def __init__(self, config: FirefliesConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(2048, config.d_model)  # Posisi maksimal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.dropout = nn.Dropout(0.1)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        batch_size, seq_length = input_ids.size()

        # Positional Encoding
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        # Embedding + Positional Encoding
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        x = self.dropout(x)

        if attention_mask is not None:
            # TransformerEncoder expects padding mask where True = ignored
            encoder_padding_mask = attention_mask == 0
        else:
            encoder_padding_mask = None

        # Transformer Encoder (batch_first = True)
        x = self.transformer(x, src_key_padding_mask=encoder_padding_mask)

        # LM head
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )


class FirefliesForCausalLM(FirefliesPreTrainedModel):
    def __init__(self, config: FirefliesConfig):
        super().__init__(config)
        self.model = FirefliesModel(config)
        self.lm_head = self.model.lm_head
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutput:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
