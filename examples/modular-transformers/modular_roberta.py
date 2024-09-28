import torch.nn as nn

from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel


class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, config.pad_token_id
        )


class RobertaModel(BertModel):
    def __init__(self, config):
        super().__init__(self, config)
        # Error out here. Why? Because `RobertaEmbeddings` is defined but not used.
        # no, because it's defined, and RobertaModel should use RobertaEmbedding
        # here if initialized that way it won't use the new embedding.
