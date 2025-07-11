
from ...utils import OptionalDependencyNotAvailable, is_torch_available

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class Siglip2TextTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=config.num_hidden_layers,
            )
            self.final_layernorm = nn.LayerNorm(config.hidden_size)

        def forward(self, input_ids, attention_mask=None):
            embeddings = self.embed(input_ids)
            if attention_mask is not None:
                # Convert attention_mask to the shape expected by nn.TransformerEncoder
                # It should be (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                extended_attention_mask = attention_mask[:, None, None, :]
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
                extended_attention_mask = extended_attention_mask.squeeze(1).squeeze(1)
            else:
                extended_attention_mask = None

            encoded = self.encoder(embeddings, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)
            return self.final_layernorm(encoded)
