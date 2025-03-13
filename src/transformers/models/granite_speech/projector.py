import torch
import torch.nn as nn
from .configuration_granite_speech import GraniteSpeechConfig
from transformers import Blip2QFormerModel
import math

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ds_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = self.window_size // self.ds_rate
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(config)
        self.linear = nn.Linear(config.hidden_size, config.llm_dim)

    def forward(self, x, atts):
        batch_size, seq_len, dim = x.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        x = nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)
        x = x.view(batch_size * nblocks, self.window_size, dim)

        query_output = self.qformer(
            query_embeds=self.query.data,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        query_proj = self.linear(
            query_output.last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.ds_rate, -1
            )
        )

        return query_proj