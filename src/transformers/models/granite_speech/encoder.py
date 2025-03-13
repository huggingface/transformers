import types
import torch
import torch.nn as nn

from .conformer import ConformerBlock
from .configuration_granite_speech import GraniteSpeechEncoderConfig


class CTCModel(nn.Module):
    def __init__(self, config: GraniteSpeechEncoderConfig):
        super(CTCModel, self).__init__()

        self.rnn_trL = [nn.Linear(config.input_dim, config.hidden_dim, bias=True)]
        for l in range(config.num_layers):
            self.rnn_trL.append(
                ConformerBlock(
                    dim=config.hidden_dim,
                    dim_head=config.dim_head,
                    heads=config.num_heads,
                    ff_mult=config.feedforward_mult,
                    conv_expansion_factor=config.conv_expansion_factor,
                    conv_kernel_size=config.conv_kernel_size,
                    context_size=config.context_size,  # attention context size
                    attn_dropout=config.dropout,
                    ff_dropout=config.dropout,
                    conv_dropout=config.dropout,
                )
            )
            self.rnn_tr = nn.Sequential(*self.rnn_trL)

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim, bias=True)
        self.context_size = config.context_size
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

    def forward(self, x: torch.Tensor):
        x = self.rnn_trL[0](x)
        for l in range(1, self.num_layers + 1):
            x = self.rnn_trL[l](x, self.context_size)
            if l == self.num_layers // 2:
                x_mid = x.clone()
                x_mid = self.out(x_mid)
                x += self.out_mid(nn.Softmax(dim=-1)(x_mid))
        return x