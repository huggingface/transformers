import torch
from torch import nn

class GatedAttention(nn.Module):
    """GatedAttention
    Args:
        in_size (int): input size
        out_size (int): output size
    """

    def __init__(self, in_size: int, out_size: int):

        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attn_weight = self.attn_softmax(self.attn_layer(x))
        x = x * attn_weight
        return x

