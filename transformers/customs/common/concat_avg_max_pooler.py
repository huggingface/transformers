import torch
import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig


class ConcatAvgMaxPooler(nn.Module):
    def __init__(self, config):
        super(ConcatAvgMaxPooler, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size * 2)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = hidden_states[:, :].permute(0, 2, 1)
        avg_pooled = self.avg_pool(output)
        max_pooled = self.max_pool(output)
        output = torch.cat([avg_pooled, max_pooled], dim=1).squeeze(2)
        output = self.dense(output)
        return self.activation(output)
