import torch

from transformers import PreTrainedModel

from .custom_configuration import CustomConfig


class CustomModel(PreTrainedModel):
    config_class = CustomConfig
    base_model_prefix = "custom"

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass
