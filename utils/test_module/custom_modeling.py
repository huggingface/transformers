import torch

from transformers import PreTrainedModel

from .custom_configuration import CustomConfig, NoSuperInitConfig


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass


class NoSuperInitModel(PreTrainedModel):
    config_class = NoSuperInitConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.attribute, config.attribute)

    def forward(self, x):
        return self.linear(x)

    def _init_weights(self, module):
        pass
