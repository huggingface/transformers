import torch

from transformers import VitDetConfig, VitDetModel


config = VitDetConfig()

model = VitDetModel(config)

outputs = model(torch.randn(1, 3, 224, 224))

print(outputs.last_hidden_state.shape)
