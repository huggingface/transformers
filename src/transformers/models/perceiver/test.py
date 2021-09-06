import torch

from transformers import PerceiverConfig, PerceiverModel


config = PerceiverConfig()
model = PerceiverModel(config)

inputs = torch.randn((2, 100))
outputs = model(inputs)

print("Shape of outputs:", outputs.last_hidden_state.shape)
