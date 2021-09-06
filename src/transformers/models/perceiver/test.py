import torch
from transformers import PerceiverConfig, PerceiverModel

config = PerceiverConfig()
model = PerceiverModel(config)

# assuming we have already turned our input_ids into embeddings
inputs = torch.randn((2, 100, 1024))
outputs = model(inputs)

print("Shape of outputs:", outputs.last_hidden_state.shape)
