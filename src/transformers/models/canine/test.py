from transformers import CanineConfig, CanineModel
import torch

config = CanineConfig()
model = CanineModel(config)

input_ids = torch.randint(0, 1, (2,100))
outputs = model(input_ids)

print(outputs.last_hidden_state.shape)