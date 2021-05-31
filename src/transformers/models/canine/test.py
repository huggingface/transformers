from transformers import CanineConfig, CanineModel
import torch

config = CanineConfig()
model = CanineModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)

input_ids = torch.randint(0, 1, (2,2048))
outputs = model(input_ids)

#print(outputs.last_hidden_state.shape)