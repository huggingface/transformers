import torch

from transformers import CanineConfig, CanineModel


config = CanineConfig()
model = CanineModel(config)

input_ids = torch.randint(0, 1, (2, 30))

outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

print(len(outputs.hidden_states))
print(outputs.hidden_states[-1].shape)