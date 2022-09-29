import torch

from transformers import LiltConfig, LiltModel


config = LiltConfig()

model = LiltModel(config)

for name, param in model.named_parameters():
    print(name, param.shape)

input_ids = torch.tensor([[1, 2]])
bbox = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])


outputs = model(input_ids, bbox=bbox)
