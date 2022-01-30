import os
import torch
import transformers

torch.manual_seed(0)

config = transformers.GPTJConfig()
config.n_embd = 16
config.n_head = 2
config.n_layer = 2
config.n_positions = 16

model = transformers.GPTJModel(config)
o = model(input_ids=torch.tensor([1, 2, 3, 4]), output_attentions=False)
o2 = model(input_ids=torch.tensor([1, 2, 3, 4]), output_attentions=True)
config.chunk_size_key=2
config.chunk_size_query=2
o3 = model(input_ids=torch.tensor([1, 2, 3, 4]), output_attentions=False)
o4 = model(input_ids=torch.tensor([1, 2, 3, 4]), output_attentions=True) # throws
