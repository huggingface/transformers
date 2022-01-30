import os
import torch
import transformers

torch.manual_seed(0)

config = transformers.GPTJConfig()
config.n_embd = 16
config.n_head = 2
config.n_layer = 2
config.n_positions = 16
config.output_attentions = True

os.environ['NEW_ATTN'] = ''
model = transformers.GPTJModel(config)
o = model(input_ids=torch.tensor([1]))

os.environ['NEW_ATTN'] = '1'
o2 = model(input_ids=torch.tensor([1]))
