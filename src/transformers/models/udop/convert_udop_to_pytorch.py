import torch
from transformers import UdopConfig, UdopForConditionalGeneration

state_dict = torch.load("/Users/nielsrogge/Downloads/udop-unimodel-large-224/pytorch_model.bin", map_location="cpu")

# for name, param in state_dict.items():
#     print(name, param.shape)

# create HF model
config = UdopConfig()

model = UdopForConditionalGeneration(config)

# load weights
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
assert missing_keys == []
assert unexpected_keys == ["pos_embed"]