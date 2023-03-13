import torch

from transformers import UdopConfig, UdopForConditionalGeneration


state_dict = torch.load("/Users/nielsrogge/Downloads/udop-unimodel-large-224/pytorch_model.bin", map_location="cpu")

print("Original state dict:")
for name, param in state_dict.items():
    print(name, param.shape)

# rename keys
for key, value in state_dict.copy().items():
    val = state_dict.pop(key)
    if "lm_head" not in key:
        key = "udop." + key
    state_dict[key] = val

# create HF model
config = UdopConfig()
model = UdopForConditionalGeneration(config)

# load weights
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
assert missing_keys == []
assert unexpected_keys == ["udop.pos_embed"]
