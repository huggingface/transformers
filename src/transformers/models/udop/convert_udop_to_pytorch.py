import torch

from transformers import UdopConfig, UdopForConditionalGeneration


state_dict = torch.load("/Users/nielsrogge/Downloads/udop-unimodel-large-224/pytorch_model.bin", map_location="cpu")

print("Original state dict:")
# for name, param in state_dict.items():
#     print(name, param.shape)

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
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
assert missing_keys == ["udop.encoder.embed_patches.proj.weight", "udop.encoder.embed_patches.proj.bias"]
assert unexpected_keys == ["udop.pos_embed"]
print("Looks ok!")

# single forward pass
print("Testing single forward pass..")
input_ids = torch.tensor([[101, 102]])
seg_data = torch.tensor([[[0, 0, 0, 0], [1, 2, 3, 4]]]).float()
image = torch.randn(1, 3, 224, 224)
decoder_input_ids = torch.tensor([[101]])

with torch.no_grad():
    outputs = model(input_ids=input_ids, seg_data=seg_data, image=image, decoder_input_ids=decoder_input_ids)

# autoregressive decoding
print("Testing generation...")
model_kwargs = {"seg_data": seg_data, "image": image}
outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
