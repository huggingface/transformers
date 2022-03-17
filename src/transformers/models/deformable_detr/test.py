import torch

from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


model = DeformableDetrForObjectDetection(DeformableDetrConfig()).to("cuda")

pixel_values = torch.randn(1, 3, 224, 224).to("cuda")

print("Dict output:")
dict_outputs = model(pixel_values)

print("tuple output:")
tuple_outputs = model(pixel_values, return_dict=False)

# print("OUTPUTS + HIDDEN STATES")

# print("Dict output:")
# dict_outputs = model(pixel_values, output_hidden_states=True)
# print("Dict output keys:", dict_outputs.keys())

# print("Tuple output:")
# tuple_outputs = model(pixel_values, output_hidden_states=True, return_dict=False)
