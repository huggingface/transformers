import torch

from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


cuda1 = torch.device("cuda:1")

config = DeformableDetrConfig(with_box_refine=True, two_stage=True)

model = DeformableDetrForObjectDetection(config).to(cuda1)

pixel_values = torch.randn(1, 3, 224, 224).to(cuda1)

outputs = model(pixel_values)

for k, v in outputs.items():
    print(k, v.shape)

# print("OUTPUTS + HIDDEN STATES")

# print("Dict output:")
# dict_outputs = model(pixel_values, output_hidden_states=True)
# print("Dict output keys:", dict_outputs.keys())

# print("Tuple output:")
# tuple_outputs = model(pixel_values, output_hidden_states=True, return_dict=False)
