import torch

from transformers import SwinConfig, SwinModel


configuration = SwinConfig()

model = SwinModel(configuration)

pixel_values = torch.randn((1, 3, 1024, 640))

outputs = model(pixel_values, output_hidden_states=True)

print(outputs.keys())

for i in outputs.reshaped_hidden_states:
    print(i.shape)
