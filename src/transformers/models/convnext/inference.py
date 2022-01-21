import torch

from transformers import ConvNextConfig, ConvNextModel


config = ConvNextConfig()
model = ConvNextModel(config)

pixel_values = torch.randn((1, 3, 224, 224))

pooled_output = model(pixel_values=pixel_values)

print("Shape of pooled output:", pooled_output.shape)
