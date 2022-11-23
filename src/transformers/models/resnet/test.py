from transformers import ResNetBackbone, ResNetConfig
import torch

model = ResNetBackbone(ResNetConfig(out_features=["stage2", "stage3"]))

pixel_values = torch.randn((1, 3, 224, 224))

outputs = model(pixel_values)

print(outputs.keys())