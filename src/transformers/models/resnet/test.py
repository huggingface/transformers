from transformers import ResNetConfig, ResNetBackbone
import torch

model = ResNetBackbone(ResNetConfig(out_features=["stem", "stage1", "stage2", "stage3", "stage4"]))

pixel_values = torch.rand(1, 3, 224, 224)

outputs = model(pixel_values)

for key, value in outputs.items():
    print(key, value.shape)