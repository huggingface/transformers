import torch

from transformers import ConvNextBackbone, ConvNextConfig


config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

model = ConvNextBackbone(config)

outputs = model(torch.randn(1, 3, 224, 224))

for i in outputs.feature_maps:
    print(i.shape)
