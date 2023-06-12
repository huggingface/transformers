import torch

from transformers import VitDetBackbone, VitDetConfig


config = VitDetConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

model = VitDetBackbone(config)

outputs = model(torch.randn(1, 3, 224, 224))

for i in outputs.feature_maps:
    print(i.shape)
