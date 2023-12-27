import torch

from transformers import TinyVitBackbone, TinyVitConfig


config = TinyVitConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

model = TinyVitBackbone(config)

outputs = model(torch.ones(1, 3, 224, 224))

for i in outputs.feature_maps:
    print(i.shape)
