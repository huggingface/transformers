import torch

from transformers import Dinov2Backbone, Dinov2Config


config = Dinov2Config(out_features=["stage1", "stage2", "stage3", "stage11"])

model = Dinov2Backbone(config)

outputs = model(torch.randn(1, 3, 224, 224))

for i in outputs.feature_maps:
    print(i.shape)
