import torch

from transformers import Swinv2Backbone, Swinv2Config


config = Swinv2Config(out_features=["stage1", "stage2", "stage3", "stage4"])

model = Swinv2Backbone(config)

pixel_values = torch.rand(1, 3, 224, 224)

outputs = model(pixel_values)

for i in outputs.feature_maps:
    print(i.shape)
