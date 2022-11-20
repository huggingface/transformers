import torch

from transformers import ConvNextBackbone, ConvNextConfig


model = ConvNextBackbone(ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"]))

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values)

for i in outputs.feature_maps:
    print(i.shape)
