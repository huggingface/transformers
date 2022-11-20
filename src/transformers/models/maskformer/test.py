from transformers import MaskFormerSwinConfig, MaskFormerSwinBackbone
import torch

model = MaskFormerSwinBackbone(MaskFormerSwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"]))

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values)

print("hello world")
for i in outputs.feature_maps:
    print(i.shape)