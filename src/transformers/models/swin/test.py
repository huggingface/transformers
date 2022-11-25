import torch

from transformers import SwinBackbone, SwinConfig


config = SwinConfig(
    output_hidden_states_before_downsampling=True,
    out_features=["stage1", "stage2", "stage3", "stage4"],
)

model = SwinBackbone(config)

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values, output_hidden_states=True)

for i in outputs.feature_maps:
    print(i.shape)
