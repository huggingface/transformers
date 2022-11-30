import torch

from transformers import BitBackbone, BitConfig


backbone_config = BitConfig(stem_type="same", layer_type="bottleneck", depths=(3, 4, 9), out_features=["stage3"])

model = BitBackbone(backbone_config)

outputs = model(torch.rand(1, 3, 224, 224))
print(outputs.feature_maps[-1].shape)
