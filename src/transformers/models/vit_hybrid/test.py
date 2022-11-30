import torch

from transformers import BitBackbone, BitConfig, ViTHybridConfig, ViTHybridForImageClassification


backbone_config = BitConfig(stem_type="same", layer_type="bottleneck", depths=(3, 4, 9), out_features=["stage3"])
config = ViTHybridConfig(backbone_config=backbone_config, image_size=384)

# model = BitBackbone(config=backbone_config)

# for name, param in model.named_parameters():
#     print(name, param.shape)

# outputs = model(torch.randn(1, 3, 384, 384))

# print(outputs.feature_maps[0].shape)

model = ViTHybridForImageClassification(config)

for name, param in model.named_parameters():
    print(name, param.shape)
