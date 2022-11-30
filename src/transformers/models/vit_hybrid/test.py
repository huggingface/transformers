from transformers import BitConfig, ViTHybridConfig, ViTHybridForImageClassification


backbone_config = BitConfig(stem_type="same", layer_type="bottleneck", depths=(3, 4, 9), out_features=["stage3"])
config = ViTHybridConfig(backbone_config=backbone_config, image_size=384)

model = ViTHybridForImageClassification(config)

for name, param in model.named_parameters():
    print(name, param.shape)
