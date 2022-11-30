from transformers import BitConfig, ViTHybridConfig, ViTHybridForImageClassification

backbone_config = BitConfig()
config = ViTHybridConfig(backbone_config=backbone_config)

model = ViTHybridForImageClassification(config)

for name, param in model.named_parameters():
    print(name, param.shape)