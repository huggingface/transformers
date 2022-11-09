from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig


backbone_config = ResNetConfig()
config = MaskFormerConfig(backbone_config=backbone_config)
config.save_pretrained(".")
model = MaskFormerForInstanceSegmentation(config)

for name, param in model.named_parameters():
    print(name, param.shape)
