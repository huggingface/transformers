from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, ResNetConfig, SwinConfig


# Instantiating from a config will always randomly initialize all the weights

# option 1: use default config
config = MaskFormerConfig()
model = MaskFormerForInstanceSegmentation(config)

# option 2: use custom Swin backbone
backbone_config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window12-384-in22k")
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)

# option 3: use custom ResNet backbone
backbone_config = ResNetConfig(depths=[3, 2, 2, 3])
config = MaskFormerConfig(backbone_config=backbone_config)
model = MaskFormerForInstanceSegmentation(config)

# Initializing using from_pretrained will load the weights from the pretrained model

model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-rcnn-base-vg-finetuned-ade20k")
