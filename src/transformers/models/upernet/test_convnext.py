from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation


backbone_config = ConvNextConfig(
    out_features=["stage1", "stage2", "stage3", "stage4"],
)

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)

for name, param in model.named_parameters():
    print(name, param.shape)
