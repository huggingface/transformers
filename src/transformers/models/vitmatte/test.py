from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting


backbone_config = VitDetConfig(
    use_absolute_position_embeddings=False, use_relative_position_embeddings=True, out_features=["stage4"]
)

config = VitMatteConfig(backbone_config=backbone_config)

model = VitMatteForImageMatting(config)

for name, param in model.named_parameters():
    print(name, param.shape)
