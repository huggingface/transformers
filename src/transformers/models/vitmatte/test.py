from transformers import VitDetConfig, VitMatteConfig, VitMatteForImageMatting

backbone_config = VitDetConfig(out_features=["stage4"])

config = VitMatteConfig(backbone_config=backbone_config)

model = VitMatteForImageMatting(config)