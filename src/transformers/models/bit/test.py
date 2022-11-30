from transformers import BitConfig, BitBackbone


backbone_config = BitConfig(stem_type="same", layer_type="bottleneck", depths=(3, 4, 9), out_features=["stage3"])

model = BitBackbone(backbone_config)

for name, param in model.named_parameters():
    print(name, param.shape)
