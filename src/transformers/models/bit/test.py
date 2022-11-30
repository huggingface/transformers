from transformers import BitConfig, BitForImageClassification


config = BitConfig(layer_type="bottleneck", stem_type="same", conv_layer="std_conv_same")

model = BitForImageClassification(config)

for name, param in model.named_parameters():
    print(name, param.shape)
