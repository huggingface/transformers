from transformers import BeitConfig, BeitBackbone

config = BeitConfig(image_size=512, num_hidden_layers=24, hidden_size=1024, intermediate_size=4096, num_attention_heads=16, use_relative_position_bias=True)

model = BeitBackbone(config)

for name, param in model.named_parameters():
    print(name, param.shape)