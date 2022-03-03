import torch

from transformers import DPTConfig, DPTForDepthEstimation


config = DPTConfig()
config.image_size = 384
config.hidden_size = 1024
config.intermediate_size = 4096
config.num_hidden_layers = 24
config.num_attention_heads = 16

model = DPTForDepthEstimation(config)

pixel_values = torch.randn((1, 3, 384, 384))

outputs = model(pixel_values)

print("Shape of logits:", outputs.logits.shape)
