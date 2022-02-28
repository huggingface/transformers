import torch

from transformers import GLPNConfig, GLPNForDepthEstimation


config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768, depths=[3, 8, 27, 3])
model = GLPNForDepthEstimation(config)

pixel_values = torch.randn(1, 3, 224, 224)

outputs = model(pixel_values)

print("Shape of logits:", outputs.logits.shape)
