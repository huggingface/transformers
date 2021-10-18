import torch

from transformers import BeitConfig, BeitForSemanticSegmentation


config = BeitConfig(image_size=512)
model = BeitForSemanticSegmentation(config)
model.eval()

pixel_values = torch.randn((1, 3, 512, 512))

outputs = model(pixel_values)

print("Shape of logits:", outputs.logits.shape)
