import torch
from PIL import Image

import requests
from transformers import DPTConfig, DPTFeatureExtractor, DPTForDepthEstimation


# feature_extractor = DPTFeatureExtractor(ensure_multiple_of=88)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# encoding = feature_extractor(image, return_tensors="pt")

# model = DPTForDepthEstimation(config)


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
