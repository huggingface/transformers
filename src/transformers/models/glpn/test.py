from PIL import Image

import requests
from transformers import GLPNConfig, GLPNFeatureExtractor, GLPNForDepthEstimation


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = GLPNFeatureExtractor()

config = GLPNConfig(hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=64, depths=[3, 8, 27, 3])
model = GLPNForDepthEstimation(config)

# pixel_values = torch.randn(1, 3, 224, 224)

pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

print("Shape of pixel values:", pixel_values.shape)

outputs = model(pixel_values)

print("Shape of logits:", outputs.logits.shape)
