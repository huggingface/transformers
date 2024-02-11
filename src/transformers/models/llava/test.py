import requests
from PIL import Image

from transformers import LlavaImageProcessor


image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

processor = LlavaImageProcessor()

pixel_values = processor([image, image], return_tensors="pt")["pixel_values"]

print(pixel_values.shape)
