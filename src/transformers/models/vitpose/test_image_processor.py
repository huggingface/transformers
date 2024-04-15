import requests
from PIL import Image

from transformers import ViTPoseImageProcessor


url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_processor = ViTPoseImageProcessor()

boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]

inputs = image_processor(images=image, boxes=boxes, return_tensors="pt")

print(inputs.pixel_values.shape)
print(inputs.pixel_values.mean())
